"""Research-grade implicit (TPMS) lattice generation.

The point of this module is transparency, not competing with commercial tools.
The whole transformation

    pressure field  ->  local material parameter  ->  implicit surface  ->  mesh

is a few lines of NumPy that anyone can read and re-run. Nothing is hidden
inside a black box, which is what makes it defensible as an experiment.

Method
------
A triply periodic minimal surface is the zero level set of a periodic function,
e.g. the gyroid

    G(x, y, z) = sin x cos y + sin y cos z + sin z cos x

A *sheet* solid is the region ``|G| < t``, where the thickness parameter ``t``
controls how much material there is. Making ``t`` a function of position,
``t = t(x, y)``, grades the structure. We evaluate ``F = |G| - t`` on a voxel
grid and extract ``F = 0`` with marching cubes.

Relative density as a function of ``t`` is *measured* by voxel counting
(:func:`calibrate_density_vs_thickness`) rather than assumed from a formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

TWO_PI = 2.0 * np.pi

__all__ = [
    "gyroid", "schwarz_p", "schwarz_d", "TPMS_FUNCTIONS",
    "GradedLattice", "generate_graded_tpms", "calibrate_density_vs_thickness",
]


def gyroid(x, y, z):
    """Gyroid: sin x cos y + sin y cos z + sin z cos x."""
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def schwarz_p(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)


def schwarz_d(x, y, z):
    return (np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z))


TPMS_FUNCTIONS: dict[str, Callable] = {
    "gyroid": gyroid,
    "schwarz_p": schwarz_p,
    "schwarz_d": schwarz_d,
}


@dataclass
class GradedLattice:
    """A generated lattice plus the provenance needed to reproduce it."""

    mesh: object                     # trimesh.Trimesh
    thickness_field: np.ndarray      # (ny, nx) thickness actually used
    relative_density: float          # measured by voxel counting
    bounds_mm: tuple
    cell_size_mm: float
    topology: str
    voxel_shape: tuple
    watertight: bool
    n_components: int = 1
    n_fragments_removed: int = 0

    @property
    def bounding_volume_mm3(self) -> float:
        (x0, x1), (y0, y1), (z0, z1) = self.bounds_mm
        return (x1 - x0) * (y1 - y0) * (z1 - z0)

    @property
    def relative_density_mesh(self) -> float | None:
        """Density from the meshed volume.

        This is a *different estimator* from the voxel count and the two do not
        agree exactly: voxel counting is a hard threshold on the sampling grid,
        while marching cubes interpolates the surface between voxels. Both are
        reported; neither is corrected to match the other."""
        if not self.watertight:
            return None
        return float(self.mesh.volume) / self.bounding_volume_mm3

    def summary(self) -> dict:
        return {
            "topology": self.topology,
            "bounds_mm": [list(b) for b in self.bounds_mm],
            "cell_size_mm": self.cell_size_mm,
            "voxel_shape": list(self.voxel_shape),
            "relative_density_voxel_count": round(self.relative_density, 4),
            "relative_density_from_mesh_volume": (
                round(self.relative_density_mesh, 4)
                if self.relative_density_mesh is not None else None),
            "thickness_min": round(float(self.thickness_field.min()), 4),
            "thickness_max": round(float(self.thickness_field.max()), 4),
            "n_vertices": int(len(self.mesh.vertices)),
            "n_faces": int(len(self.mesh.faces)),
            "watertight": bool(self.watertight),
            # >1 means the lattice fell apart into disconnected islands. Such a
            # part is unprintable and mechanically meaningless, however good its
            # density looks, so it is surfaced rather than left to be discovered
            # at the printer.
            "n_disconnected_components": int(self.n_components),
            "connected": bool(self.n_components == 1),
            "n_boundary_fragments_removed": int(self.n_fragments_removed),
            "volume_mm3": float(self.mesh.volume) if self.watertight else None,
        }

    def export_stl(self, path) -> str:
        self.mesh.export(path)
        return str(path)


def generate_graded_tpms(
    bounds_mm=((0.0, 40.0), (0.0, 40.0), (0.0, 10.0)),
    cell_size_mm: float = 8.0,
    thickness_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    constant_thickness: float = 0.6,
    resolution: int = 8,
    topology: str = "gyroid",
) -> GradedLattice:
    """Generate a graded TPMS coupon and return it as a mesh.

    Parameters
    ----------
    bounds_mm : ((x0,x1),(y0,y1),(z0,z1)) coupon extent in millimetres.
    cell_size_mm : the TPMS unit cell size; one period of the surface.
    thickness_fn : ``t(X, Y) -> array`` in the *dimensionless* units of the
        level set, evaluated on the coupon's (x, y) grid. If omitted, a constant
        thickness is used, giving a uniform lattice.
    resolution : voxels per unit cell along each axis. 8 is enough to see the
        topology; raise it for a printable mesh.
    """
    from skimage import measure

    if topology not in TPMS_FUNCTIONS:
        raise ValueError(f"unknown topology {topology!r}; have {sorted(TPMS_FUNCTIONS)}")
    if cell_size_mm <= 0 or resolution < 2:
        raise ValueError("cell_size_mm must be > 0 and resolution >= 2")

    (x0, x1), (y0, y1), (z0, z1) = bounds_mm
    if not (x1 > x0 and y1 > y0 and z1 > z0):
        raise ValueError("bounds must be increasing in every axis")

    n = [max(4, int(round((hi - lo) / cell_size_mm * resolution)))
         for lo, hi in ((x0, x1), (y0, y1), (z0, z1))]
    xs = np.linspace(x0, x1, n[0])
    ys = np.linspace(y0, y1, n[1])
    zs = np.linspace(z0, z1, n[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    k = TWO_PI / cell_size_mm  # one period per cell_size_mm
    field = TPMS_FUNCTIONS[topology](k * X, k * Y, k * Z)

    if thickness_fn is None:
        t2d = np.full((n[0], n[1]), float(constant_thickness))
    else:
        t2d = np.asarray(thickness_fn(X[:, :, 0], Y[:, :, 0]), dtype=float)
        if t2d.shape != (n[0], n[1]):
            raise ValueError(f"thickness_fn returned {t2d.shape}, expected {(n[0], n[1])}")
    if np.any(t2d < 0):
        raise ValueError("thickness must be non-negative")

    # Solid where |G| < t  ->  F = |G| - t is negative inside the material.
    F = np.abs(field) - t2d[:, :, None]

    solid_fraction = float((F < 0).mean())

    # Pad with a positive value so the isosurface closes at the coupon walls;
    # without this marching cubes leaves the boundary open and the mesh is not
    # a solid.
    Fp = np.pad(F, 1, mode="constant", constant_values=1.0)

    spacing = ((x1 - x0) / (n[0] - 1), (y1 - y0) / (n[1] - 1), (z1 - z0) / (n[2] - 1))
    verts, faces, normals, _ = measure.marching_cubes(Fp, level=0.0, spacing=spacing)
    # Undo the one-voxel pad offset and move to the requested origin.
    verts -= np.asarray(spacing)
    verts += np.array([x0, y0, z0])

    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=True)
    mesh.remove_unreferenced_vertices()

    mesh, n_fragments_removed = _drop_negligible_fragments(mesh)
    try:
        n_components = int(mesh.body_count)
    except Exception:  # pragma: no cover - depends on trimesh internals
        n_components = -1

    return GradedLattice(
        mesh=mesh,
        thickness_field=t2d,
        relative_density=solid_fraction,
        bounds_mm=bounds_mm,
        cell_size_mm=cell_size_mm,
        topology=topology,
        voxel_shape=tuple(n),
        watertight=bool(mesh.is_watertight),
        n_components=n_components,
        n_fragments_removed=n_fragments_removed,
    )


def _drop_negligible_fragments(mesh, rel_volume: float = 1e-3):
    """Remove specks left at the coupon walls by the padding.

    Marching cubes against the padded boundary can leave a few sub-cubic-mm
    slivers in corners. They are meshing artifacts, not part of the lattice, and
    they would otherwise make the connectivity count meaningless. Anything with
    at least ``rel_volume`` of the largest body's volume is kept -- a genuinely
    fragmented lattice still reports as fragmented.
    """
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:  # pragma: no cover
        return mesh, 0
    if len(parts) <= 1:
        return mesh, 0
    volumes = [abs(float(p.volume)) for p in parts]
    largest = max(volumes)
    keep = [p for p, v in zip(parts, volumes) if v >= rel_volume * largest]
    if len(keep) == len(parts):
        return mesh, 0
    import trimesh

    return trimesh.util.concatenate(keep), len(parts) - len(keep)


def calibrate_density_vs_thickness(thicknesses=None, topology: str = "gyroid",
                                   cell_size_mm: float = 8.0, resolution: int = 24) -> dict:
    """Measure relative density as a function of the thickness parameter.

    Voxel counting over whole unit cells. This turns "thicker t means denser"
    from an assumption into a measured curve, which is what the pressure->density
    mapping needs in order to be invertible.
    """
    if thicknesses is None:
        thicknesses = np.linspace(0.05, 1.5, 12)
    fn = TPMS_FUNCTIONS[topology]
    n = resolution
    g = (np.arange(n) + 0.5) / n * TWO_PI     # exactly one period, cell-centred
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    field = np.abs(fn(X, Y, Z))
    return {
        "topology": topology,
        "cell_size_mm": cell_size_mm,
        "resolution": n,
        "thickness": [float(t) for t in thicknesses],
        "relative_density": [float((field < t).mean()) for t in thicknesses],
    }
