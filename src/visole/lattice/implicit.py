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

#: A disconnected island larger than this fraction of the main body is NOT
#: discarded automatically -- it is probably real structure (e.g. the inner
#: surface of a shelled part), and deleting it would silently ruin the geometry.
MAX_DISCARDABLE_FRACTION = 0.05

__all__ = [
    "gyroid", "schwarz_p", "schwarz_d", "TPMS_FUNCTIONS",
    "GradedLattice", "generate_graded_tpms", "generate_insole_lattice",
    "calibrate_density_vs_thickness",
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
    #: Volume of disconnected islands discarded so the part is printable. Nonzero
    #: means real material was thrown away -- worth looking at, not ignoring.
    discarded_fragment_mm3: float = 0.0

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
            "discarded_fragment_mm3": round(self.discarded_fragment_mm3, 3),
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


def generate_insole_lattice(footprint: np.ndarray, thickness_field: np.ndarray, *,
                            length_mm: float = 260.0, width_mm: float = 100.0,
                            height_mm: float = 20.0, cell_size_mm: float = 8.0,
                            resolution: int = 6, topology: str = "gyroid",
                            shell_mm: float = 0.0, rim_mm: float = 2.5,
                            keep_largest_component: bool = True):
    """Graded lattice clipped to a real foot outline, not a rectangular block.

    ``footprint`` is a boolean (nu, nv) mask of which canonical plantar cells are
    actually under the foot -- roughly a third of the canonical rectangle is
    corner space outside the outline, and filling it would waste material and
    make the part wrong. ``thickness_field`` is the matching level-set thickness
    per cell, from the pressure mapping.

    Both are supplied on the canonical grid and resampled onto the voxel grid, so
    the geometry inherits exactly the pressure map the rest of the pipeline used.
    """
    from skimage import measure

    fp = np.asarray(footprint, dtype=bool)
    tf = np.asarray(thickness_field, dtype=float)
    if fp.shape != tf.shape:
        raise ValueError(f"footprint {fp.shape} and thickness {tf.shape} must match")
    if not fp.any():
        raise ValueError("empty footprint")

    nx = max(8, int(round(width_mm / cell_size_mm * resolution)))
    ny = max(8, int(round(length_mm / cell_size_mm * resolution)))
    nz = max(4, int(round(height_mm / cell_size_mm * resolution)))

    # Nearest-neighbour resample of the canonical grid onto the voxel grid.
    ui = np.clip((np.arange(nx) / max(nx - 1, 1) * (fp.shape[0] - 1)).round().astype(int),
                 0, fp.shape[0] - 1)
    vi = np.clip((np.arange(ny) / max(ny - 1, 1) * (fp.shape[1] - 1)).round().astype(int),
                 0, fp.shape[1] - 1)
    mask2d = fp[np.ix_(ui, vi)]
    t2d = tf[np.ix_(ui, vi)]

    xs = np.linspace(0.0, width_mm, nx)
    ys = np.linspace(0.0, length_mm, ny)
    zs = np.linspace(0.0, height_mm, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    k = TWO_PI / cell_size_mm
    field = np.abs(TPMS_FUNCTIONS[topology](k * X, k * Y, k * Z))

    solid = field < t2d[:, :, None]
    solid &= mask2d[:, :, None]

    # A solid rim around the outline. Clipping a periodic lattice to a foot shape
    # shears the boundary cells into slivers that are connected to nothing -- a
    # smoke test produced 30 disconnected fragments. A perimeter wall ties them
    # all in, and is what a real insole would have anyway for edge durability.
    if rim_mm > 0:
        from scipy import ndimage

        px = max(1, int(round(rim_mm / width_mm * nx)))
        eroded = ndimage.binary_erosion(mask2d, np.ones((2 * px + 1, 2 * px + 1), bool))
        rim = mask2d & ~eroded
        solid |= rim[:, :, None]

    if shell_mm > 0:  # solid skin on the top and bottom faces
        n_shell = max(1, int(round(shell_mm / height_mm * nz)))
        solid[:, :, :n_shell] |= mask2d[:, :, None]
        solid[:, :, -n_shell:] |= mask2d[:, :, None]

    F = np.where(solid, -1.0, 1.0)
    Fp = np.pad(F, 1, mode="constant", constant_values=1.0)
    spacing = (width_mm / (nx - 1), length_mm / (ny - 1), height_mm / (nz - 1))
    verts, faces, normals, _ = measure.marching_cubes(Fp, level=0.0, spacing=spacing)
    verts -= np.asarray(spacing)

    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=True)
    mesh.remove_unreferenced_vertices()
    mesh, removed = _drop_negligible_fragments(mesh)

    # Clipping to a foot outline can still leave a genuinely disconnected island
    # that is too large to call negligible (a smoke test left one of 190 mm^3
    # against 144,000 mm^3 -- 0.13%, just over the auto-drop threshold). Floating
    # material must not go to a printer, so it is removed explicitly and the
    # discarded volume is REPORTED rather than the threshold being widened until
    # the problem disappears.
    discarded_mm3 = 0.0
    kept_back = 0
    if keep_largest_component:
        try:
            parts = mesh.split(only_watertight=False)
            if len(parts) > 1:
                parts = sorted(parts, key=lambda m: abs(float(m.volume)), reverse=True)
                biggest = abs(float(parts[0].volume))
                # NEVER silently delete substantial geometry. A solid top/bottom
                # skin seals the insole into a closed box, so marching cubes emits
                # an inner *and* an outer surface; the "second component" is then
                # comparable in size to the first and deleting it would destroy
                # the part. Only genuinely small islands are dropped; anything
                # larger is kept and surfaced through n_components.
                small = [m for m in parts[1:]
                         if abs(float(m.volume)) < MAX_DISCARDABLE_FRACTION * biggest]
                large = [m for m in parts[1:] if m not in small]
                discarded_mm3 = float(sum(abs(float(m.volume)) for m in small))
                removed += len(small)
                kept_back = len(large)
                keep = [parts[0]] + large
                mesh = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
        except Exception:  # pragma: no cover
            pass

    try:
        n_components = int(mesh.body_count)
    except Exception:  # pragma: no cover
        n_components = -1

    return GradedLattice(
        mesh=mesh, thickness_field=t2d,
        discarded_fragment_mm3=discarded_mm3,
        relative_density=float(solid.sum() / max(mask2d.sum() * nz, 1)),
        bounds_mm=((0.0, width_mm), (0.0, length_mm), (0.0, height_mm)),
        cell_size_mm=cell_size_mm, topology=topology, voxel_shape=(nx, ny, nz),
        watertight=bool(mesh.is_watertight), n_components=n_components,
        n_fragments_removed=removed,
    )


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
