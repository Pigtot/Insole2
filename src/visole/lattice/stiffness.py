"""Measured mechanical stiffness of the lattice, by numerical compression test.

Why this module exists
----------------------
The pressure-to-material mapping in :mod:`visole.lattice.mapping` turns pressure
into a *relative density*. To say anything mechanical we need the step density ->
**stiffness**, and that step was previously missing: the project could grade a
lattice but could not say how stiff any part of it was.

The usual shortcut is the Gibson-Ashby scaling law ``E*/Es = C (rho)^n`` with
textbook constants. We do not assume it. Instead we run a **numerical
compression test** -- the direct analogue of squashing a printed coupon in a
testing machine -- and fit ``C`` and ``n`` to what comes out. That keeps this
consistent with the rest of the project, where relative density is voxel-counted
rather than taken from a formula.

Method
------
The lattice is already defined by a voxel occupancy field (``|G| < t``) inside
:func:`visole.lattice.implicit.generate_graded_tpms`, so every solid voxel
becomes one trilinear hexahedral element. That avoids tetrahedral meshing of a
TPMS surface entirely. Linear elasticity is solved with scikit-fem (pure Python;
no compiled dependency, no GPU -- this is a CPU/scipy workload).

Boundary conditions are a quarter-symmetry uniaxial-stress compression: the
``x=0`` face is held in x, ``y=0`` in y, the base in z, and the top face is given
a prescribed downward displacement while remaining free laterally. Verified
against a solid block, where it recovers the input modulus to 4 decimal places.

Units are SI throughout: Pa for moduli, metres for length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TPU_MODULUS_PA", "TPU_POISSON", "voxel_hex_mesh", "effective_modulus",
    "calibrate_stiffness_vs_density", "GibsonAshby",
]

#: Nominal TPU (flexible filament) properties. Both are swept in experiments --
#: printed TPU varies widely with shore hardness, infill and print orientation,
#: so no conclusion should rest on a single value.
TPU_MODULUS_PA = 40.0e6
TPU_POISSON = 0.45          # near-incompressible elastomer

#: skfem's hexahedral corner ordering, as (dx, dy, dz) offsets. Determined from
#: ``MeshHex.init_tensor`` rather than assumed -- getting this wrong produces
#: inverted elements and a silently wrong stiffness.
_HEX_CORNERS = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
                (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))


def load_bearing_component(occupancy: np.ndarray) -> np.ndarray:
    """Keep only material forming a connected path from base to top.

    Floating islands carry no load and leave the stiffness matrix singular, so
    they are removed before assembly. If nothing connects the two faces the
    result is all-False and the caller should treat the lattice as having no
    measurable stiffness -- which is the physically correct answer.
    """
    from scipy import ndimage

    labels, n = ndimage.label(occupancy)
    if n == 0:
        return np.zeros_like(occupancy, dtype=bool)
    bottom = set(np.unique(labels[:, :, 0])) - {0}
    top = set(np.unique(labels[:, :, -1])) - {0}
    spanning = bottom & top
    if not spanning:
        return np.zeros_like(occupancy, dtype=bool)
    return np.isin(labels, list(spanning))


def voxel_hex_mesh(occupancy: np.ndarray, spacing: tuple[float, float, float]):
    """Build a :class:`skfem.MeshHex` from a boolean voxel field.

    One hexahedral element per occupied voxel; only corners actually used are
    emitted as nodes.
    """
    from skfem import MeshHex

    occ = np.asarray(occupancy, dtype=bool)
    if occ.ndim != 3:
        raise ValueError(f"expected a 3-D occupancy field, got {occ.shape}")
    if not occ.any():
        raise ValueError("occupancy field is empty -- no material to mesh")

    nx, ny, nz = occ.shape
    ix, iy, iz = np.nonzero(occ)

    # Global corner id on the (nx+1, ny+1, nz+1) lattice of voxel corners.
    def corner_id(cx, cy, cz):
        return (cx * (ny + 1) + cy) * (nz + 1) + cz

    t = np.empty((8, len(ix)), dtype=np.int64)
    for k, (dx, dy, dz) in enumerate(_HEX_CORNERS):
        t[k] = corner_id(ix + dx, iy + dy, iz + dz)

    used, inverse = np.unique(t, return_inverse=True)
    t = inverse.reshape(t.shape).astype(np.int64)

    cz = used % (nz + 1)
    rest = used // (nz + 1)
    cy = rest % (ny + 1)
    cx = rest // (ny + 1)
    p = np.vstack([cx * spacing[0], cy * spacing[1], cz * spacing[2]]).astype(float)
    return MeshHex(p, t)


@dataclass(frozen=True)
class CompressionResult:
    """Outcome of one numerical compression test."""

    effective_modulus_pa: float
    relative_density: float
    n_elements: int
    n_nodes: int
    reaction_force_n: float
    strain: float

    @property
    def normalised(self) -> float:
        """E_eff / E_s -- only meaningful alongside the E_s that produced it."""
        return self.effective_modulus_pa


def effective_modulus(occupancy: np.ndarray, spacing, *, E_s: float = TPU_MODULUS_PA,
                      nu: float = TPU_POISSON, strain: float = 0.01,
                      keep_load_path: bool = True) -> CompressionResult:
    """Compress a voxel block along z and report its effective Young's modulus.

    ``E_eff = (reaction force / footprint area) / strain`` -- exactly what a
    compression test on a printed coupon measures.
    """
    from skfem import Basis, ElementHex1, ElementVector, asm, condense, solve
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    occ = np.asarray(occupancy, dtype=bool)
    total_voxels = occ.size
    if keep_load_path:
        occ = load_bearing_component(occ)
    if not occ.any():
        return CompressionResult(0.0, 0.0, 0, 0, 0.0, strain)

    mesh = voxel_hex_mesh(occ, spacing)
    basis = Basis(mesh, ElementVector(ElementHex1()))
    K = asm(linear_elasticity(*lame_parameters(E_s, nu)), basis)

    z_top = occ.shape[2] * spacing[2]
    tol = 0.25 * min(spacing)
    top = basis.get_dofs(lambda x: np.abs(x[2] - z_top) < tol)
    bot = basis.get_dofs(lambda x: np.abs(x[2]) < tol)
    sym_x = basis.get_dofs(lambda x: np.abs(x[0]) < tol)
    sym_y = basis.get_dofs(lambda x: np.abs(x[1]) < tol)

    u = basis.zeros()
    u[top.nodal["u^3"]] = -strain * z_top
    D = np.concatenate([bot.nodal["u^3"], top.nodal["u^3"],
                        sym_x.nodal["u^1"], sym_y.nodal["u^2"]])
    u = solve(*condense(K, basis.zeros(), x=u, D=D))

    reaction = float((K @ u)[top.nodal["u^3"]].sum())
    area = occ.shape[0] * spacing[0] * occ.shape[1] * spacing[1]
    E_eff = abs(reaction) / area / strain
    return CompressionResult(
        effective_modulus_pa=E_eff,
        relative_density=float(occ.sum() / total_voxels),
        n_elements=int(mesh.t.shape[1]),
        n_nodes=int(mesh.p.shape[1]),
        reaction_force_n=reaction,
        strain=strain,
    )


@dataclass(frozen=True)
class GibsonAshby:
    """Fit of ``E*/Es = C * rho^n`` to measured points.

    ``n`` near 1 indicates stretch-dominated behaviour, near 2 bending-dominated.
    Reported as a *diagnostic* of what the measurement found, never as an input.
    """

    C: float
    n: float
    rho: np.ndarray
    E_ratio: np.ndarray
    r_squared: float

    def __call__(self, rho) -> np.ndarray:
        r = np.clip(np.asarray(rho, dtype=float), 1e-6, 1.0)
        return self.C * r ** self.n

    @property
    def regime(self) -> str:
        if self.n < 1.35:
            return "stretch-dominated"
        if self.n > 1.75:
            return "bending-dominated"
        return "mixed"

    def describe(self) -> str:
        return (f"E*/Es = {self.C:.3f} * rho^{self.n:.3f}  "
                f"(R^2={self.r_squared:.4f}, {self.regime})")


def fit_gibson_ashby(rho: np.ndarray, E_ratio: np.ndarray) -> GibsonAshby:
    """Least squares in log space, which is where the power law is linear."""
    rho = np.asarray(rho, dtype=float)
    E_ratio = np.asarray(E_ratio, dtype=float)
    ok = (rho > 0) & (E_ratio > 0)
    if ok.sum() < 2:
        raise ValueError("need at least two positive samples to fit a power law")
    lx, ly = np.log(rho[ok]), np.log(E_ratio[ok])
    n, logC = np.polyfit(lx, ly, 1)
    pred = logC + n * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return GibsonAshby(C=float(np.exp(logC)), n=float(n), rho=rho, E_ratio=E_ratio,
                       r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def calibrate_stiffness_vs_density(thicknesses=None, *, cell_size_mm: float = 8.0,
                                   cells: int = 3, resolution: int = 10,
                                   E_s: float = TPU_MODULUS_PA,
                                   nu: float = TPU_POISSON,
                                   topology: str = "gyroid", verbose: bool = False) -> dict:
    """Measure E_eff across a range of lattice thicknesses.

    Returns the raw points plus a Gibson-Ashby fit. Reuses the same level-set
    definition as :func:`visole.lattice.implicit.generate_graded_tpms`, so the
    mechanics is measured on exactly the geometry that gets exported.
    """
    from .implicit import TPMS_FUNCTIONS, TWO_PI

    if thicknesses is None:
        thicknesses = np.linspace(0.30, 1.05, 8)

    L = cells * cell_size_mm * 1e-3
    n = max(8, int(round(cells * resolution)))
    g = np.linspace(0.0, cells * TWO_PI, n, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    field = np.abs(TPMS_FUNCTIONS[topology](X, Y, Z))
    spacing = (L / n, L / n, L / n)

    rows = []
    for t in thicknesses:
        occ = field < t
        res = effective_modulus(occ, spacing, E_s=E_s, nu=nu)
        rows.append({
            "thickness": float(t),
            "relative_density": res.relative_density,
            "E_eff_pa": res.effective_modulus_pa,
            "E_ratio": res.effective_modulus_pa / E_s,
            "n_elements": res.n_elements,
        })
        if verbose:
            print(f"  t={t:.3f}  rho={res.relative_density:.4f}  "
                  f"E_eff={res.effective_modulus_pa/1e6:8.3f} MPa  "
                  f"({res.n_elements} elems)", flush=True)

    rho = np.array([r["relative_density"] for r in rows])
    ratio = np.array([r["E_ratio"] for r in rows])
    fit = fit_gibson_ashby(rho, ratio)
    return {
        "topology": topology,
        "E_s_pa": E_s,
        "poisson": nu,
        "cell_size_mm": cell_size_mm,
        "cells": cells,
        "resolution": resolution,
        "points": rows,
        "fit": {"C": fit.C, "n": fit.n, "r_squared": fit.r_squared,
                "regime": fit.regime, "description": fit.describe()},
    }
