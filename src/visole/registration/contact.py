"""Foot-on-insole contact, as a Winkler elastic foundation.

What this is
------------
The insole is modelled as a bed of independent springs (a Winkler foundation):
each patch of insole resists compression in proportion to how far it is squashed,
and does not shear against its neighbours. This is the standard tractable model
in footwear biomechanics for a thin compliant layer under a much stiffer foot,
and it is the level of fidelity our inputs actually support.

Governing relations, per cell of the canonical plantar grid::

    indentation   d(u,v) = delta0 + tilt_u*(u-uc) + tilt_v*(v-vc) - s(u,v)
    pressure      p(u,v) = K(u,v) * max(d, 0)          [unilateral contact]
    equilibrium   sum p*A = W,  and the resultant acts through the target COP

``s`` is the plantar profile -- how far each part of the sole sits above the
lowest point of the foot -- so a high arch simply never reaches the insole,
which is why arch support exists and why the model reproduces heel/forefoot
loading without being told to.

``K`` is a series stiffness of insole and soft tissue::

    1/K = h / E_insole  +  1/k_tissue

Without the tissue term a rigid foot on a stiff insole produces absurd pressure
spikes at the single first-contact point. The heel pad is genuinely compliant,
so leaving it out would not be "conservative", it would be wrong.

What this is NOT
----------------
No shear, no in-plane coupling between cells, no bending of the insole plate, no
large deformation, no viscoelasticity. If two designs separate only marginally,
the honest reading is that this model cannot resolve them -- not that they are
equivalent. See ``docs/physics_model.md``.

Units are SI: metres, Pascals, Newtons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SOFT_TISSUE_STIFFNESS_PA_PER_M", "ContactSolution", "series_stiffness",
    "solve_contact", "plantar_profile_from_mesh", "dome_profile",
]

#: Effective plantar soft-tissue stiffness (Pa/m) -- pressure per unit indentation.
#:
#: Heel-pad compression is usually reported as roughly 100-200 N/mm over a ~20 cm^2
#: contact, which is 5e7-1e8 Pa/m; 2e7 is the soft end of that range, chosen so the
#: model does not understate how much the insole matters. It is a **model
#: parameter**, swept in experiments, not a measurement of these participants.
#:
#: This value is load-bearing for the conclusions. An earlier run used 2e6, which
#: permitted 21 mm of indentation -- enough to swallow the 33.7 mm arch, so the
#: entire foot registered as in contact and every design returned the same answer.
SOFT_TISSUE_STIFFNESS_PA_PER_M = 2.0e7


def series_stiffness(E_insole_pa: np.ndarray, thickness_m: float,
                     tissue_pa_per_m: float | None = SOFT_TISSUE_STIFFNESS_PA_PER_M
                     ) -> np.ndarray:
    """Combine insole and soft tissue as springs in series -> Pa per metre."""
    E = np.asarray(E_insole_pa, dtype=float)
    if thickness_m <= 0:
        raise ValueError("insole thickness must be positive")
    k_insole = np.divide(E, thickness_m, out=np.zeros_like(E), where=E > 0)
    if tissue_pa_per_m is None:
        return k_insole
    if tissue_pa_per_m <= 0:
        raise ValueError("tissue stiffness must be positive or None")
    with np.errstate(divide="ignore"):
        inv = np.where(k_insole > 0, 1.0 / np.maximum(k_insole, 1e-30), np.inf)
    return 1.0 / (inv + 1.0 / tissue_pa_per_m)


@dataclass
class ContactSolution:
    """Result of one contact solve."""

    pressure_pa: np.ndarray
    indentation_m: np.ndarray
    contact_mask: np.ndarray
    delta0_m: float
    tilt: tuple[float, float]
    applied_load_n: float
    cell_area_m2: float
    converged: bool
    residual: dict = field(default_factory=dict)

    #: k_insole / k_tissue where the foot is actually in contact. Well above 1
    #: means the soft tissue dominates the series and the insole design can barely
    #: influence the pressure distribution -- which is a physical result about the
    #: material, not a defect in the solver.
    stiffness_ratio: float = float("nan")

    @property
    def total_force_n(self) -> float:
        return float(self.pressure_pa.sum() * self.cell_area_m2)

    @property
    def contact_area_m2(self) -> float:
        return float(self.contact_mask.sum() * self.cell_area_m2)

    @property
    def peak_pressure_pa(self) -> float:
        return float(self.pressure_pa.max())

    def percentile_pressure_pa(self, q: float = 95.0) -> float:
        loaded = self.pressure_pa[self.contact_mask]
        return float(np.percentile(loaded, q)) if loaded.size else 0.0

    def center_of_pressure(self) -> tuple[float, float]:
        """COP in grid index coordinates; NaN when nothing is in contact."""
        p = self.pressure_pa
        tot = p.sum()
        if tot <= 0:
            return (float("nan"), float("nan"))
        iu, iv = np.indices(p.shape)
        return (float((p * iu).sum() / tot), float((p * iv).sum() / tot))

    def summary(self) -> dict:
        cu, cv = self.center_of_pressure()
        return {
            "stiffness_ratio_insole_over_tissue": self.stiffness_ratio,
            "peak_pressure_kpa": self.peak_pressure_pa / 1e3,
            "p95_pressure_kpa": self.percentile_pressure_pa(95) / 1e3,
            "mean_pressure_kpa": (self.total_force_n / self.contact_area_m2 / 1e3
                                  if self.contact_area_m2 > 0 else 0.0),
            "contact_area_cm2": self.contact_area_m2 * 1e4,
            "contact_cells": int(self.contact_mask.sum()),
            "total_force_n": self.total_force_n,
            "applied_load_n": self.applied_load_n,
            "force_error_pct": (100 * abs(self.total_force_n - self.applied_load_n)
                                / max(self.applied_load_n, 1e-9)),
            "cop_u": cu, "cop_v": cv,
            "indentation_max_mm": float(self.indentation_m.max() * 1000),
            "converged": bool(self.converged),
        }


def solve_contact(profile_m: np.ndarray, stiffness_pa_per_m: np.ndarray, *,
                  load_n: float, cell_area_m2: float,
                  target_cop: tuple[float, float] | None = None,
                  allow_tilt: bool = True, max_iter: int = 200,
                  stiffness_ratio: float = float("nan")) -> ContactSolution:
    """Press a rigid plantar profile into a Winkler foundation.

    Unknowns are the rigid-body descent ``delta0`` and two tilts. They are found
    by least squares on the equilibrium residuals, with the ``max(d, 0)``
    unilateral-contact nonlinearity evaluated inside the residual -- so cells lift
    off rather than being allowed to pull.

    ``target_cop`` (in grid index units) prescribes where the resultant acts.
    Supplying the *measured* COP is what makes a design comparison fair: every
    design then carries the same load through the same point, and only the
    distribution differs.
    """
    from scipy.optimize import least_squares

    s = np.asarray(profile_m, dtype=float)
    K = np.asarray(stiffness_pa_per_m, dtype=float)
    if s.shape != K.shape:
        raise ValueError(f"profile {s.shape} and stiffness {K.shape} must match")
    if load_n <= 0:
        raise ValueError("load must be positive")
    if np.any(K < 0):
        raise ValueError("stiffness must be non-negative")

    nu, nv = s.shape
    iu, iv = np.indices(s.shape)
    uc, vc = (nu - 1) / 2.0, (nv - 1) / 2.0
    du, dv = iu - uc, iv - vc

    def fields(params):
        delta0, tu, tv = params
        d = delta0 + tu * du + tv * dv - s
        d = np.maximum(d, 0.0)
        return d, K * d

    def residuals(params):
        _, tu, tv = params
        d, p = fields(params)
        force = p.sum() * cell_area_m2
        r = [(force - load_n) / load_n]
        tot = p.sum()
        if target_cop is not None and tot > 0:
            cu = (p * iu).sum() / tot
            cv = (p * iv).sum() / tot
            r += [(cu - target_cop[0]) / max(nu, 1), (cv - target_cop[1]) / max(nv, 1)]
        elif allow_tilt and tot > 0:
            r += [(p * du).sum() / tot / max(nu, 1), (p * dv).sum() / tot / max(nv, 1)]
        else:
            r += [tu, tv]
        return r

    # Start from a flat descent that would carry roughly the right load.
    k_ref = float(np.median(K[K > 0])) if np.any(K > 0) else 1.0
    guess0 = load_n / max(k_ref * cell_area_m2 * K.size, 1e-9) + float(np.median(s))
    x0 = np.array([guess0, 0.0, 0.0])
    if not allow_tilt and target_cop is None:
        sol = least_squares(lambda p: residuals([p[0], 0.0, 0.0])[:1], x0[:1],
                            max_nfev=max_iter)
        params = np.array([sol.x[0], 0.0, 0.0])
    else:
        sol = least_squares(residuals, x0, max_nfev=max_iter)
        params = sol.x

    d, p = fields(params)
    force = p.sum() * cell_area_m2
    converged = bool(sol.success) and abs(force - load_n) / load_n < 0.01
    return ContactSolution(
        pressure_pa=p, indentation_m=d, contact_mask=d > 0,
        stiffness_ratio=stiffness_ratio,
        delta0_m=float(params[0]), tilt=(float(params[1]), float(params[2])),
        applied_load_n=load_n, cell_area_m2=cell_area_m2, converged=converged,
        residual={"force_rel": float((force - load_n) / load_n),
                  "optimizer_success": bool(sol.success), "cost": float(sol.cost)},
    )


def dome_profile(shape: tuple[int, int], height_m: float = 0.02,
                 power: float = 2.0) -> np.ndarray:
    """A smooth dome, for validating the solver against Hertz-like behaviour."""
    nu, nv = shape
    u = np.linspace(-1, 1, nu)[:, None]
    v = np.linspace(-1, 1, nv)[None, :]
    r = np.sqrt(u ** 2 + v ** 2)
    return height_m * np.clip(r, 0, 1) ** power


def plantar_profile_from_mesh(vertices: np.ndarray, sole_vertex_idx: np.ndarray,
                              shape: tuple[int, int]) -> np.ndarray:
    """Height map of a foot's sole on the canonical grid, in metres.

    The sole vertices are projected onto their own long/short axes, binned to the
    grid, and reduced by **minimum z** per cell -- the lowest surface is what
    touches the insole. Zero is the lowest point of the whole sole, so the value
    is "how far above first contact this part of the foot sits".

    The canonical grid is a rectangle but a foot is not, so roughly a third of
    cells (the corners outside the foot outline) receive no sole vertex at all.
    Those are parked far above the foot so they can never come into contact,
    rather than being interpolated into contact they never had. Cells at or above
    ``profile.max()`` are exactly this set.
    """
    v = np.asarray(vertices, dtype=float)[np.asarray(sole_vertex_idx)]
    if v.size == 0:
        raise ValueError("no sole vertices supplied")

    extent = v.max(0) - v.min(0)
    long_ax, short_ax = int(np.argmax(extent[:2])), int(np.argmin(extent[:2]))
    z_ax = 2

    nu, nv = shape
    a = (v[:, short_ax] - v[:, short_ax].min()) / max(extent[short_ax], 1e-12)
    b = (v[:, long_ax] - v[:, long_ax].min()) / max(extent[long_ax], 1e-12)
    z = v[:, z_ax] - v[:, z_ax].min()

    ui = np.clip((a * (nu - 1)).round().astype(int), 0, nu - 1)
    vi = np.clip((b * (nv - 1)).round().astype(int), 0, nv - 1)

    prof = np.full(shape, np.inf)
    np.minimum.at(prof, (ui, vi), z)
    filled = np.isfinite(prof)
    if not filled.any():
        raise ValueError("projection produced no filled cells")
    # Unreached cells: park them above the tallest real point so they cannot contact.
    prof[~filled] = prof[filled].max() * 2.0 + 1e-3
    return prof - prof[filled].min()
