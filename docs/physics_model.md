# The physics model: what is modelled, and what is not

Visole's pressure-to-insole chain used to be geometry plus an assertion. This
document describes the mechanics that now sits underneath it, and — more
importantly — the boundaries of what it can support.

## The chain

```
measured pressure  ->  relative density  ->  effective modulus  ->  contact solve  ->  new pressure
   (raw counts)         DensityMapping        MEASURED by FEA       Winkler bed        (kPa, simulated)
```

Two of these steps are now measured rather than assumed:

| step | previously | now |
| --- | --- | --- |
| density → stiffness | not modelled at all | **numerical compression test**, `src/visole/lattice/stiffness.py` |
| foot → insole | not modelled at all | **Winkler foundation**, `src/visole/registration/contact.py` |

## 1. Lattice stiffness, measured

Every solid voxel of the gyroid becomes one trilinear hexahedral element, and
linear elasticity is solved with scikit-fem. The block is compressed along z with
quarter-symmetry side constraints — the numerical analogue of squashing a printed
coupon.

Measured on a 3×3×3-cell gyroid in TPU (E_s = 40 MPa, ν = 0.45):

> **E\*/Es = 0.862 · ρ^1.713   (R² = 0.9971)**

The exponent 1.71 sits between stretch-dominated (n≈1) and bending-dominated
(n≈2) — a *result*, not an input. The Gibson–Ashby form is used only to summarise
the measured points.

**Solver validation.** A solid block returns the modulus it was given to five
decimal places (E_eff/E_s = 1.00000). Tests also assert that E_eff scales exactly
linearly with E_s and is independent of applied strain — both properties linear
elasticity must have, and both of which a subtly broken assembly would violate.

**Convergence, honestly.** Repeating at half resolution moves the exponent by
0.037 (2%) but the prefactor by **30%**. So:

- the **scaling** with density is converged and trustworthy;
- the **absolute** stiffness carries roughly ±30% discretisation uncertainty;
- comparisons *between designs* share the same prefactor, so that error largely
  cancels — which is why the design comparison is more reliable than any single
  absolute stiffness value.

Voxel meshes also have stair-stepped surfaces, which stiffens the response. This
is part of that 30%.

## 2. Contact, as a Winkler foundation

The insole is a bed of independent springs. Per cell of the canonical plantar grid:

```
d(u,v) = δ₀ + θ_u(u-u_c) + θ_v(v-v_c) − s(u,v)      indentation
p(u,v) = K(u,v) · max(d, 0)                          unilateral contact
Σ p·A = W,  resultant acts through the target COP    equilibrium
1/K = h/E_insole + 1/k_tissue                        series stiffness
```

`s` is the plantar profile — how far each part of the sole sits above its lowest
point — taken from the FIND template's 19,340 labelled sole faces. A 33.7 mm arch
therefore simply never reaches a 10 mm insole, which is why the model reproduces
heel-and-forefoot loading without being told to.

**Validation gates** (all in `tests/test_contact.py`): a flat foot on a uniform
insole gives *exactly* uniform pressure; integrated pressure equals applied load
to within 1%; a dome concentrates load at its lowest point (Hertz-like); pressure
is never negative; lifted regions carry exactly zero.

### The soft-tissue term is load-bearing

`k_tissue = 2×10⁷ Pa/m`, the soft end of reported heel-pad stiffness
(~100–200 N/mm over ~20 cm²). It is a model parameter, not a measurement of these
participants.

An earlier run used 2×10⁶, which permitted **21 mm** of indentation — enough to
swallow the arch, so the whole foot registered as in contact and every design
returned an identical answer. That looked like a null result and was actually a
parameter error. It is recorded here because the failure mode is instructive: in
a series-spring model, the *softer* element sets the behaviour, so getting it
wrong silently erases the effect you are trying to measure.

## What is NOT modelled

- **No shear or in-plane coupling.** Springs are independent; the insole cannot
  transmit load sideways or bend as a plate. Real insoles do both.
- **No large deformation.** Linear elasticity, small strain.
- **No viscoelasticity or hysteresis.** TPU is rate-dependent; walking is dynamic.
  Everything here is quasi-static.
- **No foot deformation.** The foot is rigid apart from one lumped tissue spring.
- **No friction, no shear stress on the skin** — which is clinically relevant to
  ulceration and entirely absent here.
- **No inverse dynamics.** Loads are imposed, not derived from body motion.

**If two designs separate only marginally, the correct reading is that this model
cannot resolve them — not that they are equivalent.**

## Units discipline

Simulated pressures are reported in **kPa** because the load (nominal 700 N) and
the material are chosen inputs. Measured insole data stays in **raw sensor
counts** forever — no calibration to kPa exists for it. The measurement supplies
only the normalised *shape* of the load and the centre of pressure; the assumed
body weight supplies the scale. A simulated field and a measured field are never
plotted on the same colour scale or compared numerically.
