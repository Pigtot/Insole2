# Lattice stack decision

**Decision: build Visole's own implicit (TPMS) engine as the primary path, with
CadQuery + LatticeQuery as an optional comparator. nTop is not a dependency.**

Taken 2026-08-30. Reversible — the pressure→lattice contract does not depend on
which generator consumes it.

## Why not nTop

nTop does not run on this Apple Silicon machine, and the project must not be
blocked on hardware it does not have. More importantly, "I imported a pressure
map into nTop" hides the interesting step. The transformation

```
pressure → material parameter → implicit field → mesh
```

*is* the contribution, so it should be inspectable code.

## Why an in-house implicit engine first

`src/visole/lattice/implicit.py` is ~200 lines of NumPy + scikit-image
marching cubes + trimesh, all already installed and all Apple-Silicon native. It
gives:

- the gyroid (and Schwarz P / D) as one-line level-set functions;
- a spatially varying thickness `t(x, y)` driven by any pressure field;
- **measured** relative density by voxel counting, not a textbook formula;
- watertight STL export;
- a connectivity check.

Verified this session on a 48×48×12 mm coupon: four designs (solid, uniform,
graded-stiffen, graded-soften) all generate watertight and connected.

## Measurements that came out of building it

**Density is monotonic in thickness** (voxel-counted, gyroid sheet, one unit cell
at resolution 24):

| t | 0.05 | 0.31 | 0.58 | 0.84 | 1.11 | 1.37 |
| --- | --- | --- | --- | --- | --- | --- |
| ρ | 0.021 | 0.201 | 0.363 | 0.565 | 0.734 | 0.928 |

That curve is *inverted* to drive grading (`DensityToThickness`), so the mapping
rests on a measurement rather than an assumed ρ(t) relation.

**There is a connectivity floor at ρ ≈ 0.19.** Measured at cell size 8 mm:

| t | 0.20 | 0.25 | 0.30 | 0.35 | 0.50 |
| --- | --- | --- | --- | --- | --- |
| ρ | 0.127 | 0.160 | 0.190 | 0.223 | 0.320 |
| components | 2177 | 1514 | **1** | **1** | **1** |

Below the floor the sheet shatters into thousands of islands — a part that looks
fine by relative density but is unprintable. This directly changed the code: the
default `rho_min` was 0.12 (fragmenting) and is now **0.20**, and
`test_default_rho_min_is_above_the_connectivity_floor` guards it.

**Two density estimators disagree by ~5–10 %.** Voxel counting thresholds on the
sampling grid; mesh volume comes from an interpolated surface. Both are reported
in every summary and neither is tuned to match the other.

## Two density figures, one geometry

`relative_density_voxel_count` and `relative_density_from_mesh_volume` are both
in `GradedLattice.summary()`. Reporting only the flattering one would be a
quiet choice; reporting both makes the discretisation error visible.

## Comparator status

| Tool | Status | Note |
| --- | --- | --- |
| CadQuery | Verified as a **repository** (5.6k stars, active 2026-08-31); **not installed** | For the insole boundary/shell and STEP export, where a B-rep beats a mesh |
| LatticeQuery | Verified as a repository (Apache-2.0, active 2026-03-22); **not installed** | Comparator for BCC/FCC/diamond families |
| OpenSCAD | Not installed | Fallback for simple geometry only, never the main engine |
| nTop | Unavailable on this machine | Export contract kept (`docs/` + CSV/NPY/JSON) so a Windows box could consume the field |

The claims that OCP ships `macosx_11_0_arm64` wheels and that LatticeQuery
documents macOS installation are **unverified** — plausible, and testable at
Milestone 8. Visole's own engine depends on neither, so that answer cannot block
the project.

## The mapping direction stays an open hypothesis

`src/visole/lattice/mapping.py` implements `stiffen`, `soften` and `uniform` as
named, selectable hypotheses recorded in every export. Whether high-pressure
regions should become denser (spread the load) or less dense (cushion the peak)
is **not decided**, and generating geometry cannot decide it. The comparison set
A/B/C/D exists so a future physical test can.
