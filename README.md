# Visole

Investigating whether an inexpensive computational workflow can connect
**RGB video → foot geometry → plantar loading → registered pressure → a locally
graded lattice insole**.

Research project for ISEF. Software phase, Apple Silicon, no physical hardware yet.

> **Nothing here has been physically validated.** The project separates what is
> *measured*, *estimated*, *reconstructed*, *simulated* and *inferred*, and says
> which is which everywhere. See [scientific integrity](#scientific-integrity).

---

## Status

| | |
| --- | --- |
| Machine audit | Apple M1 Max, 64 GB, macOS 26.5.2, PyTorch 2.13.0, **MPS verified** |
| Dataset | Insole-GAITRite, 22 participants, 794 clips, checksums verified |
| Tests | **151 passing** |
| FOCUS | **14/14** Foot3D scans, median **2.93 mm** with given cameras; **14.1 mm** when cameras are estimated |
| Baselines | Pose kinematics: **+0.514 ± 0.119** skill vs mean predictor, **22/22** participants (leave-one-subject-out) |

## Quick start

```bash
envs/core/bin/python scripts/doctor.py
```

```bash
envs/core/bin/python scripts/download_data.py insole_gaitrite --yes
```

```bash
envs/core/bin/python scripts/inspect_gait_dataset.py --max-clips 80 --clip P1/FP/1
```

```bash
envs/core/bin/python scripts/poc_pressure_bins.py --clip P1/FP/1
```

```bash
envs/core/bin/python scripts/generate_lattice_demo.py --resolution 10
```

```bash
envs/core/bin/python -m pytest tests/
```

## What exists

```
src/visole/
  compute/device.py       single source of truth for MPS/CPU; no module assumes CUDA
  data/registry.py        every download declared with size, checksum, licence, purpose
  data/insole_gaitrite.py dataset index and loader
  data/splits.py          participant-disjoint splits that raise on overlap
  pressure/sensor_map.py  sensor geometry from the dataset's own SVGs
  pressure/bins.py        nine ordered log-spaced pressure bins
  lattice/implicit.py     gyroid/Schwarz TPMS -> marching cubes -> watertight STL
  lattice/mapping.py      pressure -> density, as competing hypotheses
  lattice/stiffness.py    voxel->hex FEA; measures E*/Es = 0.862*rho^1.71
  registration/contact.py Winkler foundation: foot pressing into the insole
  visualization/          plots that label measured vs interpolated
```

## Three findings from the first session

**1. The two insoles use different sensor numbering.** The pad geometry is a
pixel-exact mirror, but not one of the 32 indices denotes the same anatomical
site on both feet. Treating `left[i]` and `right[i]` as homologous silently
scrambles the anatomy. `SensorMap.to_other_foot()` is the only correct mapping.

**2. Video frame rate is heterogeneous.** The published description says 30 fps;
most clips are 60 fps and the rest vary between 28.92 and 30.0. Assuming 30
misaligns video and pressure by up to 2×.

**3. Participants are shod and the plantar surface is never visible.** This
changes the project's framing from PressureVision (appearance → pressure, needs
bare skin) toward UnderPressure (motion → force). See the
[audit](experiments/gait_dataset_audit/README.md).

**4. Video does predict plantar loading — but only via limb kinematics.** Coarse
whole-body motion features scored +0.009 skill against a mean predictor and
*chance* on stance-vs-swing. Swapping in ankle/knee/hip pose raised that to
**+0.514 ± 0.119 skill** with **all 22 participants** beating the mean predictor
under leave-one-subject-out, **0.85** balanced contact accuracy, and it identifies
which foot carries more load on **88.4%** of frames. The first negative was a
statement about the features, not about video — see
[Milestone 3](experiments/pressure_baseline/README.md).

And from building the lattice engine: the gyroid sheet **fragments into
thousands of islands below ρ ≈ 0.19**, so the density floor is a measured
printability constraint, not a taste parameter.

## Documentation

| Document | Contents |
| --- | --- |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | What runs on MPS, CPU, external tools |
| [docs/sources.md](docs/sources.md) | Every external resource, verified against provider APIs |
| [docs/pressurevision_to_foot.md](docs/pressurevision_to_foot.md) | What transfers from PressureVision++, what does not |
| [docs/lattice_stack_decision.md](docs/lattice_stack_decision.md) | Why an in-house implicit engine instead of nTop |
| [experiments/gait_dataset_audit/](experiments/gait_dataset_audit/README.md) | Dataset audit and corrections |
| [reports/system_report.md](reports/system_report.md) | Machine audit |

## Scientific integrity

Rules enforced in code and tests, not just prose:

- Pressure values are **raw sensor counts**, never kPa — no calibration exists.
- Interpolated heatmaps are **display only** and never additional measurements.
- Splits are **by participant**; `Split` raises if a participant appears twice.
- The pressure→density direction (stiffen vs soften) is an **open hypothesis**;
  both are implemented and the choice is recorded in every export.
- Binning is lossy, so `quantisation_error()` reports the cost.
- Failed and negative results are kept.

## Data

Insole-GAITRite, [10.5281/zenodo.19662017](https://doi.org/10.5281/zenodo.19662017),
CC-BY-4.0. Not redistributed here; `scripts/download_data.py` fetches and verifies it.
