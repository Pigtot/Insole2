# What is proven, what is not, and what the errors are

Every claim Visole makes, labelled by how it is evidenced; how the errors compose
along the chain; and the cheapest experiment that would reduce the most
uncertainty next.

Written 2026-09-02. Update this file whenever a claim changes status.

---

## 1. The four labels

The project previously used three (measured / simulated / unvalidated). That
collapsed two different kinds of not-measured. Four:

| label | means | may be reported as |
| --- | --- | --- |
| **measured** | came off an instrument, or is an empirical evaluation against instrument data | raw sensor counts, mm, an accuracy |
| **predicted** | a machine-learning model's output for an input it did not train on | counts, always with the evaluation that bounds it |
| **simulated** | a physics model's output, from assumed material and load | kPa — the load and material are chosen inputs, so the units are real |
| **unvalidated** | asserted, or never tested against anything | nothing; say so |

Two rules that follow, and are enforced in code:

- **Measured insole values stay in raw sensor counts, forever.** No calibration
  to kPa exists for this dataset. `PressureField.source_type` carries this.
- **A claim about model *performance* is measured; a model *output* is
  predicted.** "The model scores +0.514 across 22 participants" is an empirical
  result. "This video's heel load is 1,800 counts" is a prediction. Both are
  honest; they are not the same kind of statement.

### The correction this forced

The lattice stiffness law **E\*/Es = 0.862·ρ^1.713 (R² = 0.997)** was previously
labelled *measured*, on the grounds that it came from a compression test. It came
from a **numerical** compression test — scikit-fem on a voxel hex mesh, with an
assumed base modulus. Nothing physical was compressed.

Under the four labels it is **simulated**. It is now labelled that way here, in
[reflection.md](reflection.md) and on the review page. The solver is validated
(it recovers a solid block's modulus to 1.00000 and scales linearly with the
material), which makes it a *good* simulation — not a measurement.

---

## 2. The claim ledger

### Measured

| claim | value | where |
| --- | --- | --- |
| Participants, clips | 22 participants, 794 clips | [gait_dataset_audit](../experiments/gait_dataset_audit/README.md) |
| Frame rate is not 30 fps | mostly 60; rest 28.92–30.0 | same |
| Left/right insoles share no sensor index | right 0 = left 15, bijective | `sensor_map.LEFT_RIGHT_CORRESPONDENCE` |
| Sensor geometry recovered correctly | reproduces device COP at r > 0.997 | `tests/test_sensor_map.py` |
| Video/pressure sync convention | 14.5 ms median vs 556 ms for the alternative | [gait_dataset_audit](../experiments/gait_dataset_audit/README.md) |
| Pose features beat motion features | +0.009 → +0.514 skill | [pressure_baseline](../experiments/pressure_baseline/README.md) |
| Loading model skill, LOSO | **+0.514 ± 0.119**, 22/22 participants | `outputs/loso_pose.json` |
| Which foot carries more load | 88.4 % of frames | same |
| Heel-vs-forefoot resolution is real | share skill +0.364, 22/22 vs null, p = 4.8e-07 | `outputs/spatial_ladder_pose.json` |
| **Medial-lateral resolution is absent** | share +0.135, **loses to an arbitrary partition in 18/22 folds**, p = 4.3e-03 | same |
| Foot reconstruction, given cameras | **2.93 mm** median chamfer, 14/14 scans | [focus_baseline](../experiments/focus_baseline/README.md) |
| Estimating cameras costs ~7× | ~14 mm with COLMAP | same |
| MPS is numerically sound here | agrees with CPU to 3.2e-6; 20× faster | [focus_mac_port](focus_mac_port.md) |

### Predicted

Anything the loading model outputs for a new video. Bounded by the row above it:
±0.119 skill across people, amplitude slope 0.46, longitudinal resolution only.

### Simulated

| claim | value | assumptions that carry it |
| --- | --- | --- |
| Lattice stiffness law | E\*/Es = 0.862·ρ^1.713, R² 0.997 | voxel hex mesh, assumed base modulus, linear elasticity |
| Softening beats stiffening | peak pressure down in every condition tested | Winkler foundation, no shear |
| Shore 60A beats 95A | **137 vs 183 kPa**, −25 % | Gent's Shore→modulus relation, ±30 % scatter |
| Grading only matters near tissue stiffness | — | soft-tissue stiffness 2.0e7 Pa/m, assumed |
| Best printable design | Shore 60A, 14 mm, soften | 700 N load, 14 mm cap chosen by judgement |

### Unvalidated

- **That any of this reduces pressure on a real foot.** No printed part, no
  pressure sensor, no human testing.
- The 14 mm thickness cap. The unconstrained optimum wanted 20 mm; 14 mm was
  chosen on intuition about wearability, and nothing tests it.
- Densification strain 0.60 and the 50× post-densification stiffening factor.
- That a printed lattice matches the FEA-derived stiffness law at all.

---

## 3. The error budget

The pipeline is eight links. **Two have measured error bars.**

| # | link | error | status |
| --- | --- | --- | --- |
| 1 | video → plantar loading | +0.514 ± 0.119 skill; slope 0.46; no medial-lateral axis | **measured** |
| 2 | photos → 3D foot | 2.93 mm given cameras, ~14 mm estimated | **measured** |
| 3 | pressure → canonical plantar frame | — | **not measured** |
| 4 | canonical frame → insole geometry | converged only at voxel res ≥ 5 (106.2 / 91.5 / 107.7 / 108.1 cm³ at res 3/4/5/6) | partially bounded |
| 5 | density → stiffness | fit R² 0.997, but base modulus ±30 % from Shore | **simulation on an assumption** |
| 6 | stiffness → contact pressure | Winkler: no shear, no plate bending | **model, unquantified** |
| 7 | design → printed part | — | **not measured** |
| 8 | printed part → pressure on a foot | — | **does not exist** |

**These cannot yet be composed into a single number, and saying so is the
result.** A budget with six unbounded links does not have a total. Quoting one
would be inventing precision.

Two propagations that *can* be reasoned about now:

- **Amplitude compression largely cancels.** The design uses the *shape* of the
  predicted field, normalised, and takes its scale from an assumed body weight.
  So the slope-0.46 error in link 1 mostly does not reach the insole. Shape error
  does.
- **The missing medial-lateral axis propagates directly.** Link 1 cannot resolve
  it, so no downstream stage can recover it. Any medial-lateral grading in the
  final insole would be an artefact of interpolation, not of measurement. This is
  now a hard constraint on what the design is allowed to claim.

---

## 4. Milestone status

### What has actually been proven

1. Ordinary video predicts *when* and *how much* plantar loading occurs, from
   limb kinematics, and generalises to unseen people (22/22).
2. That prediction carries **real heel-vs-forefoot information**, verified
   against a null control rather than assumed from a good aggregate score.
3. Calibrated photographs reconstruct a foot to 2.93 mm; camera estimation is the
   expensive part, not the network.
4. A research CUDA codebase runs correctly on Apple Silicon, with the failure
   modes documented.

### What has not been proven

1. **That RGB can estimate plantar pressure from the plantar surface.** The
   dataset is shod. This project has never tested that question and should stop
   being described as if it had.
2. That the model resolves medial-lateral load. It does not.
3. That a printed lattice behaves like the simulated one.
4. **That any insole produced here reduces pressure on a foot.** Nothing tests it.

### The cheapest experiment that would reduce the most uncertainty

**A compression test on printed lattice coupons.** One spool of TPU and a scale.

It closes link 5 and half of link 7 at once: it replaces the Shore→modulus
conversion (±30 %) *and* the densification constants with measurements, and it is
the only way to find out whether the FEA-derived exponent survives contact with a
real printer. Every simulated conclusion in §2 currently rests on numbers this
experiment would replace.

Second cheapest: **repeated smartphone scans of one foot**, to measure link 2's
repeatability on a phone rather than on Foot3D's calibrated rig.
