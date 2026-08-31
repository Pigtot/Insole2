# Sources

Every external resource Visole depends on, with the exact use we make of it.

**Verification policy.** Nothing is listed here from memory. Each entry was checked
against the provider's own metadata API on the date shown: GitHub REST API for
repositories, the arXiv API for papers, the Zenodo REST API for datasets, and NCBI
E-utilities for PubMed records. Claims we have *not* verified are marked
`UNVERIFIED` rather than dropped, so they can be checked before being relied on.

Last verification sweep: **2026-08-30**.

---

## 1. Primary dataset

### Insole-GAITRite Dataset

| field | value |
| --- | --- |
| Title | An Open Insole-Based Plantar Pressure Dataset at Varying Cadences Compared Against GAITRite |
| Authors | Dobrescu, C. Cosmin; González Díaz, Iván; Carneros Prado, David; Fontecha Diezma, Jesús; Redondo Madruga, Marcos; Nugent, Chris; Zhang, Shuai |
| DOI | [10.5281/zenodo.19662017](https://doi.org/10.5281/zenodo.19662017) |
| Published | 2026-04-20 (version 1) |
| Licence | CC-BY-4.0 |
| Size | 1,686,755,134 B archive + two SVG sensor maps (1.687 GB total) |
| Companion tool | [MarcosRM02/GaitScope](https://github.com/MarcosRM02/GaitScope) (MIT) |

**Use in Visole:** the primary evidence base for RQ2 (does visual information predict
plantar loading?). It is the only resource we have that pairs synchronised video with
per-sensor plantar pressure.

**Verified contents** (from the Zenodo description and our own audit — see
[gait_dataset_audit](../experiments/gait_dataset_audit/README.md)):

- 3 walking conditions — `SP` (metronome 70 steps/min), `NP` (self-selected),
  `FP` (metronome 127 steps/min) — plus `STAND` and `SITDOWN`.
- Nominal rates: insoles 64 Hz, GAITRite 180 Hz, video 30 fps.
- Insole CSVs carry `PressureSensor 0`..`PressureSensor 31` plus accelerometer,
  gyroscope, `copX`, `copY`, `sumP`.
- `gaitrite_test.csv` / `gaitrite_testsets.csv` and `sync_auto.json` exist for
  walking clips only; STAND and SITDOWN have insole + video only.

**Constraints that shape every claim we may make from it:**

1. **Participants wore their own footwear.** The video shows *shod* feet. Any
   "RGB → pressure" result is about shoe/limb appearance and motion, not about the
   plantar surface. This is the single most important difference from
   PressureVision, where the bare hand's skin blanching is the visual signal.
2. **No subject-specific calibration was performed.** Channel units are not
   established as kPa. Visole labels them `raw sensor output` until proven otherwise.
3. **Participant IDs are pseudonymous and non-contiguous** (P1..P24 with gaps).
   23 were recorded originally; only those meeting quality criteria were released.
   Code counts participants; it never hardcodes 23.

---

## 2. Foot reconstruction (RQ1)

| Resource | Verified detail | Use in Visole |
| --- | --- | --- |
| [OllieBoyne/FOCUS](https://github.com/OllieBoyne/FOCUS) | MIT, last push 2025-04-17 | **Primary** multi-view foot reconstruction method |
| [FOCUS paper](https://arxiv.org/abs/2502.06367) | *FOCUS — Multi-View Foot Reconstruction From Synthetically Trained Dense Correspondences*, Boyne & Cipolla, arXiv 2025-02-10, 3DV 2025 | Method + evaluation protocol |
| [OllieBoyne/FOUND](https://github.com/OllieBoyne/FOUND) | MIT, last push 2025-02-18 | Synthetic training + surface-normal cues |
| [FOUND paper](https://arxiv.org/abs/2310.18279) | *FOUND: Foot Optimization with Uncertain Normals for Surface Deformation Using Synthetic Data*, Boyne et al., 2023-10-27, WACV 2024 | Prior stage of the same line of work |
| [OllieBoyne/FIND](https://github.com/OllieBoyne/FIND) | MIT, last push 2025-03-24 | Generative foot shape model (BMVC 2022) |
| SynFoot / Foot3D | Released via the FOUND/FIND projects | Synthetic training set and evaluation benchmark |

**Comparator (optional, only after the FOCUS baseline runs):**

| Resource | Verified detail | Use in Visole |
| --- | --- | --- |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Apache-2.0, 3.7k stars, last push 2026-08-07 | General feed-forward 3D reconstruction comparator |

> `UNVERIFIED`: the claim that MapAnything "added MPS inference support in 2026".
> Plausible given the push date but not confirmed against the repository. Check
> before relying on it; if false, MapAnything is a CPU-only comparator here.

---

## 3. Visual pressure estimation (methodological prior art)

| Resource | Verified detail | Use in Visole |
| --- | --- | --- |
| [facebookresearch/PressureVision](https://github.com/facebookresearch/PressureVision) | MIT, last push 2023-02-07 | Concept: appearance encodes contact force |
| [PressureVision paper](https://arxiv.org/abs/2203.10385) | *PressureVision: Estimating Hand Pressure from a Single RGB Image*, Grady et al., 2022-03-19, **ECCV 2022 oral** | Problem framing, evaluation ideas |
| [pgrady3/pressurevision2](https://github.com/pgrady3/pressurevision2) | MIT, last push 2023-12-26 | **Main methodological template** — see [pressurevision_to_foot.md](pressurevision_to_foot.md) |
| [PressureVision++ paper](https://arxiv.org/abs/2301.02310) | *PressureVision++: Estimating Fingertip Pressure from Diverse RGB Images*, Grady et al., 2023-01-05, **WACV 2024** | Weak supervision, ordered pressure bins, domain adaptation |

**Hard limits.** These are *hand* models. Visole never trains on hand images and
claims a foot result. PressureVisionDB is ~140 GB and is **not** downloaded during
the software phase. The PressureVision++ repository ships a checkpoint but notes its
dataset is not currently hosted, so the project must not depend on obtaining it.

## 4. Force from motion

| Resource | Verified detail | Use in Visole |
| --- | --- | --- |
| [InterDigitalInc/UnderPressure](https://github.com/InterDigitalInc/UnderPressure) | Last push 2023-05-14, licence not SPDX-tagged (**check before reuse**) | Kinematics → force baseline |
| [UnderPressure paper](https://arxiv.org/abs/2208.04598) | *UnderPressure: Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup*, Mourot et al., 2022-08-09, SCA 2022 | Evidence that motion carries loading information |

**Do not conflate** vertical ground reaction force (a scalar per foot) with a spatial
plantar pressure distribution. UnderPressure does the former.

## 5. Future hardware — optical pedobarography

| field | value |
| --- | --- |
| Title | Critical light reflection at a plastic/glass interface and its application to foot pressure measurements |
| Authors | Betts RP, Duckworth T, Austin IG, Crocker SP, Moore S |
| Journal | *Journal of Medical Engineering & Technology*, 1980 May; **4**(3):136–42 |
| PMID / DOI | 7401163 / [10.3109/03091908009161107](https://doi.org/10.3109/03091908009161107) |

Terminology: call the future device an **optical pedobarograph**. Do not call it an
"FTIR spectroscopy machine" — FTIR spectroscopy is an unrelated chemical-analysis
technique, and the optical mechanism here has not been experimentally verified by us.

## 6. Geometry and lattice generation

| Resource | Verified detail | Use in Visole |
| --- | --- | --- |
| [CadQuery/cadquery](https://github.com/CadQuery/cadquery) | 5.6k stars, actively maintained (push 2026-08-31), OCCT-based | Insole solid modelling, STEP/STL export |
| [jalovisko/LatticeQuery](https://github.com/jalovisko/LatticeQuery) | Apache-2.0, last push 2026-03-22 | Heterogeneous lattice families (BCC/FCC/diamond/TPMS) |
| scikit-image `marching_cubes` | Installed, v0.26.0 | Surface extraction for our own implicit lattice engine |
| trimesh | Installed, v5.0.0 | Mesh handling and STL export |

> `UNVERIFIED`: the specific claims that current OCP builds ship
> `macosx_11_0_arm64` wheels and that LatticeQuery documents macOS installation.
> Both are plausible and testable; they are checked at Milestone 8, and
> [lattice_stack_decision.md](lattice_stack_decision.md) records the outcome. Visole's
> own implicit engine (`src/visole/lattice/implicit.py`) deliberately depends on
> neither, so the project cannot be blocked by that answer.

nTop is **not** a dependency. See [lattice_stack_decision.md](lattice_stack_decision.md).

## 7. Compute

| Resource | Use |
| --- | --- |
| PyTorch 2.13.0 (MPS backend) | Primary compute; verified on this machine |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | Optional, only for new lightweight models written from scratch. Not used to reimplement upstream research. |
| COLMAP (Homebrew) | Not yet installed. Needed for FOCUS-SfM on uncalibrated images. |
