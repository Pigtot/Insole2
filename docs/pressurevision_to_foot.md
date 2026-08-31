# PressureVision++ → foot: what transfers, what does not

Reference: Grady et al., *PressureVision++: Estimating Fingertip Pressure from
Diverse RGB Images*, WACV 2024 ([arXiv:2301.02310](https://arxiv.org/abs/2301.02310)),
code [pgrady3/pressurevision2](https://github.com/pgrady3/pressurevision2) (MIT).
Predecessor: *PressureVision*, ECCV 2022 oral ([arXiv:2203.10385](https://arxiv.org/abs/2203.10385)).

The plan proposed adapting this method to feet. Having audited our data, the
honest summary is: **the supervision machinery transfers well; the visual premise
does not.** This document separates the two so neither gets over-claimed.

---

## The premise that does not transfer

PressureVision works because a **bare hand** occupies a large part of a
close-range frame, and pressure is visible through skin blanching, soft-tissue
deformation and contact-patch geometry.

In the Insole-GAITRite data ([audit](../experiments/gait_dataset_audit/README.md)):

| PressureVision++ condition | Our condition |
| --- | --- |
| Bare skin | **Shod** — participants wore their own footwear |
| Contact surface visible | Plantar surface **never** visible |
| Close range, hand fills frame | Whole-room view; foot ~50 px to a few hundred px |
| Static or slow motion | Walking, with motion blur on the swing foot |

So a foot model trained on this data does **not** estimate pressure from plantar
appearance. It relates *limb and shoe kinematics* to plantar loading — which is
the [UnderPressure](https://arxiv.org/abs/2208.04598) problem. Visole says so
explicitly rather than borrowing PressureVision's framing.

This is a property of *this dataset*, not of the idea. If a future capture shows
a bare foot on a transparent optical pedobarograph, the visual premise is
restored and this table should be revisited.

---

## Transfer table

| PressureVision++ method | Purpose | Foot equivalent | Now? | Needs hardware? | Scientific limitation |
| --- | --- | --- | --- | --- | --- |
| Crop the body part before inference | Removes background, normalises scale | Detect and crop foot/ankle from the walkway video | Partly | No | At the far end the crop is ~50 px of shoe; a detector is not yet built |
| Encoder–decoder (SE-ResNeXt-50 + FPN) | Dense per-pixel output | Lightweight encoder + **32-channel** head | Yes | No | We have 32 sensors, not a dense field — a dense decoder would be inventing data |
| Pressure as **9 log-spaced ordered bins** | Easier, more stable than regression | `PressureBins.log_spaced(9)` in raw counts | **Done** | No | Their edges are kPa; ours are uncalibrated counts. Not comparable across papers |
| Structure-aware loss (distant bin errors cost more) | Respects ordinality | `bin_distance_matrix()` provides the cost matrix | Representation done; loss pending | No | Untested on this data |
| Auxiliary **contact classifier** | Their largest single ablation gain | Contact / gait-phase head | Planned | No | Labels must come from GAITRite events, not visual intuition |
| Fully + **weakly** labelled data (0.5 M / 2.4 M frames) | Cheap labels scale the corpus | GAITRite `HeelOn`/`ToeOff` give *free* gait-phase labels on walking clips | Partly | No | Only 794 clips exist; there is no large weak corpus to add yet |
| Domain-adversarial alignment (gradient reversal) | Bridges lab → diverse images | Lab-insole domain vs. ordinary video | **Deferred** | No | We have **one room, one camera**. There is no second domain to align to. Premature here |
| Strong visual augmentation | Generalisation | Same, plus scale jitter for the 7× distance range | Yes | No | Cannot synthesise viewpoints the camera never saw |
| Participant-disjoint evaluation | Prevents identity leakage | `visole.data.splits`, enforced in code | **Done** | No | 22 participants is a small pool; expect wide error bars |
| ArUco calibration of image↔pressure frames | Registration | Sensor map validated against device COP (r > 0.997) | **Done** (sensor↔insole) | Yes for image↔foot | No fiducials in the released video; image↔insole registration is not currently possible |

### On the domain-adversarial head

The plan asks for a gradient-reversal domain head. We are deliberately **not**
building one yet. Domain adaptation aligns features across *different* domains;
this dataset has a single room, single camera and single viewpoint. Adding a
discriminator with nothing to discriminate adds parameters and a tuning burden
while answering no question. It becomes justified the moment a second capture
source exists, and the plan should carry it forward as conditional.

---

## What we took, concretely

Implemented in this session:

- `src/visole/pressure/bins.py` — nine ordered log-spaced bins over
  baseline-corrected counts, with a **measured** quantisation error
  (round-trip MAE ≈ 106–130 counts on P1/FP/1, median ≈ 6.7) so the cost of the
  representation is reported rather than assumed, plus an ordinal distance matrix.
- `src/visole/data/splits.py` — participant-disjoint splits that raise on overlap.
- `src/visole/pressure/sensor_map.py` — the pressure/geometry registration.

Deliberately deferred, with reasons: dense pressure decoding (we have 32
sensors), domain adaptation (one domain), any training on hand data.

## What Visole must never claim

1. That a model trained on hand images predicts foot pressure.
2. That interpolating 32 sensors produces additional measurements.
3. That raw counts are kPa.
4. That RGB alone *measures* plantar pressure — on this data it relates motion
   to loading, and even that is unproven until a baseline is evaluated.
5. That numbers here are comparable to PressureVision++'s, which are in kPa on
   a calibrated sensor with a different body part.
