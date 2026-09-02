# Visole: a working retrospective

Written 2026-09-02, after 15 commits, 8,253 lines of Python across 56 files, and
151 tests. Covering what was actually built, what it was built from, what went
wrong, and what should have been done differently.

The useful parts of this document are §3 and §5.

---

## 1. What was built, in order

**Milestone 0-1 — foundations.** Audited the machine (M1 Max, 64 GB, MPS verified
by a live matmul agreeing with CPU to 1.1e-7 — not just `is_available()`). Built
the repo skeleton and a single device abstraction so no module assumes CUDA.

**Milestone 2 — dataset audit.** Downloaded the Insole-GAITRite dataset (1.69 GB,
MD5-verified against Zenodo) and audited it *before* writing a loader. This
produced three corrections to the published description and one finding that
reframed the project (§3.1, §3.2).

**Sensor geometry.** The dataset ships sensor maps as SVG. Rather than hand-typing
coordinates, I recovered the 32 pad centroids by chaining 262 open Bézier segments
into closed loops. Validated by reproducing the device's own centre-of-pressure at
**r > 0.997** — an independent check that the extraction, the label assignment and
the left/right mirroring were all correct at once.

**Milestone 3 — video → plantar loading.** Resolved the dataset's undocumented
synchronisation empirically (§4.2), built metrics, then two baselines. The first
failed; the second worked; the gap between them is the most instructive result in
the project (§3.4). Final: **+0.514 ± 0.119 skill, 22/22 participants** under
leave-one-subject-out, 88% correct on which foot bears more load.

**Milestone 4 — FOCUS on Apple Silicon.** Ported a CUDA-oriented research codebase
to MPS. Built PyTorch3D from source, mapped which operations survive on GPU,
patched two upstream bugs. Result: **14/14 Foot3D scans at 2.93 mm median chamfer**,
474 real photographs in 3.2 minutes, with the neural stage **20× faster on MPS**
and numerically identical to CPU (3.2e-6).

**Physics.** Derived a lattice stiffness law by *numerical* compression test
(**E\*/Es = 0.862·ρ^1.713**, R² = 0.997 — **simulated, not measured**: nothing
physical was compressed), built a Winkler contact model, and used
them to answer the stiffen-vs-soften question the project had been carrying
unresolved since the start.

**Milestones 7 & 9 — integration.** Canonical plantar frame, then one command from
measured pressure to a printable, watertight, foot-shaped graded insole in 2 s.

---

## 2. Resources used

**Data**
- Insole-GAITRite (Zenodo 10.5281/zenodo.19662017, CC-BY-4.0) — 22 participants,
  794 clips, synchronised video + 32-channel insoles + GAITRite reference
- Foot3D `Multiview` — 474 calibrated photographs, 14 ground-truth foot scans

**Code**
- FOCUS / FOUND / FIND (Boyne & Cipolla) — foot reconstruction; FIND's template
  ships a **labelled plantar surface**, which turned out to be the key to
  registration
- PyTorch 2.13 (MPS), PyTorch3D 0.7.8 (built from source), COLMAP 4.1.1 (no CUDA)
- scikit-fem for FEA — pure Python, the only new dependency the physics needed
- Blender 5.2.1 headless, trimesh, scikit-image, scipy, torchvision Keypoint R-CNN

**Papers** — PressureVision and PressureVision++ (ordered bins, weak supervision),
UnderPressure (motion → force), FOCUS/FOUND/FIND, and three 2026 hand-pressure
papers the user supplied: HOPE (contact-gated prediction, per-vertex fields),
EgoPressDiff (UV-domain pressure), WristPP. Betts et al. 1980 for the optical
pedobarograph terminology.

Every citation was verified against provider APIs — GitHub, arXiv, Zenodo, PubMed
— before being written down. All checked out.

---

## 3. The major problems

### 3.1 The feet are in shoes

The project was framed as "PressureVision for feet": learn pressure from
appearance. Looking at actual video frames — before building anything — showed
participants wearing **their own footwear**. The plantar surface is never visible.

PressureVision works because bare skin blanches under load. That mechanism does
not exist here. The honest reframing is **UnderPressure** (motion → force), not
PressureVision. Everything downstream changed because of one look at the data.

### 3.2 The two insoles use different sensor numbering

Pad geometry mirrors exactly between left and right — but **not one of the 32
indices refers to the same anatomical site on both feet.** Right sensor 0 is left
sensor 15.

Concatenating `left[32] + right[32]` and assuming symmetry — which the original
plan's "output dimension: 64" implies — would have silently scrambled the anatomy,
trained fine, and produced meaningless per-sensor results. There would have been no
error message and no obvious symptom. A test now asserts the property.

### 3.3 Frame rate is not 30 fps

The published description says 30 fps. Most clips are **60 fps**, and the rest vary
between 28.92 and 30.0. Computing time as `frame / 30` would misalign video and
pressure by up to 2×.

### 3.4 I nearly published a wrong negative result

The first loading model — background subtraction, person bounding box, motion
energy — beat "predict the average" by **0.9%**, and two of four test participants
scored *negative*. The tempting write-up: "ordinary video cannot predict plantar
loading."

Before writing that, one diagnostic: *which single feature best separates stance
from swing?* The answer was **a different feature in every clip**. That is
coincidence, not signal. With pose keypoints the same diagnostic gave the **same**
features on top across clips, 5–15× stronger.

Final skill: **+0.009 → +0.514**. The first result was a statement about my
features, not about video. Publishing it as the latter would have been wrong, and
nothing in the metrics would have caught it.

### 3.5 Implausible results treated as findings

Three times a result looked like a finding and was actually a bug. Each was caught
by asking "is this physically plausible?" rather than by any test:

| Symptom | Real cause |
| --- | --- |
| Every insole design gave **identical** pressure — including solid, 50× stiffer | Soft-tissue stiffness 10× too soft; 21 mm of indentation swallowed the 33.7 mm arch |
| FOCUS scored 10.0 mm with a lopsided 3.3/16.7 error split | Comparing against **uncropped** ground truth — the scans include the leg, so the metric measured the missing leg. Correct protocol: 2.98 mm |
| Chamfer of **1.5 × 10¹⁵ mm** | Monocular SfM is scale-free; the reconstruction was 111× too large and ICP diverged |

The pattern: a suspiciously clean or suspiciously catastrophic number is a bug
signal. "All designs identical" was the most dangerous, because it reads as a
legitimate null result.

### 3.6 A cleanup routine that would have destroyed the part

My lattice generator discards disconnected fragments. With a solid top/bottom skin
the insole becomes a **sealed box**, so marching cubes emits an inner *and* an
outer surface — and the "second component" is comparable in size to the first. It
reported discarding **181,710 mm³, more than the entire part**.

Fixed by never auto-deleting anything above 5% of the main body, and reporting
discarded volume rather than widening the threshold until the problem disappeared.

### 3.7 Platform failures that don't announce themselves

- **PyTorch3D `knn_points` and `sample_points_from_meshes` segfault on MPS.** Not
  an exception — the process dies with no traceback. `PYTORCH_ENABLE_MPS_FALLBACK`
  does not help: it covers ATen operators, not third-party compiled kernels.
- **An upstream logic bug**: `if os.environ.get("KMP_DUPLICATE_LIB_OK", "True"):`
  is always true, because `"False"` is a truthy string. It disabled OpenMP
  duplicate tolerance and libomp `abort()`ed the process at the first Poisson
  reconstruction.
- **NumPy 2.0 removed `ndarray.tostring()`**, which FOCUS still calls.
- **A pytest hang I wrote myself**: re-emitting warnings while iterating the live
  `catch_warnings` list, which appends to the list being iterated.

### 3.8 Self-inflicted

- `brew reinstall x265` to fix a COLMAP link error **upgraded x265 and broke
  ffmpeg**, which the gait pipeline depends on. Lesson: reinstall the *dependent*,
  not the dependency.
- Committed **485 MB** of regenerable model intermediates. Untracked now, but git
  never forgets: `.git` is 339 MB for 139 useful files.

---

## 4. How the work was approached

**Verify before trusting.** Every dataset checksummed, every citation checked
against provider APIs, every Fusion API call validated against local stubs (which
caught a method that does not exist). The Zenodo record was read before the loader
was written.

**Validate solvers against known answers before believing outputs.** The FEA
recovers a solid block's modulus to 1.00000 and scales exactly linearly with the
material. The contact model gives *exactly* uniform pressure for a flat foot on a
uniform insole. Without those gates, §3.5 would have gone unnoticed.

**Let structure of the data dictate the method.** The synchronisation convention
was undocumented, so I tested two competing hypotheses against the data: one gave
14.5 ms median error (under one 64 Hz sample), the other 556 ms. A 38× gap is a
decision, not an estimate.

**Separate measured / simulated / unvalidated, everywhere.** Raw sensor counts are
never called kPa. Interpolated pixels are never called measurements. The pipeline
summary figure is colour-coded by evidence status for exactly this reason.

**Keep the failures.** The 1a→1b comparison is more informative than 1b alone; the
tissue-stiffness bug is documented because the failure mode is instructive.

---

## 5. What I should have done better

**Look at the data first, always.** I checked what the videos contain reasonably
early — but *after* writing parts of the loader. The shod-feet finding invalidated
a framing assumption, and five minutes of looking at frames would have surfaced it
before any code was written.

**Validate the evaluation before reporting the metric.** I reported 10.0 mm chamfer
and only then noticed the ground truth included the leg. The lopsided 3.3/16.7
split was visible in the *first* output and should have stopped me immediately —
asymmetric error is diagnostic, and I read past it.

**Check size before committing.** 485 MB went in because I ran `git add -A` without
looking at what the pipeline had written. One `du -sh` would have prevented a
problem that now needs destructive surgery to undo.

**Test cheap before running expensive.** Pose extraction ran 105 minutes. I did not
first check throughput on 5 clips and extrapolate — which would have shown the
thermal throttling (9.6 → 2.0 fps) and prompted a smaller stride.

**Calibrate against the real thing earlier.** I generated synthetic renders, judged
them by eye, and only later measured that real Foot3D photos give 1.9–10.1% mask
coverage. That number should have been the *target* from the start rather than a
retrospective check.

**Express results in the units of the decision.** I reported the insole optimum as
a "1.5 MPa base material", which reads as "buy foam" and is unhelpful for someone
printing with filament. The user pushed back, and re-expressing the same physics as
Shore hardness showed **every viable option is a normal printable TPU**. The
underlying result never changed — only the framing — but the framing was the part
that mattered for the decision.

**Watch for boundary answers.** "Softer is always better" should have immediately
read as *missing physics*, not as a result. It took a prompt to make me add
bottoming-out, which is what makes an optimum exist at all. A result pinned to the
edge of a search space is a signal, and I should treat it as one automatically.

**Chain fewer shell commands.** A multi-heredoc command hung for 10 minutes and
lost a commit. Separate steps are slower to type and much faster to debug.

---

## 6. Where the project actually stands

**Measured:** dataset audit; video → loading (+0.514, 22/22 participants), and
that its skill is real for heel-vs-forefoot but **absent for medial-lateral**;
photos → 3D foot (2.93 mm, 14/14 scans); camera estimation costs ~7×.

**Simulated:** the lattice stiffness law — reclassified from "measured" in
[evidence.md](evidence.md), because the compression test was numerical — and the
insole optimum built on it. Softening under high pressure lowers peak pressure;
the effect only matters when the insole is nearly as compliant as plantar tissue;
Shore 60A beats Shore 95A by 25%.

**Not validated at all:** anything about a real foot. No printed part, no pressure
sensor, no human testing.

The single most valuable next experiment is also the cheapest: **a compression test
on a printed lattice coupon.** It would replace two model constants (the Shore →
modulus conversion and the densification parameters) with measurements, and it
needs one spool of filament and a scale.
