# Phase D — what does camera estimation actually cost?

Milestone 4 measured **2.93 mm** median chamfer on Foot3D **using the dataset's
own camera poses**. A phone capture does not come with camera poses. This phase
closes that gap by rendering a *known* mesh realistically in Blender and making
COLMAP work the cameras out itself.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python scripts/render_synthetic_feet.py -- --mesh <scan>/mesh.obj \
  --out <rgb_dir> --views 32 --samples 48
```
```bash
envs/focus/bin/python scripts/run_focus_synthetic.py --run <run_dir> --gt <scan>/mesh.obj
```

---

## Headline

| scan | views fused | cameras **given** | cameras **estimated** | ratio |
| --- | --- | --- | --- | --- |
| 0036 | 27 / 32 | 1.87 mm | 16.13 mm | 8.6× |
| 0039 | 30 / 31 | 1.93 mm | 12.10 mm | 6.3× |
| **mean** | | **1.90 mm** | **14.12 mm** | **7.4×** |

**Camera estimation costs roughly an order of magnitude in accuracy.** The
geometry stage is not the weak link — pose estimation is.

## The renders are genuinely in-distribution

The earlier flat clay renders were rejected by FOCUS's own predictor. The
Blender renders are not. Measured by foot-mask coverage, which is what FOCUS
uses to accept or reject a view:

| source | min | median | max | usable views |
| --- | --- | --- | --- | --- |
| clay renders (previous attempt) | 0.0% | — | 26% | 12/16 |
| **Blender renders** | **3.0%** | **7.1%** | **14.3%** | **12/12** |
| **real Foot3D photographs** | 1.9% | 4.7% | 10.1% | — |

The synthetic renders sit slightly *above* the real photographs on this measure,
so framing is not the limiting factor. Realism came from a procedural skin
material — Principled BSDF with subsurface scattering, noise-driven colour and
roughness, area lighting and a textured floor — which is how FOCUS's own SynFoot
training data is made. Cycles renders on the M1 Max GPU via Metal.

## Three things had to be fixed, and each was a real bug

**1. Exposure.** Fixed-wattage area lights blew every render pure white. Light
power has to scale with the *scene*: a 0.25 m foot is not a room. Energy now
scales as `size²`.

**2. `is_world_space` must be False when COLMAP estimates the cameras.** Upstream
sets `is_world_space = not make_predictions` and it is right to. COLMAP fixes the
scene only up to a similarity transform, so there is no canonical floor at z=0.
Leaving the flag True made FOCUS's absolute "cutoff below zero" filter delete
**93 % of the point cloud** (9572 → 682) before meshing.

**3. A perfect circular camera orbit is poorly conditioned for SfM.** With
constant elevation and radius, COLMAP left one view unplaced and the
reconstruction carried a long spurious smear; chamfer was 48.8 mm. Adding
deterministic variation in elevation, distance and focal length — closer to a
real hand-held capture — let the pipeline run to completion and cut the error to
16.1 mm.

## Alignment is disclosed, and is not the bottleneck

Structure-from-motion recovers geometry only up to a **similarity transform**;
the raw reconstruction came out ~111× larger than the metric ground truth. A
scale + rotation + translation fit is therefore applied before evaluation
(closed-form centroid/RMS-radius initialisation, then ICP — plain ICP diverges
when scales differ by two orders of magnitude).

To check the alignment was not itself the error, an independent brute force over
300 random rotations plus ICP was run: it reached **22.4 mm**, *worse* than the
pipeline's 16.1 mm. So the remaining error is in the reconstruction, not in how
it was aligned.

## Limits

1. **Synthetic renders, not photographs.** Procedural skin is closer to SynFoot
   than to a phone camera. Real capture adds motion blur, rolling shutter,
   auto-exposure and cluttered backgrounds.
2. **Two scans**, not a distribution. They agree in magnitude (6.3× and 8.6×),
   which is reassuring but nowhere near enough to put an error bar on.
3. **5 of 32 views were rejected** on coverage before COLMAP even ran.
4. **The similarity alignment removes any scale error**, so this measures shape
   fidelity, not absolute size. A real insole needs absolute scale — which a
   phone capture cannot supply without a fiducial or depth sensor. That is a
   requirement this experiment surfaces rather than solves.
5. FOCUS's frame-dependent cleanup filters are disabled in this mode, so spurious
   geometry that they would normally remove survives into the mesh.

## What it means for Visole

A capture protocol needs to specify more than "take some photos":

- **vary elevation and distance** — a fixed orbit is not enough for SfM;
- **at least ~15 views with the foot clearly visible**, consistent with the
  Milestone 4 finding that `corr(log usable views, error) = −0.783`;
- **include a scale reference** (a fiducial of known size), because SfM cannot
  recover absolute scale and an insole is a metric object.

Until camera estimation improves, the 2.93 mm figure should always be quoted as
*"with known cameras"*.
