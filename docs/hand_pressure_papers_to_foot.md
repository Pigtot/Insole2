# Three 2026 hand-pressure papers → what transfers to feet

Reviewed 2026-08-31. All three estimate **hand** pressure from vision. All were
checked against their own pages; metadata below is quoted, not recalled.

| Paper | ID | Core idea |
| --- | --- | --- |
| **WristPP: A Wrist-Worn System for Hand Pose And Pressure Estimation** — Xi, Ao, Wang, Gao, Zhang, Feng, Zhou | [arXiv:2603.00606](https://arxiv.org/abs/2603.00606), 2026-02-28 | A **body-worn close-range camera**. ViT + Hand-VQVAE codebook → mesh, plus an extrinsics-conditioned per-vertex pressure branch. 133k frames, 20 subjects. MPJPE 2.9 mm, contact IoU 0.712, volumetric IoU 0.618, pressure MAE 10.4 g |
| **EgoPressDiff: Multimodal Video Diffusion for Egocentric UV-Domain Hand-Pressure Estimation** | [project page](https://egopressdiff.github.io/) | Pressure predicted as a **UV map** (surface unwrapped to 2D, warped back to the mesh), by *video* diffusion. Explicitly motivated by prior work that "discretize[s] pressure signal and process[es] frames independently, leading to quantization errors and temporal inconsistencies." Volumetric IoU +34 % relative |
| **HOPE: Hand-Object Pressure Estimation from Monocular Videos** — Jeon, Kim, Joo (SNU / RLWRLD) | [arXiv:2608.06192](https://arxiv.org/abs/2608.06192), 2026-08-06 | Per-vertex pressure + contact on the MANO mesh from 10-frame clips. **Unifies three heterogeneous supervision sources** into one vertex space. Frozen DINOv3 + a tiny transformer (2 blocks, d=64). Contact-gated output. OpenTouch MAE 1.808 kPa vs PressureVision 1.93 |

---

## The one thing all three agree on

None of them predict in camera or sensor coordinates.

> HOPE formulates pressure estimation as *"a hand-centric video prediction
> problem"* rather than sensor-coordinate prediction, "enabling generalization
> across diverse objects and interactions."

WristPP regresses per-vertex on the hand mesh; EgoPressDiff predicts in UV space
and warps back. Three independent groups, one conclusion: **a body-centred frame
is what makes different sensors, viewpoints and subjects combinable at all.**

This is exactly the "canonical plantar coordinates" the Visole plan asked for,
and it is now implemented — see [below](#what-was-implemented-from-this).

---

## The asymmetry that does not transfer

**All three papers can see the contact surface.** An egocentric or wrist camera
sees the palm and fingers; monocular hand-object video sees much of the hand. So
they can use appearance *at the contact site*: skin blanching, soft-tissue
bulging, contact-patch shape.

**The plantar surface is never visible.** Not from a bad camera angle — by
definition. It is pressed against the floor. No camera placement fixes this, and
in our data the participants are shod as well.

This is a real, structural difference between hand-pressure and foot-pressure
estimation from vision, and it is the reason Visole cannot simply be
"PressureVision for feet". Vision-based foot pressure must be inferred from
kinematics, shape and dynamics rather than read off the contact surface.

Two ways the premise is restored, both future hardware:

1. **A transparent optical pedobarograph** (Betts et al. 1980) images the plantar
   surface *through glass from below*. That restores exactly the hand papers'
   setting, and would make EgoPressDiff's UV formulation and HOPE's per-vertex
   supervision directly applicable.
2. **WristPP's wrist camera → an ankle- or shoe-mounted camera.** It still would
   not see the sole, but it would see foot pose, shoe deformation and ground
   proximity at close range instead of ~50 px across a room. WristPP is
   effectively an argument that *moving the camera onto the body* beats
   improving the model.

---

## Transfer table

| Method | Purpose | Foot equivalent | Now? | Needs hardware? | Limitation |
| --- | --- | --- | --- | --- | --- |
| Body-centred (vertex/UV) prediction | Sensors, views and subjects become comparable | **Canonical plantar (u,v) frame**; later per-vertex on the FOCUS/FIND foot mesh | **Done** (2D canonical) | No | Foot mesh version waits on FOCUS |
| Unifying heterogeneous supervision (glove + planar + contact-only) into one space | Scarce calibrated data goes further | insole 32ch + future optical pedobarograph + contact-only from video, all lifted to canonical | **Partly** | Dense source is future | Only one real pressure source exists today |
| **Contact-gated output** `p = c ⊙ p̃` | "No contact ⇒ no pressure" enforced by architecture | Per-**foot** gate × 32-channel magnitude | **Done** | No | Our gate is per foot, not per vertex |
| Masking the pressure loss on contact-only samples | Train on data lacking pressure labels | GAITRite gives contact without pressure; STAND/SITDOWN give pressure without GAITRite | **Done** (`valid` masks) | No | — |
| Sparse taxels → mesh via geodesic Gaussian filtering | Spread sparse readings over the surface | IDW lift + explicit `support` mask | **Done** | No | 32 sensors is far sparser than a 16×16 glove |
| Frozen foundation backbone + tiny head (DINOv3 + 2 blocks, d=64) | Strong features without training a big encoder | Frozen DINOv2/v3 + small head on foot crops | Not yet | No | **The most Mac-relevant idea here** |
| Video diffusion over clips instead of per-frame classification | Fixes quantisation + temporal flicker | Temporal model over gait cycles | Not yet | No | Diffusion is far too heavy for this machine and this data |
| VQ-VAE codebook indices instead of continuous regression | Discrete targets are easier to learn | Analogous to our ordered pressure bins | Bins done | No | Codebook needs a shape model first |
| Body-worn close-range camera | Escapes the tiny-far-subject regime | Ankle/shoe camera | No | **Yes** | Future capture design |

---

## What EgoPressDiff says about our binning — a caveat we should hold

EgoPressDiff's stated motivation is that prior methods "discretize pressure
signal and process frames independently, leading to **quantization errors and
temporal inconsistencies**."

That is a direct criticism of the PressureVision++ ordered-bin approach Visole
adopted. It is also *measurable*, and we measured it: on P1/FP/1 the round-trip
error of our 9-bin representation is **MAE 105–130 counts** (median 6.7, p95
~600), because the upper bins are wide. So the criticism is real and quantified
in our own units.

The conclusion is not "abandon bins" — bins remain a more stable learning target
on a skewed signal, and the comparison against scalar regression is still a
planned experiment. The conclusion is that **the binning cost must be reported
alongside any binned result**, which `PressureBins.quantisation_error()` now
does. Diffusion is not the affordable answer here; a temporal model is.

---

## What was implemented from this

**1. `src/visole/pressure/canonical_foot.py`** — the shared body-centred frame.
Normalised plantar rectangle (u: medial→lateral, v: heel→toe), both feet in the
*same* frame. This only works because the left/right sensor correspondence was
established geometrically; the two insoles do not share numbering, so a naive
mapping would mirror one foot's anatomy.

Following HOPE's discipline, `lift_to_canonical()` returns a **`support` mask**
marking cells that actually have a nearby sensor (86 % of the grid at radius
0.16). Interpolation outside it is not a measurement and must be masked out of
any loss — the same way HOPE masks its pressure loss on contact-only samples.
Round-trip lift→sample costs **25.4 counts MAE, 1.2 % of range**.

`CanonicalField` also carries `source ∈ {measured_sensor, estimated_video,
optical_measured, synthetic}`, so the different provenances the plan anticipates
can never be silently mixed.

**2. Contact-gated prediction** (`fit_contact_gated`) — `p = c ⊙ p̃`.

This should suit feet *better* than hands. A hand is partly in contact much of
the time; a foot in swing is **exactly zero across all 32 channels for roughly
half of every walking clip**. A single linear map cannot represent a hard zero
and instead smears a compromise across both phases. Gating removes that failure
mode by construction.

Two deliberate deviations from HOPE: the gate is per **foot** rather than per
vertex (a foot is down or it is not), and the magnitude head is fitted on contact
frames only, so it learns load *distribution* without being dragged toward zero.

## What Visole must not claim from these papers

1. That any of their numbers are comparable to ours — they are kPa or grams on
   calibrated hand sensors; ours are uncalibrated insole counts.
2. That a method validated on visible bare hands transfers to an invisible,
   shod plantar surface without evidence.
3. That per-vertex foot pressure is achievable before a foot mesh with
   correspondence (FOCUS/FIND) is actually running.
