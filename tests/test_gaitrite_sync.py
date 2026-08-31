"""Tests for GAITRite <-> clip time alignment and video motion features.

The foot mapping and the offset sign convention were derived empirically
(median 14.5 ms error vs 556 ms for the alternative). These tests pin that
result so a future refactor cannot silently invert it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from visole.data.insole_gaitrite import (  # noqa: E402
    GAITRITE_FOOT_TO_SIDE, NOMINAL_INSOLE_HZ, InsoleGaitRite,
)
from visole.models.video_features import (  # noqa: E402
    BASE_FEATURE_NAMES, add_temporal_context, context_feature_names,
)

DATA = REPO / "data/raw/insole_gaitrite/dataset"
requires_data = pytest.mark.skipif(not DATA.exists(), reason="dataset not downloaded")


def test_foot_mapping_is_pinned():
    assert GAITRITE_FOOT_TO_SIDE == {0: "left", 1: "right"}


# --- temporal context (no data needed) ------------------------------------
def test_context_stacks_the_requested_offsets():
    f = np.arange(20 * 3, dtype=float).reshape(20, 3)
    out = add_temporal_context(f, offsets=(-2, 0, 2))
    assert out.shape == (20, 9)
    assert np.array_equal(out[10, 3:6], f[10])       # offset 0 is the frame itself
    assert np.array_equal(out[10, 0:3], f[8])
    assert np.array_equal(out[10, 6:9], f[12])


def test_context_clamps_at_edges_rather_than_zero_padding():
    """Zero padding would inject fake stillness at clip boundaries."""
    f = np.arange(5 * 2, dtype=float).reshape(5, 2) + 1.0
    out = add_temporal_context(f, offsets=(-3, 0))
    assert np.array_equal(out[0, 0:2], f[0])
    assert np.all(out != 0)


def test_context_names_match_context_width():
    names = context_feature_names(offsets=(-2, 0, 2))
    assert len(names) == 3 * len(BASE_FEATURE_NAMES)
    assert names[0].endswith("-2")


# --- alignment against real data ------------------------------------------
@requires_data
def test_gaitrite_events_convert_into_clip_time():
    clip = InsoleGaitRite().get("P1", "FP", "1")
    ev = clip.gaitrite_events()
    assert ev is not None
    assert set(ev["side"]) <= {"left", "right"}
    inside = ev[ev["in_clip"]]
    assert len(inside) > 0
    duration = clip.insole_duration_s()
    assert inside["HeelOn_clip"].min() >= -0.2
    assert inside["HeelOn_clip"].max() <= duration + 0.2


@requires_data
def test_predicted_heel_strikes_land_on_insole_loading_onsets():
    """The core alignment claim, re-checked on real signals."""
    clip = InsoleGaitRite().get("P1", "FP", "1")
    ev = clip.gaitrite_events()
    errors = []
    for side in ("left", "right"):
        sig = clip.pressure(side, baseline_correct=True).sum(1)
        t = clip.pressure_time(side)
        loaded = sig > 0.15 * sig.max()
        onsets = t[1:][loaded[1:] & ~loaded[:-1]]
        rows = ev[(ev["side"] == side) & ev["in_clip"]]
        for on in rows["HeelOn_clip"]:
            if len(onsets):
                errors.append(float(np.min(np.abs(onsets - on))))
    assert errors
    # Should agree to within a couple of 64 Hz sample periods.
    assert np.median(errors) < 3.0 / NOMINAL_INSOLE_HZ


@requires_data
def test_contact_labels_track_measured_loading_inside_coverage():
    clip = InsoleGaitRite().get("P1", "FP", "1")
    for side in ("left", "right"):
        labels, valid = clip.contact_labels(side)
        sig = clip.pressure(side, baseline_correct=True).sum(1)
        assert labels.shape == sig.shape == valid.shape
        lab, s = labels[valid], sig[valid]
        # Inside coverage the separation is stark (measured ratio 33-49x).
        assert s[lab].mean() > 20 * max(s[~lab].mean(), 1.0)


@requires_data
def test_ignoring_the_coverage_mask_degrades_the_labels():
    """Guards the trap: outside the walkway a loaded foot looks like swing."""
    clip = InsoleGaitRite().get("P1", "FP", "1")
    labels, valid = clip.contact_labels("left")
    sig = clip.pressure("left", baseline_correct=True).sum(1)
    inside = sig[valid][~labels[valid]].max()
    everywhere = sig[~labels].max()
    assert everywhere > 5 * inside


@requires_data
def test_contact_labels_cover_a_plausible_stance_fraction():
    clip = InsoleGaitRite().get("P1", "FP", "1")
    for side in ("left", "right"):
        labels, valid = clip.contact_labels(side)
        frac = labels[valid].mean()
        assert 0.3 < frac < 0.8, f"{side} stance fraction {frac:.2f} is not gait-like"


@requires_data
def test_posture_clips_have_no_gaitrite_and_say_so():
    """STAND/SITDOWN carry no GAITRite files; the API must return None, not guess."""
    clip = InsoleGaitRite().get("P3", "STAND", "1")
    assert clip.gaitrite_path is None
    assert clip.gaitrite_events() is None
    assert clip.contact_labels("left") is None
