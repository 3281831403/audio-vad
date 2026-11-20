"""Contract tests for :mod:`vad.detector`.

The initial repository has unimplemented stubs, so all 15 tests fail with
``NotImplementedError`` while still collecting normally.  Every test calls at
least one public stub; nothing here depends on random numbers, sleeping, or
the network.
"""

from __future__ import annotations

import math

import pytest

from vad.detector import (
    VadError,
    decide,
    detect_batch,
    frame_energy,
    hangover,
    segments,
    smooth,
    spectral_flatness,
    zero_crossing_rate,
)

PARAMS = {
    "energy_on": -20.0,
    "energy_off": -30.0,
    "zcr_on": 0.20,
    "zcr_off": 0.30,
    "flatness_on": 0.50,
    "flatness_off": 0.65,
    "energy_floor": -120.0,
    "min_speech_frames": 1,
    "min_silence_frames": 1,
    "hangover_frames": 0,
}


def _track(track_id: str) -> dict[str, object]:
    """Build one valid, deterministic track for batch tests."""
    return {
        "id": track_id,
        "samples": [0.5] * 480,
        "frame_len": 160,
        "hop": 80,
        "sample_rate": 16_000,
    }


def test_defect_01_frame_energy_honors_requested_frame_length() -> None:
    """A 10 ms frame must not be replaced by a fixed 20 ms frame."""
    samples = [0.5] * 12

    energies = frame_energy(samples, frame_len=4, hop=2)

    assert len(energies) == 5
    assert energies == pytest.approx([20.0 * math.log10(0.5)] * 5, abs=1e-9)


def test_defect_02_frame_energy_is_reported_in_db() -> None:
    """Linear amplitude must be converted to dB, not returned as a linear value."""
    samples = [0.1] * 8

    energies = frame_energy(samples, frame_len=8, hop=8)

    assert energies == pytest.approx([-20.0], abs=0.01)


def test_defect_03_zero_crossing_rate_is_computed_per_frame() -> None:
    """ZCR must be one value per frame, not one value for the whole signal."""
    samples = [
        1.0, 1.0, 1.0, 1.0,
        -1.0, 1.0, -1.0, 1.0,
        1.0, -1.0, 1.0, -1.0,
        1.0, 1.0, -1.0, -1.0,
    ]

    rates = zero_crossing_rate(samples, frame_len=4, hop=4)

    assert rates == pytest.approx([0.0, 1.0, 1.0, 1.0 / 3.0], abs=1e-12)


def test_defect_04_decision_uses_energy_hysteresis() -> None:
    """A frame between off and on must keep the previous speech state."""
    energy = [-18.0, -25.0, -18.0, -35.0, -18.0]
    zcr = [0.10] * 5
    flatness = [0.20] * 5

    mask = decide(energy, zcr, flatness, dict(PARAMS))

    assert mask == [True, True, True, False, True]


def test_defect_05_zcr_also_has_hysteresis() -> None:
    """A ZCR dip between on and off must not cut an ongoing speech run."""
    energy = [0.0] * 4
    zcr = [0.10, 0.25, 0.35, 0.10]
    flatness = [0.20] * 4

    mask = decide(energy, zcr, flatness, dict(PARAMS))

    assert mask == [True, True, False, True]


def test_defect_06_smooth_bridges_short_internal_silence() -> None:
    """A one-frame hole inside a sentence must be filled."""
    mask = [True, True, False, True, True]

    smoothed = smooth(mask, min_speech_frames=1, min_silence_frames=2)

    assert smoothed == [True, True, True, True, True]


def test_defect_07_segments_use_integer_sample_boundaries() -> None:
    """Long masks must map to exact integer samples, without float drift."""
    frame_count = 100_000
    hop = 160
    mask = [False] * (frame_count - 10) + [True] * 10

    result = segments(mask, hop=hop, sample_rate=16_000)

    expected_start = (frame_count - 10) * hop
    expected_end = frame_count * hop
    assert result == [(expected_start, expected_end)]
    assert all(isinstance(boundary, int) for seg in result for boundary in seg)


def test_defect_08_hangover_extension_is_capped() -> None:
    """Hangover must not turn an arbitrarily long silence into speech."""
    mask = [True] + [False] * 5

    held = hangover(mask, frames=2)

    assert held == [True, True, True, False, False, False]
    assert len(held) == len(mask)


def test_defect_09_batch_rejects_bad_track_atomically() -> None:
    """One invalid track must abort the batch instead of being skipped."""
    good = _track("good")
    bad = {**_track("bad"), "hop": 0}

    with pytest.raises(VadError):
        detect_batch([good, bad], dict(PARAMS))


def test_defect_10_all_zero_silence_is_not_speech() -> None:
    """Silence has ZCR 0, but low ZCR alone must not imply speech."""
    energy = [-240.0, -200.0]
    zcr = [0.0, 0.0]
    flatness = [0.0, 0.0]

    mask = decide(energy, zcr, flatness, dict(PARAMS))

    assert mask == [False, False]


def test_all_zero_signal_is_silent_through_pipeline() -> None:
    """An all-zero signal must yield an all-false mask from features onward."""
    samples = [0.0] * 240
    frames = [[0.0] * 160, [0.0] * 160]

    energy = frame_energy(samples, frame_len=160, hop=80)
    zcr = zero_crossing_rate(samples, frame_len=160, hop=80)
    flatness = spectral_flatness(frames)
    mask = decide(energy, zcr, flatness, dict(PARAMS))

    assert energy == pytest.approx([-240.0, -240.0], abs=1e-9)
    assert zcr == pytest.approx([0.0, 0.0], abs=1e-12)
    assert flatness == pytest.approx([0.0, 0.0], abs=1e-12)
    assert mask == [False, False]


def test_non_positive_hop_raises_vad_error() -> None:
    """Every hop-taking API must reject zero or negative hops."""
    samples = [0.25] * 8
    calls = [
        lambda: frame_energy(samples, frame_len=4, hop=0),
        lambda: zero_crossing_rate(samples, frame_len=4, hop=-1),
        lambda: segments([True, False], hop=0, sample_rate=16_000),
    ]

    for call in calls:
        with pytest.raises(VadError):
            call()


def test_min_speech_one_preserves_single_frame() -> None:
    """The shortest allowed speech run is one frame and must survive."""
    mask = [True, False, False, True]

    smoothed = smooth(mask, min_speech_frames=1, min_silence_frames=1)

    assert smoothed == mask


def test_segments_are_adjacent_without_gaps() -> None:
    """Adjacent speech frames merge into a gap-free half-open interval."""
    mask = [True, True, False, True, True]

    result = segments(mask, hop=160, sample_rate=16_000)

    assert result == [(0, 320), (480, 800)]


def test_batch_preserves_input_order() -> None:
    """A successful batch returns keys in the caller's track order."""
    tracks = [_track("b"), _track("a"), _track("c")]

    result = detect_batch(tracks, dict(PARAMS))

    assert list(result) == ["b", "a", "c"]
    assert all(isinstance(value, list) for value in result.values())
