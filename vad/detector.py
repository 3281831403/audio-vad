"""Voice-activity-detection implementation for the ``audio-vad`` bug-fix exercise.

The public functions below implement the contracts stated in their own
docstrings.

Conventions used throughout this module
--------------------------------------
* Audio is a ``list[float]``.  ``frame_len`` and ``hop`` are expressed in
  **samples**; when the caller describes them in milliseconds, convert first
  with ``round(ms * sample_rate / 1000)``.  For example 10 ms at 16 kHz is
  160 samples, and a 5 ms hop is 80 samples.  A fixed 20 ms frame is wrong.
* Framing emits complete frames only, never zero-padded partial frames.  With
  ``n = len(samples)`` the normal frame count is ``0`` when ``n < frame_len``,
  otherwise ``1 + (n - frame_len) // hop``.
* Decision thresholds live in ``params`` and come in pairs.  ``*_on`` is the
  entry threshold; ``*_off`` is the (more permissive) threshold for staying in
  speech.  Energy uses a lower exit threshold, while ZCR and flatness use
  higher exit thresholds.  Each of the three features gets its own hysteresis.
* Segment boundaries are half-open sample intervals ``[start, end)`` computed
  with integer arithmetic.  Tests accept a boundary tolerance of one frame
  (``hop`` samples).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "VadError",
    "decide",
    "detect_batch",
    "frame_energy",
    "hangover",
    "segments",
    "smooth",
    "spectral_flatness",
    "zero_crossing_rate",
]


class VadError(ValueError):
    """Raised when audio, mask, or VAD parameters violate the contract."""


def frame_energy(
    samples: Sequence[float],
    frame_len: int,
    hop: int,
) -> list[float]:
    """Return one energy value in dB for every complete frame.

    Contract
    --------
    * ``frame_len`` and ``hop`` are sample counts and are independent: a
      10 ms frame with a 5 ms hop is legal, as is a 10 ms frame with a 10 ms
      hop.  At 16 kHz those are ``frame_len=160`` with ``hop=80`` or ``80``.
      Do not hard-code 20 ms (320 samples at 16 kHz): doing so misaligns the
      result for a 10 ms frame.
    * For each frame, ``rms = sqrt(sum(x * x) / frame_len)`` and the returned
      value is ``20 * log10(max(rms, 1e-12))``.  This is dBFS for normalized
      audio: a constant full-scale frame is about 0 dB and a constant 0.1
      frame is about -20 dB.  Linear energy (``rms`` or ``rms ** 2``) is not
      acceptable because the thresholds would no longer be comparable.
    * An all-zero frame is exactly ``-240.0`` dB.
    * Only full frames are emitted.  If fewer than ``frame_len`` samples are
      available, return ``[]``; no padding and no partial tail frame.
    * ``frame_len <= 0`` or ``hop <= 0`` raises :class:`VadError`.  A
      non-finite sample also raises :class:`VadError`.
    """

    if frame_len <= 0 or hop <= 0:
        raise VadError("frame_len and hop must be positive integers")
    for value in samples:
        if not math.isfinite(value):
            raise VadError("samples contain a non-finite value")

    frame_count = 1 + (len(samples) - frame_len) // hop if len(samples) >= frame_len else 0
    energies: list[float] = []
    for index in range(frame_count):
        start = index * hop
        frame = samples[start : start + frame_len]
        if all(value == 0.0 for value in frame):
            energies.append(-240.0)
            continue
        sum_squares = 0.0
        for value in frame:
            sum_squares += value * value
        rms = math.sqrt(sum_squares / frame_len)
        energies.append(20.0 * math.log10(max(rms, 1e-12)))
    return energies


def zero_crossing_rate(
    samples: Sequence[float],
    frame_len: int,
    hop: int,
) -> list[float]:
    """Return the per-frame zero-crossing rate in the range ``0.0..1.0``.

    Contract
    --------
    * Use the same complete-frame grid as :func:`frame_energy`: frame ``i``
      starts at ``i * hop`` and contains exactly ``frame_len`` samples.
      Computing one rate for the whole signal is wrong because it hides which
      frame contains speech.
    * A crossing is counted when consecutive samples fall on opposite sides
      of zero, using ``(a > 0) != (b > 0)``.  Samples equal to zero therefore
      count as non-positive, which makes an all-zero frame have rate ``0.0``
      rather than an undefined or positive value.
    * Divide the count by ``max(frame_len - 1, 1)`` and return a float.  The
      value is already in ``0.0..1.0``; clamp only floating-point round-off.
    * Only full frames are emitted; a short tail is ignored.  ``frame_len <=
      0`` or ``hop <= 0`` raises :class:`VadError`.
    """

    if frame_len <= 0 or hop <= 0:
        raise VadError("frame_len and hop must be positive integers")

    frame_count = 1 + (len(samples) - frame_len) // hop if len(samples) >= frame_len else 0
    rates: list[float] = []
    denominator = max(frame_len - 1, 1)
    for index in range(frame_count):
        start = index * hop
        frame = samples[start : start + frame_len]
        crossings = 0
        for position in range(frame_len - 1):
            if (frame[position] > 0) != (frame[position + 1] > 0):
                crossings += 1
        rate = crossings / denominator
        rates.append(min(1.0, max(0.0, float(rate))))
    return rates


def spectral_flatness(frames: Sequence[Sequence[float]]) -> list[float]:
    """Return a spectral-flatness proxy in ``0.0..1.0`` for each frame.

    The proxy is the ratio of the variance of first differences to the total
    variance of the frame:

    ``d[i] = frame[i + 1] - frame[i]``

    ``var_d = mean((d - mean(d)) ** 2)``

    ``var_x = mean((x - mean(x)) ** 2)``

    ``flatness = var_d / (var_x + var_d + 1e-12)``

    The result is clamped to ``[0.0, 1.0]``.  Because the denominator contains
    both terms, the value is scale-invariant and stays bounded.  A constant or
    all-zero frame has both variances equal to zero and therefore returns
    ``0.0``; a frame with fewer than two samples also returns ``0.0``.  Input
    frames are not modified and exactly one float is returned per input frame.

    The caller is responsible for slicing frames with the requested frame
    length and hop; this function deliberately knows nothing about sample
    rates or frame timing.  ``frames`` may be empty, in which case ``[]`` is
    returned.  A non-finite sample raises :class:`VadError`.
    """

    flatness_values: list[float] = []
    for frame in frames:
        for value in frame:
            if not math.isfinite(value):
                raise VadError("frame contains a non-finite value")
        if len(frame) < 2:
            flatness_values.append(0.0)
            continue

        mean_x = sum(frame) / len(frame)
        var_x = sum((value - mean_x) ** 2 for value in frame) / len(frame)

        differences = [
            frame[position + 1] - frame[position] for position in range(len(frame) - 1)
        ]
        mean_d = sum(differences) / len(differences)
        var_d = sum((value - mean_d) ** 2 for value in differences) / len(differences)

        flatness = var_d / (var_x + var_d + 1e-12)
        flatness_values.append(min(1.0, max(0.0, float(flatness))))
    return flatness_values


def _validate_decide_params(params: Mapping[str, Any]) -> dict[str, float]:
    """Validate the threshold mapping used by :func:`decide` and :func:`detect_batch`."""
    required = (
        "energy_on",
        "energy_off",
        "zcr_on",
        "zcr_off",
        "flatness_on",
        "flatness_off",
    )
    try:
        validated = {key: float(params[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise VadError(f"missing or invalid threshold parameter: {exc}") from exc

    floor_value = params.get("energy_floor", -120.0)
    try:
        validated["energy_floor"] = float(floor_value)
    except (TypeError, ValueError) as exc:
        raise VadError("invalid energy_floor parameter") from exc

    # Energy: the entry threshold is the higher (stricter) one.
    if validated["energy_on"] < validated["energy_off"]:
        raise VadError("energy_on must be greater than or equal to energy_off")
    # ZCR and flatness: the entry threshold is the lower (stricter) one.
    if validated["zcr_on"] > validated["zcr_off"]:
        raise VadError("zcr_on must be less than or equal to zcr_off")
    if validated["flatness_on"] > validated["flatness_off"]:
        raise VadError("flatness_on must be less than or equal to flatness_off")
    return validated


def decide(
    energy: Sequence[float],
    zcr: Sequence[float],
    flatness: Sequence[float],
    params: Mapping[str, Any],
) -> list[bool]:
    """Combine energy, ZCR, and flatness into a speech mask with hysteresis.

    All three input sequences must have the same length.  ``params`` must
    provide these keys:

    * ``energy_on`` / ``energy_off``: entry / stay thresholds in dB;
      ``energy_on`` is the stricter (higher) one.
    * ``zcr_on`` / ``zcr_off``: entry / stay thresholds for the zero-crossing
      rate; ``zcr_on`` is the stricter (lower) one.
    * ``flatness_on`` / ``flatness_off``: entry / stay thresholds for the
      spectral-flatness proxy; ``flatness_on`` is the stricter (lower) one.
    * ``energy_floor`` (optional, default ``-120.0``): a frame at or below
      this dB value is silence regardless of its ZCR or flatness.  This is
      what prevents an all-zero silent frame, whose ZCR is ``0.0``, from being
      classified as speech merely because low ZCR looks voiced.

    For a frame that is currently silence, enter speech only when all three
    entry tests pass.  For a frame that is currently speech, remain in speech
    while all three stay tests pass.  This independent dual-threshold
    hysteresis must apply to ZCR as well as energy; a single threshold makes
    the ZCR feature flip back and forth at a noise boundary.

    An empty input returns ``[]``.  Unequal lengths, missing parameter keys,
    or invalid threshold ordering raises :class:`VadError`.  The input
    sequences and ``params`` are not modified.
    """

    if not (len(energy) == len(zcr) == len(flatness)):
        raise VadError("energy, zcr, and flatness must have the same length")

    thresholds = _validate_decide_params(params)
    if not energy:
        return []

    floor = thresholds["energy_floor"]
    mask: list[bool] = []
    speaking = False
    for frame in range(len(energy)):
        # A frame at or below the energy floor is silence no matter what the
        # other features say; this is what rejects all-zero frames.
        if energy[frame] <= floor:
            speaking = False
            mask.append(False)
            continue

        if speaking:
            speaking = (
                energy[frame] >= thresholds["energy_off"]
                and zcr[frame] <= thresholds["zcr_off"]
                and flatness[frame] <= thresholds["flatness_off"]
            )
        else:
            speaking = (
                energy[frame] >= thresholds["energy_on"]
                and zcr[frame] <= thresholds["zcr_on"]
                and flatness[frame] <= thresholds["flatness_on"]
            )
        mask.append(speaking)
    return mask


def smooth(
    mask: Sequence[bool],
    min_speech_frames: int,
    min_silence_frames: int,
) -> list[bool]:
    """Remove tiny speech islands and bridge tiny internal silence gaps.

    The operation is deterministic and happens in this order:

    1. Every run of ``True`` shorter than ``min_speech_frames`` becomes
       ``False``.
    2. Every run of ``False`` that is shorter than ``min_silence_frames`` and
       is bounded by ``True`` on both sides becomes ``True``.  Leading and
       trailing silence are never filled because there is no speech on both
       sides.

    Bridging the short internal gap is what keeps a sentence from being cut
    into pieces.  ``min_speech_frames >= 1`` and ``min_silence_frames >= 0``;
    ``min_speech_frames=1`` must preserve a single-frame speech island.
    Negative values raise :class:`VadError`.  Return a new list of the same
    length; do not mutate ``mask``.
    """

    if min_speech_frames < 1 or min_silence_frames < 0:
        raise VadError("min_speech_frames must be >= 1 and min_silence_frames >= 0")

    result = [bool(value) for value in mask]
    if not result:
        return result

    # Encode the mask as (value, run_length) pairs for deterministic editing.
    runs: list[list[bool | int]] = []
    for value in result:
        if runs and bool(runs[-1][0]) == value:
            runs[-1][1] = int(runs[-1][1]) + 1
        else:
            runs.append([value, 1])

    # Step 1: drop speech runs shorter than min_speech_frames.
    for run in runs:
        if bool(run[0]) and int(run[1]) < min_speech_frames:
            run[0] = False

    # Step 2: bridge internal silence runs shorter than min_silence_frames.
    for index, run in enumerate(runs):
        if (
            not bool(run[0])
            and int(run[1]) < min_silence_frames
            and 0 < index < len(runs) - 1
            and bool(runs[index - 1][0])
            and bool(runs[index + 1][0])
        ):
            run[0] = True

    smoothed: list[bool] = []
    for value, length in runs:
        smoothed.extend([bool(value)] * int(length))
    return smoothed


def segments(
    mask: Sequence[bool],
    hop: int,
    sample_rate: int,
) -> list[tuple[int, int]]:
    """Convert a frame mask into half-open sample intervals.

    Each frame occupies one hop-sized cell: frame ``i`` covers samples
    ``[i * hop, (i + 1) * hop)``.  Consecutive ``True`` frames are merged into
    one interval, so a run from frame ``a`` through frame ``b`` inclusive
    becomes ``(a * hop, (b + 1) * hop)``.  The intervals are returned in time
    order and use only integer multiplication; accumulating floating-point
    seconds or repeatedly adding a rounded hop drifts on long audio and can
    shift a boundary by 10 ms.

    ``hop`` is in samples and ``sample_rate`` is positive and is used for
    validation and for expressing the hop duration in milliseconds; it does
    not change the integer boundary arithmetic.  A caller holding a hop in
    milliseconds must convert it to samples before calling.

    An all-``False`` or empty mask returns ``[]``.  A one-frame speech island
    returns exactly one ``(start, end)`` tuple with ``end - start == hop``.
    ``hop <= 0`` or ``sample_rate <= 0`` raises :class:`VadError`.  Tests
    allow a one-frame boundary tolerance.
    """

    if not isinstance(hop, int) or not isinstance(sample_rate, int):
        raise VadError("hop and sample_rate must be integers")
    if hop <= 0 or sample_rate <= 0:
        raise VadError("hop and sample_rate must be positive")

    intervals: list[tuple[int, int]] = []
    frame_count = len(mask)
    index = 0
    while index < frame_count:
        if not mask[index]:
            index += 1
            continue
        start_frame = index
        while index < frame_count and mask[index]:
            index += 1
        end_frame = index
        # Pure integer multiplication: no floating-point accumulation, so
        # boundaries never drift even on very long recordings.
        intervals.append((start_frame * hop, end_frame * hop))
    return intervals


def hangover(
    mask: Sequence[bool],
    frames: int,
) -> list[bool]:
    """Hold speech for at most ``frames`` frames after its last true frame.

    For every original ``True`` frame at index ``i``, output indices
    ``i .. min(i + frames, len(mask) - 1)`` are ``True``.  Equivalently, each
    speech tail is extended by exactly ``frames`` frames unless the end of the
    mask is reached first.

    The hold must be capped: a silent run of length ``L`` after speech may lose
    at most ``frames`` frames and must never be extended.  Output length always
    equals input length; no extra frames are appended.  Leading silence stays
    silent.  ``frames=0`` returns an unchanged copy, and ``frames < 0`` raises
    :class:`VadError`.  The input mask is not modified.
    """

    if frames < 0:
        raise VadError("frames must be non-negative")

    result = [False] * len(mask)
    if frames == 0:
        return [bool(value) for value in mask]

    last_true = -1
    for index, value in enumerate(mask):
        if value:
            last_true = index
        # Hold speech only while this frame is within the capped window of
        # the most recent true frame; beyond it silence stays silence.
        if last_true >= 0 and index - last_true <= frames:
            result[index] = True
    return result


def detect_batch(
    tracks: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, list[tuple[int, int]]]:
    """Detect speech segments for several tracks as one atomic operation.

    Each track is a mapping with:

    * ``id``: a unique, hashable identifier used as the result key;
    * ``samples``: a non-empty ``list[float]``;
    * ``frame_len``, ``hop``, ``sample_rate``: positive integers, with
      ``frame_len <= len(samples)``.

    ``params`` contains the thresholds documented by :func:`decide` plus
    ``min_speech_frames``, ``min_silence_frames``, and ``hangover_frames``.
    For each track the implementation computes frame energy, per-frame ZCR,
    flatness, the hysteretic mask, smoothing, hangover, and finally sample
    intervals with :func:`segments`.

    Return ``{track_id: [(start, end), ...]}`` in the same order as ``tracks``
    (Python dicts preserve insertion order).  The operation is all-or-nothing:
    validate every track and every parameter *before* computing any result.
    If any track is invalid, raise :class:`VadError` and return nothing; do
    not silently skip the bad track, because that leaves frames missing from
    the output.  Inputs are never modified.
    """

    # Validate parameters and every track before producing any output, so the
    # whole batch is all-or-nothing and a bad track can never be skipped.
    thresholds = _validate_decide_params(params)
    for key in ("min_speech_frames", "min_silence_frames", "hangover_frames"):
        try:
            int(params[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise VadError(f"missing or invalid batch parameter: {key}") from exc
        if int(params[key]) < 0:
            raise VadError(f"{key} must be non-negative")
    if int(params["min_speech_frames"]) < 1:
        raise VadError("min_speech_frames must be at least 1")

    seen_ids: set[Any] = set()
    for track in tracks:
        for key in ("id", "samples", "frame_len", "hop", "sample_rate"):
            if key not in track:
                raise VadError(f"track is missing required key: {key}")
        try:
            hash(track["id"])
        except TypeError as exc:
            raise VadError("track id must be hashable") from exc
        if track["id"] in seen_ids:
            raise VadError(f"duplicate track id: {track['id']!r}")
        seen_ids.add(track["id"])

        if not isinstance(track["samples"], list) or not track["samples"]:
            raise VadError("track samples must be a non-empty list")
        for key in ("frame_len", "hop", "sample_rate"):
            if not isinstance(track[key], int) or track[key] <= 0:
                raise VadError(f"track {key} must be a positive integer")
        if track["frame_len"] > len(track["samples"]):
            raise VadError("frame_len must not exceed the number of samples")
        for value in track["samples"]:
            if not math.isfinite(value):
                raise VadError("track samples contain a non-finite value")

    min_speech = int(params["min_speech_frames"])
    min_silence = int(params["min_silence_frames"])
    hold_frames = int(params["hangover_frames"])

    results: dict[Any, list[tuple[int, int]]] = {}
    for track in tracks:
        samples = track["samples"]
        frame_len = track["frame_len"]
        hop = track["hop"]

        energies = frame_energy(samples, frame_len, hop)
        rates = zero_crossing_rate(samples, frame_len, hop)
        frame_slices = [
            samples[index * hop : index * hop + frame_len] for index in range(len(energies))
        ]
        flatness = spectral_flatness(frame_slices)
        mask = decide(energies, rates, flatness, thresholds)
        mask = smooth(mask, min_speech, min_silence)
        mask = hangover(mask, hold_frames)
        results[track["id"]] = segments(mask, hop, track["sample_rate"])
    return results
