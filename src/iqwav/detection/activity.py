"""Signal activity detection and active-region extraction for IQ captures.

This module segments a complex IQ capture into "active" and "noise-only"
regions using local windowed power, a noise-floor estimate from the
lower-power windows, and a configurable threshold relative to that floor.

Scope is intentionally narrow: this is activity detection and segmentation
only. It does not attempt modulation recognition, symbol-rate estimation,
CFO estimation, timing recovery, demodulation, or multi-transmitter
separation. An active region is a contiguous span of elevated power; it may
contain zero, one, or several distinct transmissions.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..dsp.noise import signal_power

__all__ = ["ActiveRegion", "ActivityDetectionResult", "detect_activity"]


@dataclass(frozen=True)
class ActiveRegion:
    """One contiguous active region, in sample indices.

    ``end_sample`` is exclusive, so ``end_sample - start_sample`` is the
    region length in samples.
    """

    start_sample: int
    end_sample: int
    start_time_s: float
    duration_s: float
    average_power: float
    peak_power: float


@dataclass(frozen=True)
class ActivityDetectionResult:
    """Result of activity detection over an IQ capture.

    ``window_powers`` holds the mean power of each analysis window in
    input order, and ``window_size`` is the number of samples per window
    (the final window may be shorter if the input length does not divide
    evenly). ``noise_floor_power`` is the estimated mean power of
    noise-only windows, and ``threshold_power`` is the absolute power
    level (``noise_floor_power * threshold_linear``) used to classify a
    window as active.
    """

    regions: tuple[ActiveRegion, ...]
    noise_floor_power: float
    threshold_power: float
    window_size: int
    window_powers: npt.NDArray[np.float64]
    sample_rate: float


def _validate_samples(samples: np.ndarray) -> npt.NDArray[np.complexfloating]:
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(f"samples must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    if not np.iscomplexobj(values):
        raise ValueError("samples must be complex IQ data.")
    return values


def _validate_sample_rate(sample_rate: float) -> float:
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive and finite, got {sample_rate!r}.")
    return float(sample_rate)


def _validate_window_size(window_size: int, num_samples: int) -> int:
    if not isinstance(window_size, (int, np.integer)) or isinstance(window_size, bool):
        raise ValueError(f"window_size must be an int, got {type(window_size).__name__}.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size!r}.")
    if window_size > num_samples:
        raise ValueError(f"window_size ({window_size}) must not exceed the number of samples ({num_samples}).")
    return int(window_size)


def _validate_threshold(threshold_db: float) -> float:
    if not math.isfinite(threshold_db):
        raise ValueError(f"threshold_db must be finite, got {threshold_db!r}.")
    return float(threshold_db)


def _validate_merge_gap(merge_gap_samples: int) -> int:
    if not isinstance(merge_gap_samples, (int, np.integer)) or isinstance(merge_gap_samples, bool):
        raise ValueError(f"merge_gap_samples must be an int, got {type(merge_gap_samples).__name__}.")
    if merge_gap_samples < 0:
        raise ValueError(f"merge_gap_samples must be non-negative, got {merge_gap_samples!r}.")
    return int(merge_gap_samples)


def _window_powers(values: np.ndarray, window_size: int) -> npt.NDArray[np.float64]:
    num_windows = math.ceil(values.size / window_size)
    powers = np.empty(num_windows, dtype=np.float64)
    for index in range(num_windows):
        start = index * window_size
        end = min(start + window_size, values.size)
        powers[index] = signal_power(values[start:end])
    return powers


def _estimate_noise_floor(window_powers: npt.NDArray[np.float64], noise_percentile: float) -> float:
    """Estimate the noise floor as the mean power of the lower-power windows.

    Windows at or below ``noise_percentile`` of the window-power
    distribution are treated as noise-only. If every window has zero
    power, the floor is zero.
    """
    cutoff = np.percentile(window_powers, noise_percentile)
    noise_windows = window_powers[window_powers <= cutoff]
    if noise_windows.size == 0:
        noise_windows = window_powers
    return float(np.mean(noise_windows))


def detect_activity(
    samples: np.ndarray,
    sample_rate: float,
    *,
    window_size: int = 256,
    threshold_db: float = 6.0,
    merge_gap_samples: int = 0,
    noise_percentile: float = 50.0,
) -> ActivityDetectionResult:
    """Detect active regions in a complex IQ capture from windowed power.

    The capture is split into non-overlapping windows of ``window_size``
    samples (the final window may be shorter). Each window's mean power is
    computed, a noise floor is estimated from the lower-power windows (at
    or below ``noise_percentile``), and windows whose power exceeds
    ``noise_floor_power * 10**(threshold_db / 10)`` are marked active.
    Active windows separated by a gap of at most ``merge_gap_samples``
    samples are merged into a single region.

    Args:
        samples: 1-D complex IQ array. Must be non-empty with only finite
            values. Not mutated.
        sample_rate: Sample rate in Hz. Must be positive and finite.
        window_size: Number of samples per analysis window. Must be a
            positive integer not exceeding the input length.
        threshold_db: Power threshold above the estimated noise floor, in
            dB, required to classify a window as active. Must be finite.
        merge_gap_samples: Maximum sample gap between two active windows
            for them to be merged into one region. Must be non-negative.
        noise_percentile: Percentile of window powers used to select the
            noise-floor reference windows. Must be in ``(0, 100]``.

    Returns:
        An ``ActivityDetectionResult`` describing detected regions and
        detection metadata. If no window exceeds the threshold, ``regions``
        is empty. If every window is active, a single region spans the
        entire input.

    Raises:
        ValueError: If any argument fails validation.
    """
    values = _validate_samples(samples)
    sample_rate = _validate_sample_rate(sample_rate)
    window_size = _validate_window_size(window_size, values.size)
    threshold_db = _validate_threshold(threshold_db)
    merge_gap_samples = _validate_merge_gap(merge_gap_samples)
    if not math.isfinite(noise_percentile) or not (0.0 < noise_percentile <= 100.0):
        raise ValueError(f"noise_percentile must be in (0, 100], got {noise_percentile!r}.")

    window_powers = _window_powers(values, window_size)
    noise_floor_power = _estimate_noise_floor(window_powers, noise_percentile)
    threshold_linear = 10.0 ** (threshold_db / 10.0)
    threshold_power = noise_floor_power * threshold_linear

    if noise_floor_power > 0.0:
        active_mask = window_powers > threshold_power
    else:
        # A zero noise floor means no reference level exists; classify any
        # non-zero-power window as active rather than dividing by zero.
        active_mask = window_powers > 0.0

    active_window_indices = np.flatnonzero(active_mask)

    regions: list[ActiveRegion] = []
    if active_window_indices.size > 0:
        run_start = active_window_indices[0]
        run_end = active_window_indices[0]
        for window_index in active_window_indices[1:]:
            # Actual sample gap between the end of the current run and the
            # start of the next active window (0 if they are adjacent).
            actual_gap_samples = (window_index - run_end - 1) * window_size
            if actual_gap_samples <= merge_gap_samples:
                run_end = window_index
            else:
                regions.append(_build_region(run_start, run_end, window_size, window_powers, values.size, sample_rate))
                run_start = window_index
                run_end = window_index
        regions.append(_build_region(run_start, run_end, window_size, window_powers, values.size, sample_rate))

    return ActivityDetectionResult(
        regions=tuple(regions),
        noise_floor_power=noise_floor_power,
        threshold_power=threshold_power,
        window_size=window_size,
        window_powers=window_powers,
        sample_rate=sample_rate,
    )


def _build_region(
    run_start: int,
    run_end: int,
    window_size: int,
    window_powers: npt.NDArray[np.float64],
    num_samples: int,
    sample_rate: float,
) -> ActiveRegion:
    start_sample = run_start * window_size
    end_sample = min((run_end + 1) * window_size, num_samples)
    region_powers = window_powers[run_start : run_end + 1]
    duration_samples = end_sample - start_sample
    return ActiveRegion(
        start_sample=int(start_sample),
        end_sample=int(end_sample),
        start_time_s=start_sample / sample_rate,
        duration_s=duration_samples / sample_rate,
        average_power=float(np.mean(region_powers)),
        peak_power=float(np.max(region_powers)),
    )