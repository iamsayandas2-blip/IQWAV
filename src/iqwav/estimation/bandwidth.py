"""Occupied-bandwidth estimation.

This module provides a single, narrowly-scoped estimator:
:func:`estimate_occupied_bandwidth`, which reports the smallest contiguous
band of FFT frequency bins whose accumulated spectral power reaches a
requested fraction of the total measured spectral power, using the
existing IQWAV FFT machinery (:func:`iqwav.dsp.magnitude_spectrum`).

Scope
-----
This is a **cumulative-power, bin-based occupied-bandwidth estimator**. It
assumes the sample rate is already known and simply reports the narrowest
contiguous run of FFT bins that contains at least ``percent_power`` percent
of the total spectral power present in the analyzed block.

It is explicitly **not**:

- a noise-floor-aware bandwidth estimator (it does not attempt to separate
  signal power from noise power; all spectral power in the block, including
  any noise, is treated as "signal" for the purposes of this cumulative-power
  definition),
- an automatic RF signal-activity detector (it does not decide *whether* a
  signal is present; it always returns a result for any valid, non-constant
  input),
- a general-purpose blind RF bandwidth estimator, CFO estimator, SNR
  estimator, noise-floor estimator, or modulation-aware measurement.

Those remain separate, later milestones (or are permanently out of scope for
this module).
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..dsp.spectrum import magnitude_spectrum

__all__ = ["OccupiedBandwidthEstimate", "estimate_occupied_bandwidth"]

_MIN_SAMPLES = 4
# Relative-to-signal threshold below which the AC (non-DC) content of a
# signal is treated as numerically zero, i.e. a constant/zero signal with
# no oscillating component to define an occupied band for. Matches the
# convention used by iqwav.estimation.spectral.
_CONSTANT_SIGNAL_RTOL = 1e-12


@dataclass(frozen=True)
class OccupiedBandwidthEstimate:
    """Result of a single cumulative-power occupied-bandwidth estimate.

    Attributes:
        bandwidth_hz: ``upper_frequency_hz - lower_frequency_hz``, in Hz.
        lower_frequency_hz: Lower edge, in Hz, of the smallest contiguous
            FFT-bin interval whose accumulated power reaches
            ``percent_power`` percent of the total measured spectral power.
            See the module and function docstrings for the exact bin-edge
            convention.
        upper_frequency_hz: Upper edge, in Hz, of that interval.
        percent_power: The requested ``percent_power`` value that produced
            this result (echoed back for traceability), as a percentage in
            ``(0, 100]``.
    """

    bandwidth_hz: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    percent_power: float


def _validate_bandwidth_args(
    samples: np.ndarray, fs: float, percent_power: float
) -> np.ndarray:
    """Validate estimator arguments and return samples as an ndarray."""
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    if not math.isfinite(percent_power):
        raise ValueError(
            f"percent_power must be finite, got {percent_power!r}."
        )
    if percent_power <= 0.0:
        raise ValueError(
            f"percent_power must be > 0, got {percent_power!r}."
        )
    if percent_power > 100.0:
        raise ValueError(
            f"percent_power must be <= 100, got {percent_power!r}."
        )
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if samples.size < _MIN_SAMPLES:
        raise ValueError(
            f"samples must contain at least {_MIN_SAMPLES} values to "
            f"estimate an occupied bandwidth, got {samples.size}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _reject_degenerate_signal(samples: np.ndarray) -> None:
    """Raise ValueError for zero-energy or constant (DC-only) signals.

    A zero/constant signal has no spectral content away from DC, so it has
    no meaningful occupied bandwidth under the cumulative-power definition
    used here (all of its power, if any, sits in a single DC bin). Rather
    than fabricate a zero-width (or otherwise arbitrary) bandwidth for such
    a signal, this is treated as an invalid input.
    """
    energy = float(np.sum(np.abs(samples) ** 2))
    if energy == 0.0:
        raise ValueError(
            "samples have zero energy; an occupied bandwidth cannot be "
            "estimated for a zero signal."
        )
    ac = samples - np.mean(samples)
    ac_level = float(np.max(np.abs(ac))) if ac.size else 0.0
    scale = max(1.0, float(np.max(np.abs(samples))))
    if ac_level <= _CONSTANT_SIGNAL_RTOL * scale:
        raise ValueError(
            "samples are constant and contain no oscillating component; a "
            "constant (DC-only) signal has no meaningful occupied "
            "bandwidth under the cumulative-power definition used here."
        )


def _smallest_window_reaching_power(
    power: npt.NDArray[np.float64], target: float
) -> tuple[int, int]:
    """Return the ``(left, right)`` inclusive index bounds of the smallest
    contiguous window of ``power`` whose sum is >= ``target``.

    ``power`` must contain only non-negative values and ``target`` must be
    achievable (i.e. ``target <= sum(power)``, up to floating-point slop).
    Uses the standard two-pointer "minimum-length subarray with sum >= T"
    technique, which is exact for non-negative arrays: for each right edge,
    the left edge is advanced as far as possible while the window sum
    remains >= target, and the shortest window seen is kept. Because all
    terms are non-negative, this greedy shrink-from-the-left step never
    skips a shorter valid window, so the global minimum-length window is
    guaranteed to be found in a single left-to-right pass. Ties (multiple
    windows of the same minimal length) are broken by the leftmost one
    encountered during the scan.
    """
    n = power.shape[0]
    # Small relative epsilon so that the target derived from summing the
    # same array is always reachable despite floating-point round-off,
    # in particular for percent_power == 100.
    total = float(np.sum(power))
    eps = 1e-9 * max(1.0, total)
    threshold = target - eps

    best_len = None
    best = (0, n - 1)
    running = 0.0
    left = 0
    for right in range(n):
        running += float(power[right])
        while left <= right and running >= threshold:
            length = right - left + 1
            if best_len is None or length < best_len:
                best_len = length
                best = (left, right)
            running -= float(power[left])
            left += 1
    return best


def estimate_occupied_bandwidth(
    samples: np.ndarray,
    sample_rate: float,
    percent_power: float = 99.0,
) -> OccupiedBandwidthEstimate:
    """Estimate occupied bandwidth via the smallest cumulative-power band.

    Definition:
        The occupied bandwidth is defined as the smallest contiguous
        frequency interval, expressed in whole FFT bins, whose accumulated
        spectral power is at least ``percent_power`` percent of the total
        measured spectral power in the analyzed block. This is a
        **cumulative-power / bin-based** definition:

        1. The two-sided FFT spectrum of ``samples`` is computed via
           :func:`iqwav.dsp.magnitude_spectrum` (bin spacing
           ``sample_rate / N`` for ``N`` input samples).
        2. Each frequency bin's power is ``magnitude ** 2``.
        3. Bins are placed in ascending frequency order (see "Real vs.
           complex input" below for how real input is handled).
        4. The total measured spectral power is the sum of all bin powers.
        5. The smallest contiguous run of bins whose summed power is >=
           ``percent_power / 100`` of the total is located (via an exact
           two-pointer minimum-window search, since bin powers are
           non-negative).
        6. The frequency edges of that run of bins become
           ``lower_frequency_hz`` and ``upper_frequency_hz``.

    Discrete-bin boundary convention:
        Each FFT bin of width ``resolution_hz = sample_rate / N`` is treated
        as covering the half-open frequency interval
        ``[bin_freq - resolution_hz / 2, bin_freq + resolution_hz / 2)``
        centered on its bin frequency. The reported ``lower_frequency_hz``
        is therefore the *left edge* of the leftmost included bin
        (``first_bin_freq - resolution_hz / 2``) and ``upper_frequency_hz``
        is the *right edge* of the rightmost included bin
        (``last_bin_freq + resolution_hz / 2``). Consequently
        ``bandwidth_hz`` equals ``(number of bins in the run) *
        resolution_hz``, except where clamped to the physically valid range
        as described below for real-valued input.

    Real vs. complex input:
        For complex-valued ``samples`` (typical baseband IQ), the full
        two-sided spectrum is used directly: bins are ordered from
        approximately ``-sample_rate / 2`` to ``+sample_rate / 2`` and each
        bin is treated as independent physical spectral content. The
        returned interval may therefore span negative frequencies, zero,
        positive frequencies, or any combination.

        For real-valued ``samples``, the FFT spectrum is conjugate
        symmetric: a real physical component at frequency ``f`` produces
        equal-magnitude bins at both ``+f`` and ``-f`` in the raw two-sided
        FFT. These are **not** independent physical content -- they are two
        views of the same real-valued spectral component -- so treating
        them as separate would double-count energy and mis-locate the
        occupied band. To avoid this, real-valued input is *folded* onto
        the non-negative half of the spectrum before the search: for each
        non-negative bin index ``k`` with ``0 < k < N // 2``, the folded
        bin power is ``power[+k] + power[-k]``; the DC bin (``k = 0``) and,
        for even ``N``, the Nyquist bin (``k = N // 2``) are each
        self-conjugate and are used unmodified (not doubled). The search
        then runs only over this folded, non-negative axis, so
        ``lower_frequency_hz`` and ``upper_frequency_hz`` are both
        guaranteed to lie in ``[0, sample_rate / 2]`` for real input; the
        bin-edge convention above is applied identically, except that the
        DC bin's lower edge is clamped to ``0.0`` (it cannot extend to
        negative frequency) and the top bin's upper edge is clamped to
        ``sample_rate / 2``. The folded total power equals the original
        two-sided total power (no energy is discarded), so
        ``percent_power`` means the same fraction of total signal power in
        both the real and complex cases.

    Meaning of percent_power:
        ``percent_power`` is the minimum fraction, expressed as a
        percentage in ``(0, 100]``, of the total measured spectral power
        that must fall within the returned interval. Larger values can
        only produce an interval whose bandwidth is greater than or equal
        to that for a smaller value on the same signal (the minimum-window
        search is monotonic in the required power fraction).

    FFT resolution limitations:
        The finest boundary precision available is one FFT bin,
        ``resolution_hz = sample_rate / N``. This function does not
        interpolate sub-bin edges; all reported boundaries fall on bin
        edges of the analyzed block's own FFT. Longer blocks (larger
        ``N``) give finer resolution at the cost of requiring the signal
        to be reasonably stationary over the analysis window.

    Not a noise-floor-aware or activity-detecting measurement:
        This function has no notion of a noise floor and does not attempt
        to separate signal power from noise power, nor does it decide
        whether a signal is present -- see the module docstring. All
        spectral power in the analyzed block, of whatever origin, counts
        toward the cumulative-power search.

    Args:
        samples: 1-D real or complex sample array with at least 4 values.
            All values must be finite, must not be all-zero, and must not
            be constant (a zero-variance signal has no spectral content
            away from DC and therefore no meaningful occupied bandwidth
            under this definition).
        sample_rate: Sampling frequency in Hz. Must be positive and finite.
        percent_power: Fraction of total spectral power, as a percentage,
            that the returned interval must contain. Must be finite and in
            ``(0, 100]``. Defaults to 99.0.

    Returns:
        An :class:`OccupiedBandwidthEstimate` with the bandwidth, lower and
        upper frequency edges (both in Hz), and the ``percent_power`` used.

    Raises:
        ValueError: If ``sample_rate`` is not positive and finite, if
            ``percent_power`` is not finite or not in ``(0, 100]``, if
            ``samples`` is not a one-dimensional array of at least 4 finite
            values, or if ``samples`` is all-zero or constant.
    """
    samples = _validate_bandwidth_args(samples, sample_rate, percent_power)
    _reject_degenerate_signal(samples)

    n = samples.size
    resolution = sample_rate / n
    freqs, magnitude = magnitude_spectrum(samples, fs=sample_rate, fftshift=False)
    power = magnitude.astype(np.float64) ** 2

    if np.iscomplexobj(samples):
        # Reorder to strictly ascending frequency (centered / fftshift
        # ordering) so "contiguous in frequency" matches "contiguous in
        # array index".
        order = np.argsort(freqs, kind="stable")
        freqs_sorted = freqs[order]
        power_sorted = power[order]

        total = float(np.sum(power_sorted))
        target = (percent_power / 100.0) * total
        left, right = _smallest_window_reaching_power(power_sorted, target)

        lower = float(freqs_sorted[left]) - resolution / 2.0
        upper = float(freqs_sorted[right]) + resolution / 2.0
    else:
        # Fold the conjugate-symmetric two-sided spectrum onto the
        # non-negative axis: bin k and bin (n - k) mod n are the same
        # physical real-valued frequency component. DC (k = 0) and, for
        # even n, Nyquist (k = n // 2) are self-conjugate and are not
        # doubled.
        half = n // 2
        folded = power[: half + 1].copy()
        for k in range(1, (n + 1) // 2):
            folded[k] += power[n - k]
        folded_freqs = freqs[: half + 1].copy()  # 0 .. Nyquist (or just below)
        if n % 2 == 0:
            # numpy.fft.fftfreq assigns the Nyquist bin (index n // 2, for
            # even n) the negative value -fs/2; since Nyquist's sign is
            # inherently ambiguous, use +fs/2 to keep this axis strictly
            # ascending and non-negative, matching the documented
            # real-signal convention.
            folded_freqs[half] = abs(folded_freqs[half])

        total = float(np.sum(folded))
        target = (percent_power / 100.0) * total
        left, right = _smallest_window_reaching_power(folded, target)

        lower = float(folded_freqs[left]) - resolution / 2.0
        upper = float(folded_freqs[right]) + resolution / 2.0
        lower = max(0.0, lower)
        upper = min(sample_rate / 2.0, upper)

    return OccupiedBandwidthEstimate(
        bandwidth_hz=upper - lower,
        lower_frequency_hz=lower,
        upper_frequency_hz=upper,
        percent_power=float(percent_power),
    )