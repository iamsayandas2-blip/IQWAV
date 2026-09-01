"""Spectral peak frequency estimation.

This module provides a single, narrowly-scoped estimator:
:func:`estimate_peak_frequency`, which locates the frequency of the
dominant spectral component in a block of sampled data using the
existing IQWAV FFT machinery (:func:`iqwav.dsp.magnitude_spectrum`).

Scope
-----
This is a *spectral peak estimator*, not a blind RF carrier estimator.
It assumes the sample rate is already known and simply reports where
the largest FFT magnitude occurs (optionally refined to sub-bin
precision). It does not attempt occupied-bandwidth estimation, noise
floor / SNR estimation, activity detection, carrier recovery, CFO
correction, timing recovery, baud-rate estimation, or modulation
classification -- those are separate, later milestones.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..dsp.spectrum import magnitude_spectrum

__all__ = ["PeakFrequencyEstimate", "estimate_peak_frequency"]

_MIN_SAMPLES = 4
# Relative-to-signal threshold below which the AC (non-DC) content of a
# signal is treated as numerically zero, i.e. a constant/zero signal with
# no oscillating component to localize a frequency for.
_CONSTANT_SIGNAL_RTOL = 1e-12


@dataclass(frozen=True)
class PeakFrequencyEstimate:
    """Result of a single dominant-frequency spectral estimate.

    Attributes:
        frequency_hz: The final frequency estimate in Hz. Equal to
            ``bin_frequency_hz`` when ``refined`` is False, or to the
            sub-bin-refined frequency when ``refined`` is True.
        bin_frequency_hz: The center frequency, in Hz, of the raw FFT bin
            that had the largest magnitude, with no sub-bin refinement.
        resolution_hz: The raw FFT bin spacing ``fs / N`` in Hz, i.e. the
            best frequency resolution available before any refinement.
            This is the width of one FFT bin, not an error bound.
        bin_index: Index of the selected peak bin into NumPy's standard
            (unshifted) FFT bin ordering: ``bin_index / N * fs`` for
            ``bin_index <= N // 2`` and ``(bin_index - N) / N * fs``
            otherwise, matching ``numpy.fft.fftfreq``.
        refined: Whether ``frequency_hz`` includes sub-bin parabolic
            interpolation (True) or is the raw bin center (False).
    """

    frequency_hz: float
    bin_frequency_hz: float
    resolution_hz: float
    bin_index: int
    refined: bool


def _validate_estimate_args(samples: np.ndarray, fs: float) -> np.ndarray:
    """Validate estimator arguments and return samples as an ndarray."""
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
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
            f"estimate a spectral peak, got {samples.size}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _reject_constant_signal(samples: np.ndarray) -> None:
    """Raise ValueError if samples have no oscillating (AC) content."""
    ac = samples - np.mean(samples)
    ac_level = float(np.max(np.abs(ac))) if ac.size else 0.0
    scale = max(1.0, float(np.max(np.abs(samples))))
    if ac_level <= _CONSTANT_SIGNAL_RTOL * scale:
        raise ValueError(
            "samples are constant (or all zero) and contain no oscillating "
            "component; a dominant frequency cannot be estimated for a "
            "signal with no spectral content away from DC."
        )


def _wrap_bin_to_freq(bin_pos: float, n: int, fs: float) -> float:
    """Map a (possibly fractional) FFT bin index to a signed frequency.

    Generalizes ``numpy.fft.fftfreq`` to non-integer bin positions by
    wrapping into the half-open interval ``(-n/2, n/2]`` of bins before
    scaling by ``fs / n``.
    """
    wrapped = ((bin_pos + n / 2.0) % n) - n / 2.0
    return wrapped * fs / n


def _parabolic_log_refine(
    magnitude: np.ndarray, peak_index: int
) -> float:
    """Return a sub-bin offset via parabolic interpolation of log-magnitude.

    Fits a parabola through the log-magnitude of the peak bin and its two
    circular neighbors and returns the offset (in bins, within [-0.5, 0.5])
    of the parabola's vertex relative to ``peak_index``. This is the
    standard three-point log-magnitude interpolator for FFT peaks; it
    assumes the true spectral peak is well-approximated locally by a
    parabola in log-magnitude, which holds for an isolated sinusoid
    observed through a rectangular window. If the three magnitudes are
    degenerate (e.g. a flat top) no offset is applied.
    """
    n = magnitude.size
    left = magnitude[(peak_index - 1) % n]
    center = magnitude[peak_index]
    right = magnitude[(peak_index + 1) % n]
    # Guard against exact zeros before taking logs (should not occur for a
    # genuine peak bin, but keeps this function total).
    eps = np.finfo(np.float64).tiny
    y_left = math.log(max(left, eps))
    y_center = math.log(max(center, eps))
    y_right = math.log(max(right, eps))
    denom = y_left - 2.0 * y_center + y_right
    if denom == 0.0:
        return 0.0
    delta = 0.5 * (y_left - y_right) / denom
    # Numerical safety: the vertex of a genuine local parabola around the
    # peak bin lies within one bin of it.
    return float(np.clip(delta, -0.5, 0.5))


def estimate_peak_frequency(
    samples: np.ndarray,
    fs: float,
    *,
    refine: bool = True,
) -> PeakFrequencyEstimate:
    """Estimate the frequency of the dominant spectral component.

    This locates the largest-magnitude bin of the FFT-based two-sided
    spectrum (via :func:`iqwav.dsp.magnitude_spectrum`) and reports its
    frequency, optionally refined to sub-bin precision by parabolic
    interpolation of the local log-magnitude.

    Real vs. complex input:
        For complex-valued ``samples`` (typical baseband IQ), the full
        two-sided spectrum is searched and the returned frequency is
        **signed**: a tone at ``+f`` estimates to approximately ``+f``,
        and a tone at ``-f`` estimates to approximately ``-f``.

        For real-valued ``samples``, the spectrum of a real signal is
        conjugate-symmetric, so a real tone at frequency ``f`` produces
        equal-magnitude peaks at both ``+f`` and ``-f``. In that case the
        search is restricted to the non-negative half of the spectrum
        (``0`` to ``fs / 2`` inclusive) and the returned frequency is
        therefore always **non-negative**; sign is not a meaningful
        concept for a real-valued tone.

    Sample-rate units:
        ``fs`` is in Hz (samples per second) and must be positive and
        finite.

    Output units:
        The returned ``frequency_hz`` (and other frequency fields) are in
        Hz, using the same axis convention as the rest of IQWAV's
        FFT-based DSP: ``numpy.fft.fftfreq`` bin ordering, wrapped to the
        signed range ``(-fs/2, fs/2]``.

    Resolution and refinement:
        The raw FFT bin spacing is ``fs / N`` for ``N`` input samples;
        this is the coarse resolution before any refinement, and is
        returned as ``resolution_hz``. When ``refine=True`` (the
        default), the estimate is sharpened with standard three-point
        parabolic interpolation of the local log-magnitude around the
        peak bin (see :func:`_parabolic_log_refine`), which assumes an
        isolated sinusoidal peak observed through a rectangular window.
        This does not change ``bin_frequency_hz``, which always reports
        the unrefined bin center. When ``refine=False``, ``frequency_hz``
        equals ``bin_frequency_hz`` and no interpolation is performed.

    This function is a spectral peak estimator: it assumes ``fs`` is
    already known and does not attempt blind carrier detection, CFO
    estimation, occupied-bandwidth estimation, or noise-floor/SNR
    estimation.

    Args:
        samples: 1-D real or complex sample array with at least 4 values.
            All values must be finite. Must not be constant (a
            zero-variance/all-zero signal has no oscillating component
            to localize a frequency for and raises ``ValueError``).
        fs: Sampling frequency in Hz. Must be positive and finite.
        refine: If True (default), apply sub-bin parabolic log-magnitude
            interpolation. If False, return the raw FFT bin center.

    Returns:
        A :class:`PeakFrequencyEstimate` with the final frequency, the
        raw bin frequency, the FFT resolution, the selected bin index,
        and whether refinement was applied.

    Raises:
        ValueError: If ``fs`` is not positive and finite, if ``samples``
            is not a one-dimensional array of at least 4 finite values,
            or if ``samples`` is constant (including all-zero).
    """
    samples = _validate_estimate_args(samples, fs)
    _reject_constant_signal(samples)

    n = samples.size
    freqs, magnitude = magnitude_spectrum(samples, fs=fs, fftshift=False)

    if np.iscomplexobj(samples):
        search_magnitude = magnitude
    else:
        # Restrict to the non-negative half: unshifted bins 0 .. n // 2
        # cover frequencies 0 .. fs/2 for numpy.fft.fftfreq ordering.
        half = n // 2 + 1
        search_magnitude = magnitude[:half]

    peak_index = int(np.argmax(search_magnitude))
    bin_frequency = float(freqs[peak_index])
    if not np.iscomplexobj(samples):
        # numpy.fft.fftfreq assigns the Nyquist bin (index n // 2, for even
        # n) a negative frequency; since Nyquist's sign is inherently
        # ambiguous, take the magnitude to keep the documented "always
        # non-negative for real input" contract exact at that one bin.
        bin_frequency = abs(bin_frequency)
    resolution = fs / n

    if not refine:
        return PeakFrequencyEstimate(
            frequency_hz=bin_frequency,
            bin_frequency_hz=bin_frequency,
            resolution_hz=resolution,
            bin_index=peak_index,
            refined=False,
        )

    if np.iscomplexobj(samples):
        delta_bins = _parabolic_log_refine(magnitude, peak_index)
        refined_frequency = _wrap_bin_to_freq(peak_index + delta_bins, n, fs)
    else:
        # Refine using the full circular spectrum (so interpolation can see
        # both neighbors even at the band edges 0 and fs/2), but clamp the
        # result back to the non-negative half, consistent with the
        # real-signal convention described above.
        delta_bins = _parabolic_log_refine(magnitude, peak_index)
        refined_frequency = _wrap_bin_to_freq(peak_index + delta_bins, n, fs)
        if refined_frequency < 0.0:
            refined_frequency = -refined_frequency

    return PeakFrequencyEstimate(
        frequency_hz=refined_frequency,
        bin_frequency_hz=bin_frequency,
        resolution_hz=resolution,
        bin_index=peak_index,
        refined=True,
    )