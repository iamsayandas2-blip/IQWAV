"""Explicit-region noise-floor and SNR estimation."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..dsp.spectrum import magnitude_spectrum

__all__ = ["NoiseFloorEstimate", "SNREstimate", "estimate_noise_floor", "estimate_snr"]

_MIN_SAMPLES = 4


@dataclass(frozen=True)
class NoiseFloorEstimate:
    """Noise-density result for an explicitly supplied frequency region.

    ``noise_power`` is selected bin power, ``noise_bandwidth_hz`` is the
    selected physical bin width, and their quotient is in sample-units²/Hz.
    ``noise_floor_db`` is referenced to 1 sample-unit²/Hz.
    """

    noise_power_density: float
    noise_floor_db: float
    noise_power: float
    noise_bandwidth_hz: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    resolution_hz: float


@dataclass(frozen=True)
class SNREstimate:
    """SNR result from explicit signal and noise-reference regions."""

    snr_db: float
    signal_power: float
    estimated_noise_power: float
    measured_signal_region_power: float
    noise_power_density: float
    signal_bandwidth_hz: float
    noise_bandwidth_hz: float
    signal_lower_frequency_hz: float
    signal_upper_frequency_hz: float
    noise_lower_frequency_hz: float
    noise_upper_frequency_hz: float
    resolution_hz: float


def _validate_samples(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive and finite, got {sample_rate!r}.")
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(f"samples must be one-dimensional, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("samples must contain at least one value.")
    if values.size < _MIN_SAMPLES:
        raise ValueError(f"samples must contain at least {_MIN_SAMPLES} values for spectral analysis, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    return values


def _validate_region(region: tuple[float, float], sample_rate: float, *, name: str, minimum: float) -> tuple[float, float]:
    try:
        lower, upper = region
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-value (lower_hz, upper_hz) region.") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} bounds must be finite.")
    if lower >= upper:
        raise ValueError(f"{name} must have lower_hz < upper_hz, got {region!r}.")
    if lower < minimum or upper > sample_rate / 2.0:
        raise ValueError(f"{name} must lie within [{minimum}, {sample_rate / 2.0}] Hz, got {region!r}.")
    return float(lower), float(upper)


def _physical_spectrum(samples: np.ndarray, sample_rate: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ascending frequencies, bin powers, and physical bin widths.

    Bin power is ``abs(FFT)**2 / N**2``; Parseval makes its sum equal the
    sample mean-square power. Real input folds conjugate bins as in Phase 2B.
    """
    n = samples.size
    resolution = sample_rate / n
    frequencies, magnitude = magnitude_spectrum(samples, fs=sample_rate, fftshift=False)
    powers = magnitude.astype(np.float64) ** 2 / (n * n)
    if np.iscomplexobj(samples):
        order = np.argsort(frequencies, kind="stable")
        return frequencies[order].astype(np.float64), powers[order], np.full(n, resolution)

    half = n // 2
    folded = powers[: half + 1].copy()
    for index in range(1, (n + 1) // 2):
        folded[index] += powers[n - index]
    folded_frequencies = frequencies[: half + 1].astype(np.float64)
    if n % 2 == 0:
        folded_frequencies[-1] = sample_rate / 2.0
    widths = np.full(half + 1, resolution)
    widths[0] = resolution / 2.0
    if n % 2 == 0:
        widths[-1] = resolution / 2.0
    return folded_frequencies, folded, widths


def _region_measurement(frequencies: npt.NDArray[np.float64], powers: npt.NDArray[np.float64], widths: npt.NDArray[np.float64], region: tuple[float, float], *, name: str) -> tuple[float, float]:
    """Measure bins whose centers satisfy ``lower_hz <= f < upper_hz``."""
    lower, upper = region
    # The ordinary convention is half-open.  The real-valued Nyquist bin is
    # the sole endpoint bin on the physical non-negative axis, so include it
    # when a region ends exactly at that valid axis endpoint.
    include_upper_endpoint = math.isclose(upper, float(frequencies[-1]))
    selected = (frequencies >= lower) & (
        (frequencies < upper) | (include_upper_endpoint & (frequencies == upper))
    )
    if not np.any(selected):
        raise ValueError(f"{name} contains no usable FFT-bin centers at this resolution.")
    return float(np.sum(powers[selected])), float(np.sum(widths[selected]))


def estimate_noise_floor(samples: np.ndarray, sample_rate: float, noise_region_hz: tuple[float, float]) -> NoiseFloorEstimate:
    """Estimate noise density in a caller-designated noise-only region.

    ``(lower_hz, upper_hz)`` uses a half-open bin-center convention:
    ``lower_hz <= f < upper_hz`` (except a real Nyquist bin is included when
    ``upper_hz == sample_rate/2``). Complex inputs use independent signed bins
    on ``[-sample_rate/2, sample_rate/2]``. Real inputs use Phase 2B's folded
    non-negative spectrum on ``[0, sample_rate/2]``; DC and Nyquist count once.
    A zero-power reference is rejected because it has no finite dB floor.
    """
    values = _validate_samples(samples, sample_rate)
    minimum = -sample_rate / 2.0 if np.iscomplexobj(values) else 0.0
    region = _validate_region(noise_region_hz, sample_rate, name="noise_region_hz", minimum=minimum)
    frequencies, powers, widths = _physical_spectrum(values, sample_rate)
    noise_power, noise_bandwidth = _region_measurement(frequencies, powers, widths, region, name="noise_region_hz")
    density = noise_power / noise_bandwidth
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("noise_region_hz has zero or non-finite measured noise power; a finite noise floor cannot be estimated.")
    return NoiseFloorEstimate(density, 10.0 * math.log10(density), noise_power, noise_bandwidth, region[0], region[1], sample_rate / values.size)


def estimate_snr(samples: np.ndarray, sample_rate: float, signal_region_hz: tuple[float, float], noise_region_hz: tuple[float, float]) -> SNREstimate:
    """Estimate SNR from explicit signal and noise-reference regions.

    Signal-region power includes noise. The reference density is multiplied by
    signal-region bin bandwidth, then subtracted. Non-positive residual signal
    power (or zero reference noise) raises ``ValueError`` instead of producing
    NaN or infinity.
    """
    values = _validate_samples(samples, sample_rate)
    minimum = -sample_rate / 2.0 if np.iscomplexobj(values) else 0.0
    signal_region = _validate_region(signal_region_hz, sample_rate, name="signal_region_hz", minimum=minimum)
    noise_region = _validate_region(noise_region_hz, sample_rate, name="noise_region_hz", minimum=minimum)
    if max(signal_region[0], noise_region[0]) < min(signal_region[1], noise_region[1]):
        raise ValueError("signal_region_hz and noise_region_hz must not overlap.")
    frequencies, powers, widths = _physical_spectrum(values, sample_rate)
    signal_total, signal_bandwidth = _region_measurement(frequencies, powers, widths, signal_region, name="signal_region_hz")
    noise_total, noise_bandwidth = _region_measurement(frequencies, powers, widths, noise_region, name="noise_region_hz")
    density = noise_total / noise_bandwidth
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("noise_region_hz has zero or non-finite measured noise power; SNR cannot be estimated.")
    estimated_noise = density * signal_bandwidth
    signal_power = signal_total - estimated_noise
    tolerance = np.finfo(np.float64).eps * max(signal_total, estimated_noise, 1.0)
    if signal_power <= tolerance:
        raise ValueError("signal_region_hz has no positive residual signal power after noise-density subtraction; SNR is undefined.")
    ratio = signal_power / estimated_noise
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("SNR calculation produced a non-finite or non-positive ratio.")
    return SNREstimate(10.0 * math.log10(ratio), signal_power, estimated_noise, signal_total, density, signal_bandwidth, noise_bandwidth, signal_region[0], signal_region[1], noise_region[0], noise_region[1], sample_rate / values.size)
