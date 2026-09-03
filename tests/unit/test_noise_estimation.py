"""Unit tests for explicit-region noise-floor and SNR estimation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.estimation import (
    NoiseFloorEstimate,
    SNREstimate,
    estimate_noise_floor,
    estimate_snr,
)

FS = 1024.0
N = 1024


def _flat_complex_noise(bin_power: float = 0.002) -> np.ndarray:
    """Complex samples with exactly equal FFT-bin power and seeded phases."""
    phases = np.random.default_rng(71).uniform(0.0, 2.0 * np.pi, N)
    phases[100] = np.pi / 2.0
    phases[N - 120] = np.pi / 2.0
    spectrum = N * np.sqrt(bin_power) * np.exp(1j * phases)
    return np.fft.ifft(spectrum)


def test_awgn_floor_agrees_with_known_variance_with_finite_sample_tolerance():
    rng = np.random.default_rng(17)
    samples = rng.normal(scale=0.5, size=32768)
    result = estimate_noise_floor(samples, 32768.0, (1000.0, 15000.0))
    # Folding the real two-sided spectrum yields the one-sided PSD 2*variance/fs.
    assert result.noise_power_density == pytest.approx(2 * 0.25 / 32768.0, rel=0.12)
    assert result.noise_floor_db == pytest.approx(10 * np.log10(2 * 0.25 / 32768.0), abs=0.55)


@pytest.mark.parametrize("variance", [0.04, 0.25, 1.0])
def test_awgn_floor_tracks_several_power_levels(variance):
    rng = np.random.default_rng(100 + int(variance * 100))
    samples = rng.normal(scale=np.sqrt(variance), size=32768)
    result = estimate_noise_floor(samples, 32768.0, (2000.0, 14000.0))
    assert result.noise_power_density == pytest.approx(2 * variance / 32768.0, rel=0.13)


def test_noise_region_is_explicitly_used():
    samples = _flat_complex_noise()
    samples += np.exp(2j * np.pi * 200.0 * np.arange(N) / FS)
    quiet = estimate_noise_floor(samples, FS, (300.0, 400.0))
    contaminated = estimate_noise_floor(samples, FS, (200.0, 201.0))
    assert quiet.noise_power_density == pytest.approx(0.002, abs=1e-12)
    assert contaminated.noise_power_density > 100 * quiet.noise_power_density


def test_snr_scales_noise_to_signal_region_bandwidth():
    samples = _flat_complex_noise()
    samples += np.exp(2j * np.pi * 100.0 * np.arange(N) / FS)
    result = estimate_snr(samples, FS, (100.0, 110.0), (300.0, 500.0))
    # Each bin has 0.002 noise power.  Ten signal bins therefore contain
    # 0.02 noise, not the 0.4 noise power in the 200-Hz reference region.
    assert result.noise_power_density == pytest.approx(0.002, abs=1e-12)
    assert result.estimated_noise_power == pytest.approx(0.02, abs=1e-12)
    assert result.signal_power == pytest.approx(1.0, abs=1e-12)
    assert result.snr_db == pytest.approx(10.0 * np.log10(50.0), abs=1e-10)


def test_tone_plus_awgn_has_approximately_known_snr():
    rng = np.random.default_rng(23)
    n = 32768
    fs = float(n)
    noise_variance = 0.01
    noise = rng.normal(scale=np.sqrt(noise_variance / 2), size=n) + 1j * rng.normal(scale=np.sqrt(noise_variance / 2), size=n)
    tone = np.exp(2j * np.pi * 400.0 * np.arange(n) / fs)
    result = estimate_snr(tone + noise, fs, (390.0, 410.0), (1000.0, 9000.0))
    expected = 10.0 * np.log10(1.0 / (noise_variance * 20.0 / n))
    assert result.snr_db == pytest.approx(expected, abs=1.2)


def test_complex_positive_and_negative_regions_remain_independent():
    samples = _flat_complex_noise()
    samples += np.exp(-2j * np.pi * 120.0 * np.arange(N) / FS)
    negative = estimate_snr(samples, FS, (-120.0, -119.0), (250.0, 350.0))
    positive = estimate_noise_floor(samples, FS, (120.0, 121.0))
    assert negative.signal_power == pytest.approx(1.0, abs=1e-12)
    assert positive.noise_power_density == pytest.approx(0.002, abs=1e-12)


def test_real_input_folds_conjugate_components_once():
    rng = np.random.default_rng(4)
    n = 32768
    fs = float(n)
    noise = rng.normal(scale=np.sqrt(0.04), size=n)
    tone = 2.0 * np.cos(2 * np.pi * 300.0 * np.arange(n) / fs)
    result = estimate_snr(tone + noise, fs, (295.0, 305.0), (1000.0, 9000.0))
    assert result.signal_power == pytest.approx(2.0, rel=0.03)
    assert result.noise_power_density == pytest.approx(2 * 0.04 / n, rel=0.12)


def test_dc_and_even_nyquist_use_half_width_physical_bins():
    dc = np.ones(N)
    nyquist = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    dc_result = estimate_noise_floor(dc, FS, (0.0, 1.0))
    nyquist_result = estimate_noise_floor(nyquist, FS, (FS / 2.0 - 0.5, FS / 2.0))
    assert dc_result.noise_power == pytest.approx(1.0)
    assert dc_result.noise_bandwidth_hz == pytest.approx(0.5)
    assert dc_result.noise_power_density == pytest.approx(2.0)
    assert nyquist_result.noise_power == pytest.approx(1.0)
    assert nyquist_result.noise_bandwidth_hz == pytest.approx(0.5)
    assert nyquist_result.noise_power_density == pytest.approx(2.0)


def test_repeatability_and_frozen_result_dataclasses():
    samples = _flat_complex_noise()
    first = estimate_noise_floor(samples, FS, (20.0, 30.0))
    second = estimate_noise_floor(samples, FS, (20.0, 30.0))
    assert first == second
    assert isinstance(first, NoiseFloorEstimate)
    snr = estimate_snr(samples + np.exp(2j * np.pi * 100 * np.arange(N) / FS), FS, (100.0, 101.0), (300.0, 400.0))
    assert isinstance(snr, SNREstimate)
    with pytest.raises(FrozenInstanceError):
        first.noise_power = 0.0
    with pytest.raises(FrozenInstanceError):
        snr.snr_db = 0.0


@pytest.mark.parametrize("sample_rate", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_sample_rates_raise(sample_rate):
    with pytest.raises(ValueError):
        estimate_noise_floor(np.ones(N), sample_rate, (1.0, 2.0))


@pytest.mark.parametrize("samples", [np.array([]), np.ones(3), np.ones((4, 4)), np.array([1.0, np.nan, 2.0, 3.0])])
def test_invalid_sample_arrays_raise(samples):
    with pytest.raises(ValueError):
        estimate_noise_floor(samples, FS, (1.0, 2.0))


@pytest.mark.parametrize("region", [(1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf")), (-1.0, 2.0), (0.1, 0.2)])
def test_invalid_or_empty_real_regions_raise(region):
    with pytest.raises(ValueError):
        estimate_noise_floor(np.ones(N), FS, region)


def test_complex_out_of_range_and_empty_regions_raise():
    samples = _flat_complex_noise()
    with pytest.raises(ValueError):
        estimate_noise_floor(samples, FS, (-FS / 2 - 1.0, -1.0))
    with pytest.raises(ValueError):
        estimate_noise_floor(samples, FS, (0.1, 0.2))


def test_zero_noise_and_zero_residual_signal_raise():
    tone = np.zeros(N, dtype=np.complex128)
    with pytest.raises(ValueError):
        estimate_noise_floor(tone, FS, (300.0, 400.0))
    with pytest.raises(ValueError):
        estimate_snr(_flat_complex_noise(), FS, (100.0, 110.0), (300.0, 400.0))


def test_overlapping_signal_and_noise_regions_raise():
    with pytest.raises(ValueError):
        estimate_snr(_flat_complex_noise(), FS, (100.0, 110.0), (105.0, 120.0))
