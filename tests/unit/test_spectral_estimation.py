"""Unit tests for iqwav.estimation.spectral.estimate_peak_frequency."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn
from iqwav.estimation import PeakFrequencyEstimate, estimate_peak_frequency
from iqwav.modulation import generate_iq_tone, generate_real_tone

FS = 1000.0
DURATION = 1.0  # -> N = 1000 samples, resolution = 1.0 Hz, on-bin freqs integral


def test_complex_positive_frequency_tone_on_bin():
    """A tone exactly on an FFT bin should be recovered almost exactly."""
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=DURATION)
    result = estimate_peak_frequency(samples, fs=FS)
    assert isinstance(result, PeakFrequencyEstimate)
    assert result.frequency_hz == pytest.approx(137.0, abs=0.05)
    assert result.bin_frequency_hz == pytest.approx(137.0)
    assert result.bin_index == 137


def test_complex_negative_frequency_tone_on_bin():
    """A tone at a negative frequency must estimate as negative, not |f|."""
    _, samples = generate_iq_tone(fs=FS, freq=-137.0, duration=DURATION)
    result = estimate_peak_frequency(samples, fs=FS)
    assert result.frequency_hz == pytest.approx(-137.0, abs=0.05)
    assert result.bin_frequency_hz == pytest.approx(-137.0)
    assert result.frequency_hz < 0.0


def test_real_sinusoid_returns_nonnegative_frequency():
    """Real input has a symmetric spectrum; the convention is |f| >= 0."""
    _, samples = generate_real_tone(fs=FS, freq=137.0, duration=DURATION)
    result = estimate_peak_frequency(samples, fs=FS)
    assert result.frequency_hz == pytest.approx(137.0, abs=0.05)
    assert result.frequency_hz >= 0.0

    _, samples_neg = generate_real_tone(fs=FS, freq=-137.0, duration=DURATION)
    result_neg = estimate_peak_frequency(samples_neg, fs=FS)
    # cos(2*pi*(-f)*t) == cos(2*pi*f*t): a "negative" real tone is
    # indistinguishable from its positive counterpart, so both report +137.
    assert result_neg.frequency_hz == pytest.approx(137.0, abs=0.05)


def test_off_bin_tone_refinement_improves_on_raw_bin():
    """Sub-bin interpolation should land closer to the truth than the bin."""
    true_freq = 137.4  # deliberately not aligned to an integer Hz bin
    _, samples = generate_iq_tone(fs=FS, freq=true_freq, duration=DURATION)

    refined = estimate_peak_frequency(samples, fs=FS, refine=True)
    raw = estimate_peak_frequency(samples, fs=FS, refine=False)

    assert raw.frequency_hz == raw.bin_frequency_hz
    assert refined.bin_frequency_hz == raw.bin_frequency_hz
    assert not raw.refined
    assert refined.refined

    raw_error = abs(raw.frequency_hz - true_freq)
    refined_error = abs(refined.frequency_hz - true_freq)
    # The raw bin can be off by up to half the resolution (0.5 Hz here);
    # refinement should land substantially closer to the true frequency.
    assert refined_error < raw_error
    assert refined_error < 0.5 * raw_error


def test_off_bin_tone_within_half_bin_of_raw_estimate():
    """Even without refinement, the raw bin must be within resolution/2."""
    true_freq = 250.6
    _, samples = generate_iq_tone(fs=FS, freq=true_freq, duration=DURATION)
    result = estimate_peak_frequency(samples, fs=FS, refine=False)
    assert abs(result.frequency_hz - true_freq) <= result.resolution_hz / 2.0 + 1e-9


def test_noisy_tone_remains_recoverable():
    """A tone should still be the dominant peak at a moderate SNR."""
    rng = np.random.default_rng(42)
    _, samples = generate_iq_tone(fs=FS, freq=222.0, duration=DURATION, amplitude=1.0)
    noisy = add_awgn(samples, snr_db=10.0, rng=rng)
    result = estimate_peak_frequency(noisy, fs=FS)
    assert result.frequency_hz == pytest.approx(222.0, abs=2.0)


def test_multiple_tones_selects_strongest():
    """The estimator must pick the higher-amplitude of two simultaneous tones."""
    _, strong = generate_iq_tone(fs=FS, freq=180.0, duration=DURATION, amplitude=1.0)
    _, weak = generate_iq_tone(fs=FS, freq=-310.0, duration=DURATION, amplitude=0.2)
    result = estimate_peak_frequency(strong + weak, fs=FS)
    assert result.frequency_hz == pytest.approx(180.0, abs=0.05)

    # Swap which tone is stronger and confirm selection follows amplitude.
    _, strong2 = generate_iq_tone(fs=FS, freq=-310.0, duration=DURATION, amplitude=1.0)
    _, weak2 = generate_iq_tone(fs=FS, freq=180.0, duration=DURATION, amplitude=0.2)
    result2 = estimate_peak_frequency(strong2 + weak2, fs=FS)
    assert result2.frequency_hz == pytest.approx(-310.0, abs=0.05)


@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_sample_rate_raises(fs):
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=0.1)
    with pytest.raises(ValueError):
        estimate_peak_frequency(samples, fs=fs)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.array([]), fs=FS)


def test_insufficient_samples_raise():
    for n in (1, 2, 3):
        with pytest.raises(ValueError):
            estimate_peak_frequency(np.ones(n, dtype=np.complex128), fs=FS)


def test_two_dimensional_samples_raise():
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.ones((4, 4)), fs=FS)


def test_nonfinite_samples_raise():
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.array([1.0, np.nan, 2.0, 3.0]), fs=FS)
    with pytest.raises(ValueError):
        estimate_peak_frequency(
            np.array([1.0 + 1.0j, np.inf + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j]), fs=FS
        )


def test_zero_signal_raises():
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.zeros(16), fs=FS)
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.zeros(16, dtype=np.complex128), fs=FS)


def test_constant_nonzero_signal_raises():
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.full(16, 5.0), fs=FS)
    with pytest.raises(ValueError):
        estimate_peak_frequency(np.full(16, 2.0 + 3.0j), fs=FS)


def test_result_is_frozen_dataclass_with_documented_fields():
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=DURATION)
    result = estimate_peak_frequency(samples, fs=FS)
    assert hasattr(result, "frequency_hz")
    assert hasattr(result, "bin_frequency_hz")
    assert hasattr(result, "resolution_hz")
    assert hasattr(result, "bin_index")
    assert hasattr(result, "refined")
    assert result.resolution_hz == pytest.approx(FS / samples.size)
    with pytest.raises(Exception):
        result.frequency_hz = 0.0  # frozen dataclass must reject mutation