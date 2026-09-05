"""Unit tests for iqwav.dsp.band_extraction.extract_band."""

import numpy as np
import pytest

from iqwav.dsp import BandExtractionResult, extract_band
from iqwav.modulation import generate_iq_tone

FS = 1000.0
DURATION = 1.0  # seconds -> 1000 samples


def _tone(freq: float, fs: float = FS, duration: float = DURATION) -> np.ndarray:
    _, iq = generate_iq_tone(fs=fs, freq=freq, duration=duration)
    return iq


def _passband_amplitude(signal: np.ndarray) -> float:
    # Skip filter transient at the start.
    return float(np.max(np.abs(signal[300:])))


def test_positive_frequency_band_preserves_in_band_tone():
    tone = _tone(150.0)
    result = extract_band(tone, FS, 100.0, 200.0)
    assert _passband_amplitude(result.samples) > 0.7


def test_negative_frequency_band_preserves_in_band_tone():
    tone = _tone(-150.0)
    result = extract_band(tone, FS, -200.0, -100.0)
    assert _passband_amplitude(result.samples) > 0.7


def _dominant_frequency(signal: np.ndarray, fs: float) -> float:
    seg = signal[300:600]
    spectrum = np.fft.fftshift(np.fft.fft(seg))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(seg), d=1.0 / fs))
    return float(freqs[np.argmax(np.abs(spectrum))])


def test_positive_band_tone_stays_at_original_frequency():
    # Proves the mix-down/filter/mix-up sign convention: the extracted
    # tone must remain at its original positive frequency, not be left at
    # baseband or flipped in sign.
    tone = _tone(150.0)
    result = extract_band(tone, FS, 100.0, 200.0)
    peak_freq = _dominant_frequency(result.samples, FS)
    assert peak_freq == pytest.approx(150.0, abs=5.0)


def test_negative_band_extraction_does_not_return_positive_tone():
    # Extracting a negative-frequency band must not accidentally return
    # the corresponding positive-frequency tone (a sign-convention bug
    # would mix down/up with the wrong sign and alias the positive tone
    # into the negative band, or vice versa).
    tone = _tone(-150.0)
    result = extract_band(tone, FS, -200.0, -100.0)
    peak_freq = _dominant_frequency(result.samples, FS)
    assert peak_freq == pytest.approx(-150.0, abs=5.0)
    assert peak_freq < 0


def test_low_frequency_band_near_dc():
    tone = _tone(10.0)
    result = extract_band(tone, FS, -20.0, 20.0)
    assert _passband_amplitude(result.samples) > 0.7


def test_band_containing_dc():
    tone = _tone(0.0)
    result = extract_band(tone, FS, -50.0, 50.0)
    assert _passband_amplitude(result.samples) > 0.7


def test_tone_outside_passband_strongly_attenuated():
    tone = _tone(400.0)
    result = extract_band(tone, FS, 100.0, 200.0)
    in_band = _passband_amplitude(_tone(150.0))
    assert _passband_amplitude(result.samples) < 0.1 * in_band


def test_multiple_tones_only_one_in_band():
    mixed = _tone(150.0) + _tone(400.0) + _tone(-300.0)
    result = extract_band(mixed, FS, 100.0, 200.0)
    single = extract_band(_tone(150.0), FS, 100.0, 200.0)
    # Extracted band should closely match the single in-band tone alone.
    diff = np.max(np.abs(result.samples[300:] - single.samples[300:]))
    assert diff < 0.2


def test_sample_count_preserved():
    tone = _tone(150.0)
    result = extract_band(tone, FS, 100.0, 200.0)
    assert result.samples.shape[0] == tone.shape[0]
    assert result.input_length == tone.shape[0]
    assert result.output_length == tone.shape[0]


def test_input_not_mutated():
    tone = _tone(150.0)
    original = tone.copy()
    extract_band(tone, FS, 100.0, 200.0)
    assert np.array_equal(tone, original)


def test_output_is_new_array():
    tone = _tone(150.0)
    result = extract_band(tone, FS, 100.0, 200.0)
    assert result.samples is not tone


def test_deterministic_output():
    tone = _tone(150.0)
    r1 = extract_band(tone, FS, 100.0, 200.0)
    r2 = extract_band(tone, FS, 100.0, 200.0)
    assert np.array_equal(r1.samples, r2.samples)


def test_different_sample_rates():
    fs = 4000.0
    tone = _tone(600.0, fs=fs, duration=0.5)
    result = extract_band(tone, fs, 500.0, 700.0)
    assert result.sample_rate == fs
    assert _passband_amplitude(result.samples) > 0.7


def test_result_metadata_fields():
    tone = _tone(150.0)
    result = extract_band(tone, FS, 100.0, 200.0, numtaps=51)
    assert isinstance(result, BandExtractionResult)
    assert result.sample_rate == FS
    assert result.lower_hz == 100.0
    assert result.upper_hz == 200.0
    assert result.numtaps == 51


def test_invalid_sample_rate():
    tone = _tone(150.0)
    with pytest.raises(ValueError):
        extract_band(tone, 0.0, 100.0, 200.0)
    with pytest.raises(ValueError):
        extract_band(tone, -1000.0, 100.0, 200.0)
    with pytest.raises(ValueError):
        extract_band(tone, float("nan"), 100.0, 200.0)


def test_invalid_cutoffs_lower_not_less_than_upper():
    tone = _tone(150.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 200.0, 100.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, 100.0)


def test_invalid_cutoffs_non_finite():
    tone = _tone(150.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, float("nan"), 200.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, float("inf"))


def test_empty_input():
    with pytest.raises(ValueError):
        extract_band(np.array([], dtype=np.complex128), FS, 100.0, 200.0)


def test_real_valued_input_rejected():
    real_signal = np.ones(500, dtype=np.float64)
    with pytest.raises(ValueError):
        extract_band(real_signal, FS, 100.0, 200.0)


def test_non_finite_input_rejected():
    tone = _tone(150.0)
    tone[10] = np.nan
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, 200.0)
    tone2 = _tone(150.0)
    tone2[10] = np.inf
    with pytest.raises(ValueError):
        extract_band(tone2, FS, 100.0, 200.0)


def test_band_outside_nyquist_rejected():
    tone = _tone(150.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 400.0, 600.0)  # upper >= fs/2
    with pytest.raises(ValueError):
        extract_band(tone, FS, -600.0, -400.0)  # lower <= -fs/2


def test_multidimensional_input_rejected():
    bad = np.zeros((10, 2), dtype=np.complex128)
    with pytest.raises(ValueError):
        extract_band(bad, FS, 100.0, 200.0)


def test_too_short_input_rejected():
    with pytest.raises(ValueError):
        extract_band(np.array([1.0 + 0j], dtype=np.complex128), FS, 100.0, 200.0)


def test_invalid_numtaps_below_minimum():
    tone = _tone(150.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, 200.0, numtaps=1)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, 200.0, numtaps=0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, 200.0, numtaps=-5)


def test_invalid_numtaps_non_integer():
    tone = _tone(150.0)
    with pytest.raises(ValueError):
        extract_band(tone, FS, 100.0, 200.0, numtaps=50.5)