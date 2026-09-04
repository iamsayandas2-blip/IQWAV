"""Unit tests for CFO correction in iqwav.dsp.frequency_correction.

These cover the narrow, known-offset correction only: a known sample
rate, a known frequency offset, and the elementwise conjugate-phasor
multiply documented in the module docstring. No CFO estimation, carrier
recovery, or timing recovery is involved.
"""

import numpy as np
import pytest

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import apply_frequency_offset, correct_frequency_offset
from iqwav.modulation import bpsk_waveform, generate_iq_tone, qpsk_waveform


def test_zero_frequency_offset_leaves_signal_unchanged():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.05, amplitude=1.5)
    corrected = correct_frequency_offset(samples, sample_rate=1000.0, frequency_offset_hz=0.0)
    np.testing.assert_allclose(corrected, samples)


def test_positive_frequency_offset_correction_recovers_tone():
    fs = 1000.0
    _, original = generate_iq_tone(fs=fs, freq=100.0, duration=0.5, amplitude=1.5)
    received = apply_frequency_offset(original, fs=fs, freq_offset_hz=250.0)
    corrected = correct_frequency_offset(received, sample_rate=fs, frequency_offset_hz=250.0)
    np.testing.assert_allclose(corrected, original, atol=1e-9)


def test_negative_frequency_offset_correction_recovers_tone():
    fs = 1000.0
    _, original = generate_iq_tone(fs=fs, freq=100.0, duration=0.5, amplitude=1.5)
    received = apply_frequency_offset(original, fs=fs, freq_offset_hz=-180.0)
    corrected = correct_frequency_offset(received, sample_rate=fs, frequency_offset_hz=-180.0)
    np.testing.assert_allclose(corrected, original, atol=1e-9)


@pytest.mark.parametrize("fs", [500.0, 1000.0, 44100.0, 2_000_000.0])
def test_multiple_sample_rates_round_trip(fs):
    _, original = generate_iq_tone(fs=fs, freq=fs / 20.0, duration=0.01, amplitude=0.7)
    received = apply_frequency_offset(original, fs=fs, freq_offset_hz=fs / 37.0)
    corrected = correct_frequency_offset(
        received, sample_rate=fs, frequency_offset_hz=fs / 37.0
    )
    np.testing.assert_allclose(corrected, original, atol=1e-9)


@pytest.mark.parametrize("samples_per_symbol", [1, 2, 4, 8, 16])
def test_multiple_samples_per_symbol_bpsk_round_trip(samples_per_symbol):
    fs = 1000.0
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    waveform = bpsk_waveform(bits, samples_per_symbol)
    received = apply_frequency_offset(waveform, fs=fs, freq_offset_hz=13.0)
    corrected = correct_frequency_offset(received, sample_rate=fs, frequency_offset_hz=13.0)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)


def test_controlled_bpsk_waveform_with_injected_cfo_restored():
    fs = 2000.0
    bits = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    waveform = bpsk_waveform(bits, samples_per_symbol=8)
    received = apply_frequency_offset(waveform, fs=fs, freq_offset_hz=60.0)
    corrected = correct_frequency_offset(received, sample_rate=fs, frequency_offset_hz=60.0)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    recovered = bpsk_demodulate(corrected, samples_per_symbol=8)
    np.testing.assert_array_equal(recovered, bits)


def test_controlled_qpsk_waveform_with_injected_cfo_restored():
    fs = 2000.0
    bits = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    waveform = qpsk_waveform(bits, samples_per_symbol=10)
    received = apply_frequency_offset(waveform, fs=fs, freq_offset_hz=-95.0)
    corrected = correct_frequency_offset(received, sample_rate=fs, frequency_offset_hz=-95.0)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    recovered = qpsk_demodulate(corrected, samples_per_symbol=10)
    np.testing.assert_array_equal(recovered, bits)


def test_offset_larger_than_symbol_rate_still_corrects():
    # Symbol rate here is fs / samples_per_symbol = 2000 / 4 = 500 Hz;
    # use an offset well beyond that.
    fs = 2000.0
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    waveform = bpsk_waveform(bits, samples_per_symbol=4)
    large_offset = 900.0
    received = apply_frequency_offset(waveform, fs=fs, freq_offset_hz=large_offset)
    corrected = correct_frequency_offset(
        received, sample_rate=fs, frequency_offset_hz=large_offset
    )
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)


def test_magnitude_is_preserved():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=123.0, duration=0.5, amplitude=1.5)
    corrected = correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=321.0)
    np.testing.assert_allclose(np.abs(corrected), np.abs(samples))


def test_sample_count_is_preserved():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=100.0, duration=0.05)
    corrected = correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=42.0)
    assert corrected.shape == samples.shape


def test_input_array_is_not_mutated():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=100.0, duration=0.05)
    before = samples.copy()
    correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=42.0)
    np.testing.assert_array_equal(samples, before)


def test_output_is_a_new_array():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=100.0, duration=0.05)
    corrected = correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=42.0)
    assert corrected is not samples


def test_deterministic_output():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=100.0, duration=0.05, amplitude=1.5)
    first = correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=17.0)
    second = correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=17.0)
    np.testing.assert_array_equal(first, second)


def test_output_dtype_is_complex128():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=100.0, duration=0.05)
    corrected = correct_frequency_offset(samples, sample_rate=fs, frequency_offset_hz=50.0)
    assert corrected.dtype == np.complex128
    assert np.iscomplexobj(corrected)


@pytest.mark.parametrize("sample_rate", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_sample_rate_raises(sample_rate):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, sample_rate=sample_rate, frequency_offset_hz=10.0)


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -float("inf")])
def test_invalid_frequency_offset_raises(offset):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, sample_rate=1000.0, frequency_offset_hz=offset)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        correct_frequency_offset(
            np.array([], dtype=np.complex128), sample_rate=1000.0, frequency_offset_hz=10.0
        )


def test_non_one_dimensional_input_raises():
    samples = np.ones((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, sample_rate=1000.0, frequency_offset_hz=10.0)


def test_non_finite_input_raises():
    samples = np.array([1.0 + 2.0j, np.nan + 1.0j, 3.0 + 4.0j])
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, sample_rate=1000.0, frequency_offset_hz=10.0)


def test_real_valued_input_rejected():
    samples = np.ones(16, dtype=np.float64)
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, sample_rate=1000.0, frequency_offset_hz=10.0)