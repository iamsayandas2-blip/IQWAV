"""Unit tests for the controlled BPSK/QPSK receiver pipeline (Phase 2I).

Covers only the narrow, controlled case: known sample rate, known
integer samples-per-symbol, known modulation, known constant CFO, and
one constant integer timing offset recovered by Phase 2H. No blind
estimation of any kind is exercised or expected.
"""

import numpy as np
import pytest

from iqwav.dsp import apply_frequency_offset
from iqwav.modulation import bpsk_modulate, bpsk_waveform, qpsk_modulate, qpsk_waveform
from iqwav.pipeline import ReceiverPipelineResult, run_receiver_pipeline

FS = 48000.0
SPS = 8
N_SYMBOLS = 64


def _bits(count, seed, values=2):
    return np.random.default_rng(seed).integers(0, values, count, dtype=np.int64)


def _build(modulation, sps, timing_offset, freq_offset_hz, fs=FS, seed=0):
    """Build a controlled waveform with a timing offset and CFO applied."""
    if modulation == "bpsk":
        bits = _bits(N_SYMBOLS, seed)
        waveform = bpsk_waveform(bits, sps)
    else:
        bits = _bits(N_SYMBOLS * 2, seed)
        waveform = qpsk_waveform(bits, sps)
    padded = np.concatenate([np.zeros(timing_offset, dtype=waveform.dtype), waveform])
    offset_applied = apply_frequency_offset(padded, fs, freq_offset_hz)
    return offset_applied, bits


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_clean_signal_zero_cfo_zero_offset(modulation):
    samples, bits = _build(modulation, SPS, timing_offset=0, freq_offset_hz=0.0)
    result = run_receiver_pipeline(samples, FS, SPS, modulation, 0.0)
    assert np.array_equal(result.bits, bits)
    assert result.timing_offset == 0


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("cfo", [500.0, -500.0, 0.0])
def test_cfo_signs(modulation, cfo):
    samples, bits = _build(modulation, SPS, timing_offset=3, freq_offset_hz=cfo)
    result = run_receiver_pipeline(samples, FS, SPS, modulation, cfo)
    assert np.array_equal(result.bits, bits)
    assert result.timing_offset == 3
    assert result.frequency_offset_hz == cfo


@pytest.mark.parametrize("fs", [8000.0, 48000.0, 1_000_000.0])
def test_multiple_sample_rates(fs):
    samples, bits = _build("bpsk", SPS, timing_offset=2, freq_offset_hz=100.0, fs=fs)
    result = run_receiver_pipeline(samples, fs, SPS, "bpsk", 100.0)
    assert np.array_equal(result.bits, bits)
    assert result.sample_rate == fs


@pytest.mark.parametrize("sps", [1, 2, 4, 8, 16])
def test_multiple_samples_per_symbol(sps):
    samples, bits = _build("qpsk", sps, timing_offset=0, freq_offset_hz=-200.0)
    result = run_receiver_pipeline(samples, FS, sps, "qpsk", -200.0)
    assert np.array_equal(result.bits, bits)


@pytest.mark.parametrize("timing_offset", [0, 1, 3, 7])
def test_multiple_integer_timing_phases(timing_offset):
    samples, bits = _build(
        "bpsk", SPS, timing_offset=timing_offset, freq_offset_hz=50.0
    )
    result = run_receiver_pipeline(samples, FS, SPS, "bpsk", 50.0)
    assert result.timing_offset == timing_offset
    assert np.array_equal(result.bits, bits)


def test_returned_metadata():
    samples, _ = _build("bpsk", SPS, timing_offset=2, freq_offset_hz=100.0)
    result = run_receiver_pipeline(samples, FS, SPS, "bpsk", 100.0)
    assert isinstance(result, ReceiverPipelineResult)
    assert result.modulation == "bpsk"
    assert result.sample_rate == FS
    assert result.samples_per_symbol == SPS
    assert result.frequency_offset_hz == 100.0
    assert result.cfo_corrected_samples.shape == samples.shape
    assert result.timing_recovery.samples_per_symbol == SPS
    assert result.symbol_count == result.timing_recovery.symbol_count


def test_input_not_mutated():
    samples, _ = _build("qpsk", SPS, timing_offset=1, freq_offset_hz=100.0)
    original = samples.copy()
    run_receiver_pipeline(samples, FS, SPS, "qpsk", 100.0)
    assert np.array_equal(samples, original)


def test_deterministic_repeated_calls():
    samples, _ = _build("bpsk", SPS, timing_offset=2, freq_offset_hz=75.0)
    result_a = run_receiver_pipeline(samples, FS, SPS, "bpsk", 75.0)
    result_b = run_receiver_pipeline(samples, FS, SPS, "bpsk", 75.0)
    assert np.array_equal(result_a.bits, result_b.bits)
    assert result_a.timing_offset == result_b.timing_offset


def test_invalid_modulation():
    samples, _ = _build("bpsk", SPS, timing_offset=0, freq_offset_hz=0.0)
    with pytest.raises(ValueError):
        run_receiver_pipeline(samples, FS, SPS, "8psk", 0.0)


@pytest.mark.parametrize("bad_rate", [0.0, -48000.0, float("nan"), float("inf")])
def test_invalid_sample_rate(bad_rate):
    samples, _ = _build("bpsk", SPS, timing_offset=0, freq_offset_hz=0.0)
    with pytest.raises(ValueError):
        run_receiver_pipeline(samples, bad_rate, SPS, "bpsk", 0.0)


@pytest.mark.parametrize("bad_sps", [0, -1, 2.5])
def test_invalid_samples_per_symbol(bad_sps):
    samples, _ = _build("bpsk", SPS, timing_offset=0, freq_offset_hz=0.0)
    with pytest.raises(ValueError):
        run_receiver_pipeline(samples, FS, bad_sps, "bpsk", 0.0)


def test_empty_input():
    with pytest.raises(ValueError):
        run_receiver_pipeline(np.array([], dtype=np.complex128), FS, SPS, "bpsk", 0.0)


def test_insufficient_input():
    # Fewer samples than Phase 2H needs to score four windows per phase.
    short = np.ones(SPS * 2, dtype=np.complex128)
    with pytest.raises(ValueError):
        run_receiver_pipeline(short, FS, SPS, "bpsk", 0.0)


def test_no_blind_operation_when_cfo_wrong():
    """Supplying an incorrect CFO must not be silently corrected.

    The pipeline performs no blind CFO estimation, so an intentionally
    wrong CFO value should generally corrupt symbols rather than being
    quietly fixed up.
    """
    samples, bits = _build("bpsk", SPS, timing_offset=0, freq_offset_hz=3000.0)
    result = run_receiver_pipeline(samples, FS, SPS, "bpsk", 0.0)
    assert not np.array_equal(result.bits, bits)