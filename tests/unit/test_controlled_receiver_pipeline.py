"""Unit tests for iqwav.pipeline.controlled_receiver.run_controlled_receiver_pipeline.

These cover the Phase 2K integration of the Phase 2I controlled receiver
chain and the Phase 2J parameter-estimation chain: known or estimated
samples-per-symbol/modulation, optional known CFO, and always-recovered
integer timing phase, over rectangular-pulse BPSK/QPSK waveforms.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform, symbols_to_samples
from iqwav.pipeline import (
    ControlledReceiverResult,
    run_controlled_receiver_pipeline,
)
from iqwav.pipeline import controlled_receiver

N_SYMBOLS = 300


def _bits(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, count)


def _bpsk(samples_per_symbol: int, *, n_symbols: int = N_SYMBOLS, seed: int = 1):
    bits = _bits(n_symbols, seed)
    return bits, bpsk_waveform(bits, samples_per_symbol)


def _qpsk(samples_per_symbol: int, *, n_symbols: int = N_SYMBOLS, seed: int = 2):
    bits = _bits(2 * n_symbols, seed)
    return bits, qpsk_waveform(bits, samples_per_symbol)


# --------------------------------------------------------------------------
# Clean signals, fully known parameters (Phase 2I style)
# --------------------------------------------------------------------------


def test_known_bpsk_parameters_recover_bits():
    bits, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(
        wave, 48000.0, samples_per_symbol=8, modulation="bpsk"
    )
    assert result.status == "complete"
    assert result.samples_per_symbol == 8
    assert result.samples_per_symbol_known is True
    assert result.modulation == "bpsk"
    assert result.modulation_known is True
    assert np.array_equal(result.bits, bits)


def test_known_qpsk_parameters_recover_bits():
    bits, wave = _qpsk(8)
    result = run_controlled_receiver_pipeline(
        wave, 48000.0, samples_per_symbol=8, modulation="qpsk"
    )
    assert result.status == "complete"
    assert result.modulation == "qpsk"
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Clean signals, fully estimated parameters (Phase 2J style)
# --------------------------------------------------------------------------


def test_estimated_bpsk_parameters_recover_bits():
    bits, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(wave, 48000.0)
    assert result.status == "complete"
    assert result.samples_per_symbol == 8
    assert result.samples_per_symbol_known is False
    assert result.modulation == "bpsk"
    assert result.modulation_known is False
    assert np.array_equal(result.bits, bits)


def test_estimated_qpsk_parameters_recover_bits():
    bits, wave = _qpsk(8)
    result = run_controlled_receiver_pipeline(wave, 48000.0)
    assert result.status == "complete"
    assert result.modulation == "qpsk"
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Mixed known/estimated parameters
# --------------------------------------------------------------------------


def test_known_sps_estimated_modulation():
    bits, wave = _qpsk(10)
    result = run_controlled_receiver_pipeline(wave, 48000.0, samples_per_symbol=10)
    assert result.samples_per_symbol_known is True
    assert result.modulation_known is False
    assert result.modulation == "qpsk"
    assert np.array_equal(result.bits, bits)


def test_estimated_sps_known_modulation():
    bits, wave = _bpsk(10)
    result = run_controlled_receiver_pipeline(wave, 48000.0, modulation="bpsk")
    assert result.samples_per_symbol_known is False
    assert result.samples_per_symbol == 10
    assert result.modulation_known is True
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Metadata is always populated
# --------------------------------------------------------------------------


def test_metadata_fields_populated_for_clean_signal():
    _, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(
        wave, 48000.0, samples_per_symbol=8, modulation="bpsk"
    )
    assert result.peak_frequency is not None
    assert result.occupied_bandwidth is not None
    assert result.symbol_rate is not None
    assert result.symbol_rate.samples_per_symbol == 8
    assert result.modulation_estimate is not None
    assert result.timing is not None
    assert result.timing_offset is not None
    assert result.stage_status["peak_frequency"] == "success"
    assert result.stage_status["occupied_bandwidth"] == "success"
    assert result.stage_status["symbol_rate"] == "success"
    assert result.stage_status["timing_recovery"] == "success"
    assert result.stage_status["modulation_classification"] == "success"
    assert result.stage_status["demodulation"] == "success"


# --------------------------------------------------------------------------
# CFO: zero, positive, negative, known
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfo", [37.5, -37.5, 0.0])
def test_known_cfo_signs_are_corrected(cfo):
    bits, wave = _bpsk(10)
    fs = 10000.0
    offset_wave = apply_frequency_offset(wave, fs, cfo)
    result = run_controlled_receiver_pipeline(
        offset_wave,
        fs,
        samples_per_symbol=10,
        modulation="bpsk",
        frequency_offset_hz=cfo,
    )
    assert result.frequency_offset_hz == cfo
    assert result.cfo_corrected_samples is not None
    assert result.stage_status["cfo_correction"] == "success"
    assert np.array_equal(result.bits, bits)


def test_no_cfo_supplied_leaves_correction_fields_none():
    bits, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(
        wave, 48000.0, samples_per_symbol=8, modulation="bpsk"
    )
    assert result.frequency_offset_hz is None
    assert result.cfo_corrected_samples is None
    assert result.stage_status["cfo_correction"] == "skipped"
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Multiple sample rates and SPS values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fs", [1000.0, 44100.0, 2_000_000.0])
def test_multiple_sample_rates(fs):
    bits, wave = _bpsk(10)
    result = run_controlled_receiver_pipeline(wave, fs, samples_per_symbol=10)
    assert result.sample_rate == fs
    assert np.array_equal(result.bits, bits)


@pytest.mark.parametrize("sps", [4, 8, 16, 20])
def test_multiple_sps_values(sps):
    bits, wave = _qpsk(sps)
    result = run_controlled_receiver_pipeline(wave, 48000.0, modulation="qpsk")
    assert result.samples_per_symbol == sps
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Multiple integer timing phases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phase", [0, 1, 3, 7])
def test_multiple_timing_phases(phase):
    bits, wave = _bpsk(8)
    padded = np.concatenate([np.zeros(phase, dtype=complex), wave])
    result = run_controlled_receiver_pipeline(
        padded, 48000.0, samples_per_symbol=8, modulation="bpsk"
    )
    assert result.timing_offset == phase
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Determinism / non-mutation
# --------------------------------------------------------------------------


def test_deterministic_results():
    _, wave = _bpsk(12)
    result_a = run_controlled_receiver_pipeline(wave, 48000.0)
    result_b = run_controlled_receiver_pipeline(wave, 48000.0)
    assert np.array_equal(result_a.bits, result_b.bits)
    assert result_a.samples_per_symbol == result_b.samples_per_symbol
    assert result_a.modulation == result_b.modulation
    assert result_a.status == result_b.status


def test_input_samples_not_mutated():
    _, wave = _bpsk(8)
    original = wave.copy()
    run_controlled_receiver_pipeline(wave, 48000.0)
    assert np.array_equal(wave, original)


def test_cfo_corrected_samples_share_no_memory_with_input():
    _, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(
        wave, 48000.0, samples_per_symbol=8, frequency_offset_hz=10.0
    )
    assert not np.shares_memory(result.cfo_corrected_samples, wave)


# --------------------------------------------------------------------------
# Moderate AWGN
# --------------------------------------------------------------------------


def test_moderate_awgn_bpsk_still_recovers_bits():
    bits, wave = _bpsk(16, n_symbols=500)
    rng = np.random.default_rng(42)
    noisy = add_awgn(wave, 15.0, rng=rng)
    result = run_controlled_receiver_pipeline(noisy, 48000.0)
    assert result.status == "complete"
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Graceful partial results / failure handling
# --------------------------------------------------------------------------


def test_pure_noise_fails_without_fabricating_bits():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(600) + 1j * rng.standard_normal(600)
    result = run_controlled_receiver_pipeline(noise, 48000.0)
    assert result.status == "failed"
    assert result.failure_reason is not None
    assert result.bits is None
    assert result.stage_status["symbol_rate"] == "failed"
    assert "symbol_rate" in result.failure_reasons


def test_known_sps_survives_symbol_rate_estimation_failure():
    # A constant-symbol waveform has no transitions, so Phase 2E's
    # symbol-rate estimator fails, but a caller-supplied
    # samples_per_symbol lets the rest of the chain still proceed.
    sps = 8
    symbols = np.ones(200, dtype=complex)
    wave = symbols_to_samples(symbols, sps)
    result = run_controlled_receiver_pipeline(wave, 48000.0, samples_per_symbol=sps)
    assert result.samples_per_symbol == sps
    assert result.samples_per_symbol_known is True
    assert result.stage_status["symbol_rate"] == "failed"
    # A degenerate, single-point constellation is modulation-ambiguous,
    # so bits must not be fabricated even though timing recovery ran.
    # (The constant waveform also has no spectral content away from DC,
    # so the always-computed Phase 2A/2B metadata stages fail too.)
    assert result.bits is None
    assert result.status in ("failed", "partial")


def test_ambiguous_modulation_without_known_modulation_does_not_fabricate_bits():
    sps = 8
    symbols = np.ones(200, dtype=complex)
    wave = symbols_to_samples(symbols, sps)
    result = run_controlled_receiver_pipeline(wave, 48000.0, samples_per_symbol=sps)
    assert result.modulation is None
    assert result.bits is None
    assert result.status in ("failed", "partial")
    # Demodulation was skipped (never attempted), not failed, so it must
    # not appear in failure_reasons under the documented contract.
    assert result.stage_status["demodulation"] == "skipped"
    assert "demodulation" not in result.failure_reasons
    assert result.failure_reason is not None


def test_missing_sps_skips_timing_without_failure_reasons_entry():
    # samples_per_symbol cannot be estimated (constant block) and was not
    # supplied, so timing recovery never runs: it must be reported as
    # "skipped", not "failed", and must not appear in failure_reasons.
    tiny = np.array([1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j])
    result = run_controlled_receiver_pipeline(tiny, 1000.0)
    assert result.samples_per_symbol is None
    assert result.stage_status["timing_recovery"] == "skipped"
    assert "timing_recovery" not in result.failure_reasons
    assert result.bits is None
    assert result.failure_reason is not None


def test_too_short_block_reports_failure_not_fabricated_bits():
    tiny = np.array([1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j])
    result = run_controlled_receiver_pipeline(tiny, 1000.0)
    assert result.status in ("failed", "partial")
    assert result.bits is None
    assert result.failure_reason is not None


# --------------------------------------------------------------------------
# Result dataclass conventions
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Demodulation-failure contract
# --------------------------------------------------------------------------


def test_demodulation_failure_reports_partial_result_without_raising(monkeypatch):
    def _forced_failure(*args, **kwargs):
        raise ValueError("forced demodulation failure")

    monkeypatch.setattr(controlled_receiver, "bpsk_demodulate", _forced_failure)

    _, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(
        wave, 48000.0, samples_per_symbol=8, modulation="bpsk"
    )
    assert result.bits is None
    assert result.stage_status["demodulation"] == "failed"
    assert "demodulation" in result.failure_reasons
    assert "forced demodulation failure" in result.failure_reasons["demodulation"]


def test_result_is_frozen():
    _, wave = _bpsk(8)
    result = run_controlled_receiver_pipeline(wave, 48000.0)
    assert isinstance(result, ControlledReceiverResult)
    with pytest.raises(FrozenInstanceError):
        result.status = "complete"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rejects_real_valued_samples():
    with pytest.raises(ValueError):
        run_controlled_receiver_pipeline(np.ones(100), 1000.0)


def test_rejects_non_positive_sample_rate():
    _, wave = _bpsk(8)
    with pytest.raises(ValueError):
        run_controlled_receiver_pipeline(wave, 0.0)


def test_rejects_empty_samples():
    with pytest.raises(ValueError):
        run_controlled_receiver_pipeline(np.array([], dtype=complex), 1000.0)


def test_rejects_invalid_samples_per_symbol():
    _, wave = _bpsk(8)
    with pytest.raises(ValueError):
        run_controlled_receiver_pipeline(wave, 48000.0, samples_per_symbol=0)


def test_rejects_invalid_modulation():
    _, wave = _bpsk(8)
    with pytest.raises(ValueError):
        run_controlled_receiver_pipeline(wave, 48000.0, modulation="16qam")


def test_rejects_non_finite_frequency_offset():
    _, wave = _bpsk(8)
    with pytest.raises(ValueError):
        run_controlled_receiver_pipeline(
            wave, 48000.0, frequency_offset_hz=float("nan")
        )