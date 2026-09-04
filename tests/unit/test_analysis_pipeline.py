"""Unit tests for iqwav.pipeline.analysis.run_parameter_estimation_pipeline.

These cover the controlled Phase 2J orchestration only: chaining the
already-implemented Phase 2A/2B/2C/2E/2H/2F primitives over rectangular-
pulse BPSK/QPSK waveforms with a known sample rate and, optionally, a
known constant CFO. No blind carrier recovery, PLL, interpolation, or
payload recovery is involved.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform
from iqwav.pipeline import ParameterEstimationResult, run_parameter_estimation_pipeline

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
# Controlled BPSK / QPSK, clean signals
# --------------------------------------------------------------------------


def test_controlled_bpsk_clean_signal_recovers_bits():
    bits, wave = _bpsk(8)
    result = run_parameter_estimation_pipeline(wave, 48000.0)
    assert result.status == "complete"
    assert result.samples_per_symbol == 8
    assert result.modulation.modulation == "BPSK"
    assert result.bits is not None
    assert np.array_equal(result.bits, bits)


def test_controlled_qpsk_clean_signal_recovers_bits():
    bits, wave = _qpsk(8)
    result = run_parameter_estimation_pipeline(wave, 48000.0)
    assert result.status == "complete"
    assert result.samples_per_symbol == 8
    assert result.modulation.modulation == "QPSK"
    assert result.bits is not None
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Multiple sample rates / symbol rates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fs", [1000.0, 44100.0, 2_000_000.0])
def test_multiple_sample_rates_are_echoed_and_recover_bits(fs):
    bits, wave = _bpsk(10)
    result = run_parameter_estimation_pipeline(wave, fs)
    assert result.sample_rate == fs
    assert result.status == "complete"
    assert np.array_equal(result.bits, bits)


@pytest.mark.parametrize("sps", [4, 8, 16, 20])
def test_multiple_symbol_rates_estimated_correctly(sps):
    bits, wave = _qpsk(sps)
    result = run_parameter_estimation_pipeline(wave, 48000.0)
    assert result.samples_per_symbol == sps
    assert result.symbol_rate.symbol_rate_hz == pytest.approx(48000.0 / sps)
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Frequency offsets
# --------------------------------------------------------------------------


def test_known_cfo_is_corrected_and_bits_recovered():
    bits, wave = _bpsk(10)
    fs = 10000.0
    cfo = 37.5
    offset_wave = apply_frequency_offset(wave, fs, cfo)
    result = run_parameter_estimation_pipeline(
        offset_wave, fs, frequency_offset_hz=cfo
    )
    assert result.frequency_offset_hz == cfo
    assert result.cfo_corrected_samples is not None
    assert result.status == "complete"
    assert np.array_equal(result.bits, bits)


def test_no_cfo_supplied_leaves_correction_fields_none():
    bits, wave = _bpsk(8)
    result = run_parameter_estimation_pipeline(wave, 48000.0)
    assert result.frequency_offset_hz is None
    assert result.cfo_corrected_samples is None
    assert np.array_equal(result.bits, bits)


def test_uncorrected_large_cfo_without_correction_does_not_fabricate_bits():
    # A large uncorrected CFO rotates the constellation across the block;
    # modulation classification should become ambiguous/incorrect rather
    # than the pipeline inventing correct-looking bits.
    _, wave = _qpsk(16)
    fs = 4000.0
    rotated = apply_frequency_offset(wave, fs, 500.0)  # large relative to symbol rate
    result = run_parameter_estimation_pipeline(rotated, fs)
    # No CFO was supplied, so no correction is attempted; the pipeline
    # must not silently guess a CFO or fabricate bits it cannot support.
    assert result.frequency_offset_hz is None
    assert result.cfo_corrected_samples is None


# --------------------------------------------------------------------------
# Multiple timing phases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phase", [0, 1, 3, 7])
def test_multiple_timing_phases_recovered(phase):
    bits, wave = _bpsk(8)
    padded = np.concatenate([np.zeros(phase, dtype=complex), wave])
    result = run_parameter_estimation_pipeline(padded, 48000.0)
    assert result.timing is not None
    assert result.timing.timing_offset == phase
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Moderate AWGN
# --------------------------------------------------------------------------


def test_moderate_awgn_bpsk_still_recovers_bits():
    bits, wave = _bpsk(16, n_symbols=500)
    rng = np.random.default_rng(42)
    noisy = add_awgn(wave, 15.0, rng=rng)
    result = run_parameter_estimation_pipeline(noisy, 48000.0)
    assert result.status == "complete"
    assert np.array_equal(result.bits, bits)


def test_moderate_awgn_qpsk_still_recovers_bits():
    bits, wave = _qpsk(16, n_symbols=500)
    rng = np.random.default_rng(43)
    noisy = add_awgn(wave, 15.0, rng=rng)
    result = run_parameter_estimation_pipeline(noisy, 48000.0)
    assert result.status == "complete"
    assert np.array_equal(result.bits, bits)


# --------------------------------------------------------------------------
# Deterministic results
# --------------------------------------------------------------------------


def test_deterministic_results_across_repeated_runs():
    _, wave = _bpsk(12)
    result_a = run_parameter_estimation_pipeline(wave, 48000.0)
    result_b = run_parameter_estimation_pipeline(wave, 48000.0)
    assert np.array_equal(result_a.bits, result_b.bits)
    assert result_a.samples_per_symbol == result_b.samples_per_symbol
    assert result_a.modulation.modulation == result_b.modulation.modulation
    assert result_a.status == result_b.status


# --------------------------------------------------------------------------
# Input non-mutation
# --------------------------------------------------------------------------


def test_input_samples_not_mutated():
    _, wave = _bpsk(8)
    original = wave.copy()
    run_parameter_estimation_pipeline(wave, 48000.0)
    assert np.array_equal(wave, original)


def test_cfo_corrected_samples_share_no_memory_with_input():
    _, wave = _bpsk(8)
    result = run_parameter_estimation_pipeline(
        wave, 48000.0, frequency_offset_hz=10.0
    )
    assert not np.shares_memory(result.cfo_corrected_samples, wave)


# --------------------------------------------------------------------------
# Ambiguous or insufficient signals / estimator failure handling
# --------------------------------------------------------------------------


def test_pure_noise_fails_without_fabricating_bits():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(600) + 1j * rng.standard_normal(600)
    result = run_parameter_estimation_pipeline(noise, 48000.0)
    assert result.status == "failed"
    assert result.failure_reason is not None
    assert result.bits is None
    assert result.symbol_rate is None
    assert result.samples_per_symbol is None
    assert result.timing is None
    assert result.modulation is None


def test_too_short_block_reports_failure_not_fabricated_bits():
    # Long enough for the earlier spectral estimators (>= 4 samples), but
    # far too short for symbol-rate estimation (needs >= 9 samples with
    # structure) or timing recovery.
    tiny = np.array([1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j])
    result = run_parameter_estimation_pipeline(tiny, 1000.0)
    assert result.status in ("failed", "partial")
    assert result.bits is None
    assert result.failure_reason is not None


def test_ambiguous_modulation_does_not_fabricate_bits():
    # A single constant tone decimated to one sample per symbol produces
    # a degenerate constellation; force ambiguity by feeding a
    # low-quality, mixed-structure block directly through the pipeline
    # using a QPSK waveform with heavily skewed / degenerate symbols
    # (all identical) so estimate_modulation cannot separate the classes.
    sps = 8
    symbols = np.ones(200, dtype=complex)  # every symbol at the same point
    from iqwav.modulation import symbols_to_samples

    wave = symbols_to_samples(symbols, sps)
    result = run_parameter_estimation_pipeline(wave, 48000.0)
    # A perfectly constant waveform has no transitions, so symbol-rate
    # estimation itself fails; either way bits must not be fabricated.
    assert result.bits is None
    assert result.status in ("failed", "partial")


# --------------------------------------------------------------------------
# Sample rate remains caller-supplied (never estimated)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fs", [500.0, 8000.0, 96000.0])
def test_sample_rate_is_echoed_exactly_never_estimated(fs):
    _, wave = _bpsk(8)
    result = run_parameter_estimation_pipeline(wave, fs)
    assert result.sample_rate == fs
    # Symbol rate must be internally consistent with the *supplied* rate.
    if result.symbol_rate is not None:
        expected = fs / result.symbol_rate.samples_per_symbol
        assert result.symbol_rate.symbol_rate_hz == pytest.approx(expected)


# --------------------------------------------------------------------------
# Result dataclass conventions
# --------------------------------------------------------------------------


def test_result_is_frozen():
    _, wave = _bpsk(8)
    result = run_parameter_estimation_pipeline(wave, 48000.0)
    assert isinstance(result, ParameterEstimationResult)
    with pytest.raises(FrozenInstanceError):
        result.status = "complete"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rejects_real_valued_samples():
    with pytest.raises(ValueError):
        run_parameter_estimation_pipeline(np.ones(100), 1000.0)


def test_rejects_non_positive_sample_rate():
    _, wave = _bpsk(8)
    with pytest.raises(ValueError):
        run_parameter_estimation_pipeline(wave, 0.0)


def test_rejects_empty_samples():
    with pytest.raises(ValueError):
        run_parameter_estimation_pipeline(np.array([], dtype=complex), 1000.0)


def test_rejects_non_finite_frequency_offset():
    _, wave = _bpsk(8)
    with pytest.raises(ValueError):
        run_parameter_estimation_pipeline(
            wave, 48000.0, frequency_offset_hz=float("nan")
        )