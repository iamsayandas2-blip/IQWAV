"""Unit tests for iqwav.estimation.symbol_rate.estimate_symbol_rate.

These cover the bounded, controlled-signal symbol-rate primitive only:
rectangular-pulse BPSK/QPSK waveforms built by :mod:`iqwav.modulation` with
a known integer ``samples_per_symbol``, so the ground-truth symbol rate is
``FS / samples_per_symbol`` exactly. No blind baud-rate estimation, timing
recovery, carrier recovery, or modulation recognition is involved.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset
from iqwav.estimation import SymbolRateEstimate, estimate_symbol_rate
from iqwav.modulation import bpsk_waveform, qpsk_waveform, symbols_to_samples

FS = 48000.0
N_SYMBOLS = 400  # long enough that every searched period sees many symbols


def _bits(count: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-random bits, so transitions are dense."""
    return np.random.default_rng(seed).integers(0, 2, count)


def _bpsk(samples_per_symbol: int, *, n_symbols: int = N_SYMBOLS, seed: int = 1):
    return bpsk_waveform(_bits(n_symbols, seed), samples_per_symbol)


def _qpsk(samples_per_symbol: int, *, n_symbols: int = N_SYMBOLS, seed: int = 2):
    return qpsk_waveform(_bits(2 * n_symbols, seed), samples_per_symbol)


# --------------------------------------------------------------------------
# Known BPSK / QPSK symbol rates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sps", [2, 3, 4, 5, 8, 10, 16, 25, 32, 50, 64])
def test_clean_bpsk_recovers_known_symbol_rate(sps):
    result = estimate_symbol_rate(_bpsk(sps, seed=10 + sps), FS)
    assert result.samples_per_symbol == sps
    assert result.symbol_rate_hz == pytest.approx(FS / sps)


@pytest.mark.parametrize("sps", [2, 4, 8, 10, 16, 32])
def test_clean_qpsk_recovers_known_symbol_rate(sps):
    result = estimate_symbol_rate(_qpsk(sps, seed=100 + sps), FS)
    assert result.samples_per_symbol == sps
    assert result.symbol_rate_hz == pytest.approx(FS / sps)


@pytest.mark.parametrize("sample_rate", [8000.0, 44100.0, 48000.0, 1_000_000.0])
def test_symbol_rate_scales_with_sample_rate(sample_rate):
    result = estimate_symbol_rate(_bpsk(8, seed=3), sample_rate)
    assert result.samples_per_symbol == 8
    assert result.symbol_rate_hz == pytest.approx(sample_rate / 8)
    assert result.sample_rate_hz == sample_rate


def test_zero_error_against_known_ground_truth():
    for sps in (4, 6, 8, 12, 20):
        truth = FS / sps
        result = estimate_symbol_rate(_bpsk(sps, seed=200 + sps), FS)
        assert abs(result.symbol_rate_hz - truth) == 0.0


def test_clean_waveform_is_perfectly_concentrated():
    result = estimate_symbol_rate(_bpsk(8, seed=4), FS)
    assert result.quality == pytest.approx(1.0)
    assert result.concentration == pytest.approx(1.0)
    assert result.boundary_offset == 0
    assert result.symbol_count == N_SYMBOLS


def test_real_valued_rectangular_waveform():
    symbols = np.where(_bits(300, 5) == 0, 1.0, -1.0)
    samples = symbols_to_samples(symbols, 8)
    result = estimate_symbol_rate(samples, FS)
    assert result.samples_per_symbol == 8
    assert result.quality == pytest.approx(1.0)


def test_reported_resolution_is_the_integer_period_grid():
    result = estimate_symbol_rate(_bpsk(10, seed=6), FS)
    expected = FS / 10 - FS / 11
    assert result.symbol_rate_resolution_hz == pytest.approx(expected)
    assert result.symbol_rate_resolution_hz == pytest.approx(FS / (10 * 11))


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0, 5.0])
@pytest.mark.parametrize("sps", [4, 8, 16])
def test_noisy_bpsk_still_recovers_known_rate(snr_db, sps):
    noisy = add_awgn(_bpsk(sps, seed=7), snr_db=snr_db, rng=np.random.default_rng(11))
    result = estimate_symbol_rate(noisy, FS)
    assert result.samples_per_symbol == sps
    assert result.symbol_rate_hz == pytest.approx(FS / sps)


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0])
@pytest.mark.parametrize("sps", [4, 8, 16])
def test_noisy_qpsk_still_recovers_known_rate(snr_db, sps):
    noisy = add_awgn(_qpsk(sps, seed=8), snr_db=snr_db, rng=np.random.default_rng(12))
    result = estimate_symbol_rate(noisy, FS)
    assert result.samples_per_symbol == sps
    assert result.symbol_rate_hz == pytest.approx(FS / sps)


def test_quality_decreases_as_noise_increases():
    clean = _bpsk(8, seed=9)
    qualities = [
        estimate_symbol_rate(
            add_awgn(clean, snr_db=snr, rng=np.random.default_rng(13)), FS
        ).quality
        for snr in (30.0, 20.0, 10.0, 5.0)
    ]
    assert estimate_symbol_rate(clean, FS).quality > qualities[0]
    assert qualities == sorted(qualities, reverse=True)
    assert all(0.0 < q < 1.0 for q in qualities)


def test_noisy_boundary_phase_is_unchanged():
    noisy = add_awgn(_bpsk(8, seed=14), snr_db=15.0, rng=np.random.default_rng(15))
    assert estimate_symbol_rate(noisy, FS).boundary_offset == 0


# --------------------------------------------------------------------------
# Adversarial finite-data period-selection cases
# --------------------------------------------------------------------------


def test_adversarial_finite_transition_pattern_does_not_promote_multiple():
    """True SPS=8 must beat a misleading finite-data SPS=16 candidate."""
    symbols = np.array(
        [0, 0, 1, 0, 1, 0, 0, 1, 1, 1,
         1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    )
    samples = symbols_to_samples(np.where(symbols == 0, 1.0, -1.0), 8)

    result = estimate_symbol_rate(samples, FS)

    assert result.samples_per_symbol == 8
    assert result.quality == pytest.approx(1.0)
    assert result.effective_transitions >= 2.0


def test_genuinely_ambiguous_observable_period_can_choose_multiple():
    """Transitions only every second symbol make 16 the observable period."""
    symbols = np.array([0, 0, 1, 1] * 20)
    samples = symbols_to_samples(np.where(symbols == 0, 1.0, -1.0), 8)

    result = estimate_symbol_rate(samples, FS)

    assert result.samples_per_symbol == 16
    assert result.quality == pytest.approx(1.0)


def test_relaxed_quality_ratio_does_not_reintroduce_false_multiple():
    """A relaxed ratio must not override a substantially better divisor."""
    result = estimate_symbol_rate(_bpsk(8, seed=31), FS, quality_ratio=0.05)

    assert result.samples_per_symbol == 8


# --------------------------------------------------------------------------
# Result object contract
# --------------------------------------------------------------------------


def test_result_type_and_documented_fields():
    result = estimate_symbol_rate(_bpsk(8, seed=16), FS)
    assert isinstance(result, SymbolRateEstimate)
    assert set(vars(result)) == {
        "symbol_rate_hz",
        "samples_per_symbol",
        "sample_rate_hz",
        "symbol_rate_resolution_hz",
        "quality",
        "concentration",
        "boundary_offset",
        "symbol_count",
        "effective_transitions",
        "searched_samples_per_symbol",
    }
    assert isinstance(result.symbol_rate_hz, float)
    assert isinstance(result.samples_per_symbol, int)
    assert isinstance(result.sample_rate_hz, float)
    assert isinstance(result.symbol_rate_resolution_hz, float)
    assert isinstance(result.quality, float)
    assert isinstance(result.concentration, float)
    assert isinstance(result.boundary_offset, int)
    assert isinstance(result.symbol_count, int)
    assert isinstance(result.effective_transitions, float)
    assert isinstance(result.searched_samples_per_symbol, tuple)


def test_result_is_frozen():
    result = estimate_symbol_rate(_bpsk(8, seed=17), FS)
    with pytest.raises(FrozenInstanceError):
        result.symbol_rate_hz = 1.0
    with pytest.raises(FrozenInstanceError):
        result.samples_per_symbol = 3


def test_repeated_calls_are_deterministic():
    samples = add_awgn(_bpsk(8, seed=18), snr_db=10.0, rng=np.random.default_rng(19))
    first = estimate_symbol_rate(samples, FS)
    second = estimate_symbol_rate(samples, FS)
    third = estimate_symbol_rate(samples.copy(), FS)
    assert first == second == third


def test_input_samples_are_not_modified():
    samples = add_awgn(_bpsk(8, seed=20), snr_db=10.0, rng=np.random.default_rng(21))
    original = samples.copy()
    estimate_symbol_rate(samples, FS)
    assert np.array_equal(samples, original)


def test_list_input_is_accepted():
    samples = _bpsk(4, n_symbols=50, seed=22)
    result = estimate_symbol_rate(samples.tolist(), FS)
    assert result.samples_per_symbol == 4


def test_searched_range_is_echoed():
    result = estimate_symbol_rate(
        _bpsk(8, seed=23), FS, min_samples_per_symbol=3, max_samples_per_symbol=20
    )
    assert result.searched_samples_per_symbol == (3, 20)


def test_searched_range_is_capped_by_block_length():
    samples = _bpsk(2, n_symbols=20, seed=24)  # N = 40 -> max period (40-1)//4 = 9
    result = estimate_symbol_rate(samples, FS)
    assert result.searched_samples_per_symbol == (2, 9)
    assert result.samples_per_symbol == 2


# --------------------------------------------------------------------------
# Boundary phase, symbol count, search bounds
# --------------------------------------------------------------------------


def test_leading_partial_symbol_shifts_boundary_offset():
    trimmed = _bpsk(8, n_symbols=300, seed=25)[3:]
    result = estimate_symbol_rate(trimmed, FS)
    assert result.samples_per_symbol == 8
    assert result.boundary_offset == 5  # (0 - 3) mod 8
    assert result.symbol_count == 299


@pytest.mark.parametrize("trim", [0, 1, 2, 3, 4, 5, 6, 7])
def test_boundary_offset_tracks_the_trim(trim):
    samples = _bpsk(8, n_symbols=200, seed=26)[trim:]
    result = estimate_symbol_rate(samples, FS)
    assert result.samples_per_symbol == 8
    assert result.boundary_offset == (-trim) % 8


def test_true_period_above_search_bound_returns_a_divisor():
    result = estimate_symbol_rate(
        _bpsk(32, n_symbols=200, seed=27), FS, max_samples_per_symbol=8
    )
    assert 32 % result.samples_per_symbol == 0
    assert result.samples_per_symbol == 8


def test_narrow_search_range_pins_the_period():
    result = estimate_symbol_rate(
        _bpsk(8, seed=28), FS, min_samples_per_symbol=8, max_samples_per_symbol=8
    )
    assert result.samples_per_symbol == 8
    assert result.searched_samples_per_symbol == (8, 8)


def test_minimum_supported_block_length():
    symbols = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    result = estimate_symbol_rate(symbols_to_samples(symbols, 2), FS)  # N = 12
    assert result.samples_per_symbol == 2
    assert result.symbol_rate_hz == pytest.approx(FS / 2)


def test_symbol_count_matches_the_block():
    result = estimate_symbol_rate(_bpsk(16, n_symbols=37, seed=29), FS)
    assert result.samples_per_symbol == 16
    assert result.symbol_count == 37


# --------------------------------------------------------------------------
# Estimator parameters
# --------------------------------------------------------------------------


def test_quality_ratio_of_one_still_finds_the_clean_period():
    result = estimate_symbol_rate(_bpsk(8, seed=30), FS, quality_ratio=1.0)
    assert result.samples_per_symbol == 8


def test_very_low_quality_ratio_does_not_prefer_a_multiple():
    # A relaxed ratio cannot override the divisor-consistency guard.
    result = estimate_symbol_rate(_bpsk(8, seed=31), FS, quality_ratio=0.05)
    assert result.samples_per_symbol == 8


def test_tightened_min_quality_rejects_a_noisy_block():
    noisy = add_awgn(_bpsk(8, seed=32), snr_db=20.0, rng=np.random.default_rng(33))
    assert estimate_symbol_rate(noisy, FS).quality < 0.9
    with pytest.raises(ValueError, match="min_quality"):
        estimate_symbol_rate(noisy, FS, min_quality=0.9)


def test_min_quality_zero_returns_a_low_quality_result():
    noise = np.random.default_rng(34).normal(size=4096) + 1j * np.random.default_rng(
        35
    ).normal(size=4096)
    result = estimate_symbol_rate(noise, FS, min_quality=0.0)
    assert result.quality < 0.02  # honest, but useless: caller must check


def test_returned_quality_respects_min_quality():
    for snr in (20.0, 10.0, 5.0):
        noisy = add_awgn(_bpsk(8, seed=36), snr_db=snr, rng=np.random.default_rng(37))
        result = estimate_symbol_rate(noisy, FS, min_quality=0.05)
        assert result.quality >= 0.05


# --------------------------------------------------------------------------
# Degenerate and unidentifiable signals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(64),
        np.zeros(64, dtype=complex),
        np.full(64, 2.5),
        np.full(64, 1.0 + 1.0j),
    ],
    ids=["zeros-real", "zeros-complex", "constant-real", "constant-complex"],
)
def test_constant_signal_is_rejected(samples):
    with pytest.raises(ValueError, match="never change value"):
        estimate_symbol_rate(samples, FS)


def test_single_transition_block_is_rejected():
    step = np.concatenate([np.ones(30), -np.ones(30)])
    with pytest.raises(ValueError, match="effective transition"):
        estimate_symbol_rate(step, FS)


def test_pure_noise_is_rejected():
    noise = np.random.default_rng(38).normal(size=4096) + 1j * np.random.default_rng(
        39
    ).normal(size=4096)
    with pytest.raises(ValueError, match="min_quality"):
        estimate_symbol_rate(noise, FS)


def test_one_sample_per_symbol_is_rejected():
    # sps = 1 has no intra-symbol structure; it looks like a random sequence
    # and no candidate period in [2, 64] explains it.
    samples = bpsk_waveform(_bits(4000, 40), 1)
    with pytest.raises(ValueError, match="min_quality"):
        estimate_symbol_rate(samples, FS)


def test_strictly_monotonic_ramp_is_rejected():
    with pytest.raises(ValueError, match="above the level expected by chance"):
        estimate_symbol_rate(np.arange(9.0), FS)


# --------------------------------------------------------------------------
# Invalid inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_rate", [0.0, -1.0, -48000.0, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        estimate_symbol_rate(_bpsk(8, n_symbols=50, seed=41), sample_rate)


def test_empty_samples():
    with pytest.raises(ValueError, match="at least one value"):
        estimate_symbol_rate(np.array([]), FS)


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_too_few_samples(size):
    with pytest.raises(ValueError, match="at least 9 values"):
        estimate_symbol_rate(np.arange(float(size)), FS)


@pytest.mark.parametrize(
    "samples",
    [np.zeros((4, 16)), np.zeros((16, 1)), np.zeros((2, 2, 4)), np.array(1.0)],
    ids=["2d", "column", "3d", "scalar"],
)
def test_non_one_dimensional_samples(samples):
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_symbol_rate(samples, FS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_real_samples(bad):
    samples = _bpsk(8, n_symbols=50, seed=42).real.copy()
    samples[17] = bad
    with pytest.raises(ValueError, match="finite"):
        estimate_symbol_rate(samples, FS)


@pytest.mark.parametrize(
    "bad", [complex(float("nan"), 0.0), complex(0.0, float("inf"))]
)
def test_non_finite_complex_samples(bad):
    samples = _bpsk(8, n_symbols=50, seed=43).astype(complex)
    samples[23] = bad
    with pytest.raises(ValueError, match="finite"):
        estimate_symbol_rate(samples, FS)


@pytest.mark.parametrize("value", [1, 0, -2])
def test_min_samples_per_symbol_below_two(value):
    with pytest.raises(ValueError, match="min_samples_per_symbol must be >= 2"):
        estimate_symbol_rate(_bpsk(8, seed=44), FS, min_samples_per_symbol=value)


@pytest.mark.parametrize("value", [2.5, 4.0, "4", None, True])
def test_non_integer_min_samples_per_symbol(value):
    with pytest.raises(ValueError, match="min_samples_per_symbol must be an integer"):
        estimate_symbol_rate(_bpsk(8, seed=45), FS, min_samples_per_symbol=value)


def test_max_below_min_samples_per_symbol():
    with pytest.raises(ValueError, match="max_samples_per_symbol must be >= 8"):
        estimate_symbol_rate(
            _bpsk(8, seed=46),
            FS,
            min_samples_per_symbol=8,
            max_samples_per_symbol=7,
        )


@pytest.mark.parametrize("value", [16.0, "16", None])
def test_non_integer_max_samples_per_symbol(value):
    with pytest.raises(ValueError, match="max_samples_per_symbol must be an integer"):
        estimate_symbol_rate(_bpsk(8, seed=47), FS, max_samples_per_symbol=value)


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_invalid_quality_ratio(value):
    with pytest.raises(ValueError, match="quality_ratio"):
        estimate_symbol_rate(_bpsk(8, seed=48), FS, quality_ratio=value)


@pytest.mark.parametrize("value", [None, "x", [0.5]])
def test_non_numeric_quality_ratio(value):
    with pytest.raises(ValueError, match="quality_ratio"):
        estimate_symbol_rate(_bpsk(8, seed=49), FS, quality_ratio=value)


@pytest.mark.parametrize("value", [-0.1, 1.0, 2.0, float("nan"), float("-inf")])
def test_invalid_min_quality(value):
    with pytest.raises(ValueError, match="min_quality"):
        estimate_symbol_rate(_bpsk(8, seed=50), FS, min_quality=value)


@pytest.mark.parametrize("value", [None, "x", [0.1]])
def test_non_numeric_min_quality(value):
    with pytest.raises(ValueError, match="min_quality"):
        estimate_symbol_rate(_bpsk(8, seed=51), FS, min_quality=value)


def test_block_too_short_for_requested_search_range():
    samples = _bpsk(2, n_symbols=10, seed=52)  # N = 20 -> supports periods <= 4
    with pytest.raises(ValueError, match="supports candidate symbol periods"):
        estimate_symbol_rate(samples, FS, min_samples_per_symbol=8)


def test_validation_order_sample_rate_before_samples():
    with pytest.raises(ValueError, match="sample_rate"):
        estimate_symbol_rate(np.array([]), 0.0)


# --------------------------------------------------------------------------
# Robustness to a small residual carrier offset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfo_fraction", [0.0, 0.001, 0.005, 0.01])
def test_small_residual_cfo_does_not_bias_the_estimate(cfo_fraction):
    sps = 8
    samples = _qpsk(sps, seed=53)
    rotated = apply_frequency_offset(samples, FS, cfo_fraction * FS / sps)
    result = estimate_symbol_rate(rotated, FS)
    assert result.samples_per_symbol == sps
    assert result.symbol_rate_hz == pytest.approx(FS / sps)
    assert result.quality > 0.9


def test_large_residual_cfo_lowers_quality():
    sps = 8
    samples = _qpsk(sps, seed=54)
    mild = estimate_symbol_rate(
        apply_frequency_offset(samples, FS, 0.01 * FS / sps), FS
    )
    harsh = estimate_symbol_rate(
        apply_frequency_offset(samples, FS, 0.25 * FS / sps), FS
    )
    assert harsh.samples_per_symbol == sps
    assert harsh.quality < mild.quality
