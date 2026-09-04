"""Unit tests for bounded symbol-timing recovery in iqwav.synchronization.

These cover the narrow, controlled case only: rectangular-pulse BPSK and
QPSK sample blocks, a known integer samples-per-symbol, one constant
timing offset, and an integer timing-phase search bounded to a single
symbol period. No symbol-rate estimation, CFO estimation, carrier
recovery, interpolation, or timing loop is involved.
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn, apply_phase_offset
from iqwav.estimation import estimate_modulation, estimate_symbol_rate
from iqwav.modulation import (
    bpsk_modulate,
    bpsk_waveform,
    qpsk_modulate,
    qpsk_waveform,
    symbols_to_samples,
)
from iqwav.synchronization import SymbolTimingRecovery, recover_symbol_timing

# The search is entirely in samples; this rate only appears where a
# neighbouring estimator needs one.
FS = 48000.0
SPS = 8
N_SYMBOLS = 64
ALL_SPS = [1, 2, 3, 4, 5, 8, 10, 16]


def _bits(count, seed):
    """Reproducible bit stream."""
    return np.random.default_rng(seed).integers(0, 2, count, dtype=np.int64)


def _expected_offset(drop, samples_per_symbol):
    """Boundary phase left behind after discarding ``drop`` leading samples."""
    return (-drop) % samples_per_symbol


def _skipped_symbols(drop, samples_per_symbol):
    """Whole symbols lost when ``drop`` leading samples are discarded."""
    return -(-drop // samples_per_symbol)


# ---------------------------------------------------------------------------
# Zero timing offset
# ---------------------------------------------------------------------------


def test_zero_timing_offset_bpsk_selects_phase_zero():
    bits = _bits(N_SYMBOLS, 1)
    result = recover_symbol_timing(bpsk_waveform(bits, SPS), SPS)
    assert result.timing_offset == 0
    assert result.first_symbol_index == SPS // 2
    assert result.quality == 1.0
    assert result.symbol_count == N_SYMBOLS
    np.testing.assert_array_equal(result.symbols, bpsk_modulate(bits))


def test_zero_timing_offset_qpsk_selects_phase_zero():
    bits = _bits(2 * N_SYMBOLS, 2)
    result = recover_symbol_timing(qpsk_waveform(bits, SPS), SPS)
    assert result.timing_offset == 0
    assert result.quality == 1.0
    assert result.symbol_count == N_SYMBOLS
    np.testing.assert_array_equal(result.symbols, qpsk_modulate(bits))


def test_zero_timing_offset_is_reported_as_uniquely_identifiable():
    result = recover_symbol_timing(bpsk_waveform(_bits(N_SYMBOLS, 3), SPS), SPS)
    assert result.tied_offsets == (0,)
    assert result.margin > 0.0


# ---------------------------------------------------------------------------
# Integer timing offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 20])
def test_integer_timing_offset_bpsk_recovered(drop):
    bits = _bits(N_SYMBOLS, 4)
    symbols = bpsk_modulate(bits)
    result = recover_symbol_timing(bpsk_waveform(bits, SPS)[drop:], SPS)
    skipped = _skipped_symbols(drop, SPS)
    assert result.timing_offset == _expected_offset(drop, SPS)
    assert result.quality == 1.0
    assert result.symbol_count == N_SYMBOLS - skipped
    np.testing.assert_array_equal(result.symbols, symbols[skipped:])


@pytest.mark.parametrize("drop", [1, 3, 4, 7, 8, 13, 24])
def test_integer_timing_offset_qpsk_recovered(drop):
    bits = _bits(2 * N_SYMBOLS, 5)
    symbols = qpsk_modulate(bits)
    result = recover_symbol_timing(qpsk_waveform(bits, SPS)[drop:], SPS)
    skipped = _skipped_symbols(drop, SPS)
    assert result.timing_offset == _expected_offset(drop, SPS)
    assert result.quality == 1.0
    np.testing.assert_array_equal(result.symbols, symbols[skipped:])


@pytest.mark.parametrize("drop", [1, 5, 9, 17])
def test_offset_of_a_whole_symbol_period_is_reported_as_phase_zero(drop):
    # Dropping an exact multiple of the period leaves the boundary at 0.
    bits = _bits(N_SYMBOLS, 6)
    whole = drop * SPS
    result = recover_symbol_timing(bpsk_waveform(bits, SPS)[whole:], SPS)
    assert result.timing_offset == 0
    assert result.symbol_count == N_SYMBOLS - drop
    np.testing.assert_array_equal(result.symbols, bpsk_modulate(bits)[drop:])


# ---------------------------------------------------------------------------
# Every timing phase, for several samples-per-symbol values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("samples_per_symbol", ALL_SPS)
def test_every_timing_phase_recovered_bpsk(samples_per_symbol):
    bits = _bits(N_SYMBOLS, 7)
    symbols = bpsk_modulate(bits)
    waveform = bpsk_waveform(bits, samples_per_symbol)
    for drop in range(samples_per_symbol):
        result = recover_symbol_timing(waveform[drop:], samples_per_symbol)
        skipped = _skipped_symbols(drop, samples_per_symbol)
        assert result.timing_offset == _expected_offset(drop, samples_per_symbol)
        assert result.quality == 1.0
        np.testing.assert_array_equal(result.symbols, symbols[skipped:])


@pytest.mark.parametrize("samples_per_symbol", [2, 3, 4, 5, 8, 10, 16])
def test_every_timing_phase_recovered_qpsk(samples_per_symbol):
    bits = _bits(2 * N_SYMBOLS, 8)
    symbols = qpsk_modulate(bits)
    waveform = qpsk_waveform(bits, samples_per_symbol)
    for drop in range(samples_per_symbol):
        result = recover_symbol_timing(waveform[drop:], samples_per_symbol)
        skipped = _skipped_symbols(drop, samples_per_symbol)
        assert result.timing_offset == _expected_offset(drop, samples_per_symbol)
        np.testing.assert_array_equal(result.symbols, symbols[skipped:])


# ---------------------------------------------------------------------------
# End-to-end recovery through the existing demodulators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [0, 1, 4, 7, 10])
def test_bpsk_bits_recovered_through_existing_demodulator(drop):
    bits = _bits(N_SYMBOLS, 9)
    result = recover_symbol_timing(bpsk_waveform(bits, SPS)[drop:], SPS)
    skipped = _skipped_symbols(drop, SPS)
    recovered = bpsk_demodulate(result.symbols, samples_per_symbol=1)
    np.testing.assert_array_equal(recovered, bits[skipped:])


@pytest.mark.parametrize("drop", [0, 2, 5, 8, 15])
def test_qpsk_bits_recovered_through_existing_demodulator(drop):
    bits = _bits(2 * N_SYMBOLS, 10)
    result = recover_symbol_timing(qpsk_waveform(bits, SPS)[drop:], SPS)
    skipped = _skipped_symbols(drop, SPS)
    recovered = qpsk_demodulate(result.symbols, samples_per_symbol=1)
    np.testing.assert_array_equal(recovered, bits[2 * skipped :])


@pytest.mark.parametrize("samples_per_symbol", [2, 4, 8, 16])
def test_bpsk_bits_recovered_for_several_samples_per_symbol(samples_per_symbol):
    bits = _bits(N_SYMBOLS, 11)
    drop = samples_per_symbol - 1
    waveform = bpsk_waveform(bits, samples_per_symbol)[drop:]
    result = recover_symbol_timing(waveform, samples_per_symbol)
    recovered = bpsk_demodulate(result.symbols, samples_per_symbol=1)
    np.testing.assert_array_equal(recovered, bits[1:])


# ---------------------------------------------------------------------------
# Symbol count, sample indices, and the reported offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [0, 1, 3, 7, 8, 12])
def test_symbol_count_follows_the_documented_formula(drop):
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 12), SPS)[drop:]
    result = recover_symbol_timing(waveform, SPS)
    expected = (waveform.size - result.timing_offset) // SPS
    assert result.symbol_count == expected
    assert result.symbols.shape == (expected,)


@pytest.mark.parametrize("extra", [0, 1, 2, 5, 7])
def test_trailing_partial_symbol_is_never_returned(extra):
    bits = _bits(N_SYMBOLS, 13)
    waveform = np.concatenate(
        [bpsk_waveform(bits, SPS), np.zeros(extra, dtype=np.complex128)]
    )
    result = recover_symbol_timing(waveform, SPS)
    assert result.timing_offset == 0
    assert result.symbol_count == N_SYMBOLS
    np.testing.assert_array_equal(result.symbols, bpsk_modulate(bits))


@pytest.mark.parametrize("samples_per_symbol", ALL_SPS)
def test_scored_window_count_never_exceeds_returned_symbol_count(
    samples_per_symbol,
):
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 14), samples_per_symbol)
    result = recover_symbol_timing(waveform, samples_per_symbol)
    assert result.scored_symbol_count <= result.symbol_count
    expected = (waveform.size - (samples_per_symbol - 1)) // samples_per_symbol
    assert result.scored_symbol_count == expected


@pytest.mark.parametrize("drop", [0, 2, 6, 9])
def test_returned_symbols_are_the_input_samples_at_the_midpoints(drop):
    waveform = qpsk_waveform(_bits(2 * N_SYMBOLS, 15), SPS)[drop:]
    result = recover_symbol_timing(waveform, SPS)
    indices = (
        result.timing_offset
        + np.arange(result.symbol_count) * SPS
        + SPS // 2
    )
    np.testing.assert_array_equal(result.symbols, waveform[indices])
    assert result.first_symbol_index == indices[0]


@pytest.mark.parametrize("samples_per_symbol", ALL_SPS)
def test_selected_offset_stays_inside_one_symbol_period(samples_per_symbol):
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 16), samples_per_symbol)[3:]
    result = recover_symbol_timing(waveform, samples_per_symbol)
    assert 0 <= result.timing_offset < samples_per_symbol
    assert len(result.phase_qualities) == samples_per_symbol
    assert all(0 <= offset < samples_per_symbol for offset in result.tied_offsets)


# ---------------------------------------------------------------------------
# Determinism and input preservation
# ---------------------------------------------------------------------------


def test_repeated_calls_are_deterministic():
    waveform = qpsk_waveform(_bits(2 * N_SYMBOLS, 17), SPS)[3:]
    first = recover_symbol_timing(waveform, SPS)
    second = recover_symbol_timing(waveform, SPS)
    np.testing.assert_array_equal(first.symbols, second.symbols)
    assert first.timing_offset == second.timing_offset
    assert first.first_symbol_index == second.first_symbol_index
    assert first.symbol_count == second.symbol_count
    assert first.scored_symbol_count == second.scored_symbol_count
    assert first.quality == second.quality
    assert first.margin == second.margin
    assert first.phase_qualities == second.phase_qualities
    assert first.tied_offsets == second.tied_offsets


def test_degenerate_tie_break_is_deterministic():
    block = np.zeros(5 * SPS, dtype=np.complex128)
    first = recover_symbol_timing(block, SPS)
    second = recover_symbol_timing(block, SPS)
    assert first.timing_offset == second.timing_offset == 0
    assert first.tied_offsets == second.tied_offsets


def test_input_array_is_not_mutated():
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 18), SPS)[5:]
    before = waveform.copy()
    recover_symbol_timing(waveform, SPS)
    np.testing.assert_array_equal(waveform, before)
    assert waveform.dtype == before.dtype


def test_read_only_input_is_accepted():
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 19), SPS)[5:].copy()
    waveform.setflags(write=False)
    result = recover_symbol_timing(waveform, SPS)
    assert result.timing_offset == _expected_offset(5, SPS)


def test_returned_symbols_do_not_share_memory_with_the_input():
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 20), SPS)
    result = recover_symbol_timing(waveform, SPS)
    assert not np.shares_memory(result.symbols, waveform)
    result.symbols[0] = 12.5 + 0.5j
    assert waveform[SPS // 2] != 12.5 + 0.5j


# ---------------------------------------------------------------------------
# Invalid samples-per-symbol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("samples_per_symbol", [0, -1, -8])
def test_non_positive_samples_per_symbol_raises(samples_per_symbol):
    block = np.ones(64, dtype=np.complex128)
    with pytest.raises(ValueError, match="samples_per_symbol must be >= 1"):
        recover_symbol_timing(block, samples_per_symbol)


@pytest.mark.parametrize(
    "samples_per_symbol",
    [8.0, 8.5, "8", None, True, False, [8], np.float64(8.0), 2 + 0j],
)
def test_non_integer_samples_per_symbol_raises(samples_per_symbol):
    block = np.ones(64, dtype=np.complex128)
    with pytest.raises(ValueError, match="samples_per_symbol must be an integer"):
        recover_symbol_timing(block, samples_per_symbol)


@pytest.mark.parametrize("samples_per_symbol", [np.int64(8), np.int32(8), 8])
def test_integer_like_samples_per_symbol_is_accepted(samples_per_symbol):
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 21), 8)
    result = recover_symbol_timing(waveform, samples_per_symbol)
    assert result.samples_per_symbol == 8
    assert isinstance(result.samples_per_symbol, int)


def test_samples_per_symbol_of_one_leaves_a_single_trivial_phase():
    symbols = bpsk_modulate(_bits(N_SYMBOLS, 22))
    result = recover_symbol_timing(symbols, 1)
    assert result.timing_offset == 0
    assert result.first_symbol_index == 0
    assert result.phase_qualities == (1.0,)
    assert result.tied_offsets == (0,)
    assert result.quality == 1.0
    assert result.margin == 1.0
    assert result.symbol_count == N_SYMBOLS
    np.testing.assert_array_equal(result.symbols, symbols)


# ---------------------------------------------------------------------------
# Invalid sample blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        np.array([], dtype=np.complex128),
        np.array([], dtype=np.float64),
        [],
    ],
)
def test_empty_input_raises(block):
    with pytest.raises(ValueError, match="at least one value"):
        recover_symbol_timing(block, SPS)


@pytest.mark.parametrize(
    "block",
    [
        np.ones((8, 8), dtype=np.complex128),
        np.ones((64, 1), dtype=np.complex128),
        np.ones((2, 4, 8), dtype=np.complex128),
        np.complex128(1 + 1j),
    ],
)
def test_non_one_dimensional_input_raises(block):
    with pytest.raises(ValueError, match="must be one-dimensional"):
        recover_symbol_timing(block, SPS)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(bad):
    block = bpsk_waveform(_bits(N_SYMBOLS, 23), SPS).copy()
    block[17] = bad
    with pytest.raises(ValueError, match="only finite values"):
        recover_symbol_timing(block, SPS)


@pytest.mark.parametrize("samples_per_symbol", [1, 2, 3, 4, 8, 16])
def test_insufficient_samples_raises(samples_per_symbol):
    shortest = 5 * samples_per_symbol - 1
    block = np.ones(shortest - 1, dtype=np.complex128)
    with pytest.raises(ValueError, match="samples must contain at least"):
        recover_symbol_timing(block, samples_per_symbol)


@pytest.mark.parametrize("samples_per_symbol", [1, 2, 3, 4, 8, 16])
def test_shortest_accepted_block_scores_four_windows(samples_per_symbol):
    shortest = 5 * samples_per_symbol - 1
    block = np.ones(shortest, dtype=np.complex128)
    result = recover_symbol_timing(block, samples_per_symbol)
    assert result.scored_symbol_count == 4


def test_insufficient_samples_message_names_the_requirement():
    with pytest.raises(ValueError, match="at least 39 values"):
        recover_symbol_timing(np.ones(38, dtype=np.complex128), 8)


@pytest.mark.parametrize(
    "tie_tolerance", [1.0, 1.5, -0.1, np.nan, np.inf, -np.inf]
)
def test_tie_tolerance_out_of_range_raises(tie_tolerance):
    block = bpsk_waveform(_bits(N_SYMBOLS, 24), SPS)
    with pytest.raises(ValueError, match=r"tie_tolerance must lie within"):
        recover_symbol_timing(block, SPS, tie_tolerance=tie_tolerance)


@pytest.mark.parametrize("tie_tolerance", [None, "wide", [0.1], 1j])
def test_non_real_tie_tolerance_raises(tie_tolerance):
    block = bpsk_waveform(_bits(N_SYMBOLS, 25), SPS)
    with pytest.raises(ValueError, match="tie_tolerance must be a real number"):
        recover_symbol_timing(block, SPS, tie_tolerance=tie_tolerance)


# ---------------------------------------------------------------------------
# Constant and degenerate blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        np.zeros(5 * SPS, dtype=np.complex128),
        np.zeros(5 * SPS, dtype=np.float64),
        np.full(5 * SPS, 2.5 + 1.5j),
        np.full(5 * SPS, -3.0),
    ],
)
def test_constant_block_reports_every_phase_as_tied(block):
    result = recover_symbol_timing(block, SPS)
    assert result.timing_offset == 0
    assert result.quality == 1.0
    assert result.margin == 0.0
    assert result.tied_offsets == tuple(range(SPS))
    assert result.phase_qualities == (1.0,) * SPS


def test_constant_block_still_returns_symbols_and_does_not_raise():
    block = np.full(5 * SPS, 0.75 - 0.25j)
    result = recover_symbol_timing(block, SPS)
    assert result.symbol_count == 5
    np.testing.assert_array_equal(result.symbols, np.full(5, 0.75 - 0.25j))


def test_stream_repeating_one_symbol_is_reported_as_ambiguous():
    waveform = symbols_to_samples(np.full(N_SYMBOLS, 1 + 0j), SPS)
    result = recover_symbol_timing(waveform, SPS)
    assert result.tied_offsets == tuple(range(SPS))
    assert result.margin == 0.0
    np.testing.assert_array_equal(result.symbols, np.full(N_SYMBOLS, 1 + 0j))


# ---------------------------------------------------------------------------
# Identifiability of the timing offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [0, 2, 5, 7])
def test_a_single_observed_transition_pins_the_phase(drop):
    symbols = np.array([1, 1, 1, -1, -1, -1, -1], dtype=np.complex128)
    waveform = symbols_to_samples(symbols, SPS)[drop:]
    result = recover_symbol_timing(waveform, SPS)
    assert result.timing_offset == _expected_offset(drop, SPS)
    assert result.quality == 1.0
    assert result.tied_offsets == (result.timing_offset,)
    assert result.margin > 0.0


def test_transition_outside_the_scored_region_is_reported_as_a_tie():
    # The lone transition sits in the leading samples_per_symbol - 1
    # samples, which no candidate window past phase 0 can see.
    block = np.full(5 * SPS, 1 + 0j)
    block[0] = -1 + 0j
    result = recover_symbol_timing(block, SPS)
    assert result.tied_offsets == tuple(range(1, SPS))
    assert result.margin == 0.0
    assert result.phase_qualities[0] < 1.0


def test_unique_offset_and_positive_margin_agree():
    waveform = qpsk_waveform(_bits(2 * N_SYMBOLS, 26), SPS)[6:]
    result = recover_symbol_timing(waveform, SPS)
    assert (len(result.tied_offsets) == 1) is (result.margin > 1e-12)


def test_wider_tie_tolerance_reports_near_ties_without_moving_the_choice():
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 27), SPS)[4:]
    strict = recover_symbol_timing(waveform, SPS)
    loose = recover_symbol_timing(waveform, SPS, tie_tolerance=0.99)
    assert strict.tied_offsets == (strict.timing_offset,)
    assert loose.tied_offsets == tuple(range(SPS))
    assert loose.timing_offset == strict.timing_offset
    assert loose.quality == strict.quality
    assert loose.margin == strict.margin


def test_zero_tie_tolerance_reports_only_exact_ties():
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 28), SPS)[2:]
    result = recover_symbol_timing(waveform, SPS, tie_tolerance=0.0)
    assert result.tied_offsets == (result.timing_offset,)
    constant = recover_symbol_timing(
        np.full(5 * SPS, 1 + 0j), SPS, tie_tolerance=0.0
    )
    assert constant.tied_offsets == tuple(range(SPS))


# ---------------------------------------------------------------------------
# Fractional timing offsets
# ---------------------------------------------------------------------------


def _fractional_delay_block(bits, samples_per_symbol, delay, oversample=8):
    """Rectangular waveform sampled on a grid shifted by ``delay`` samples.

    The same symbols are generated on an ``oversample`` times finer grid
    and every ``oversample``-th sample is taken, so each returned sample
    is still exactly one symbol value: an ideally sampled rectangular
    pulse whose symbol boundaries now fall between samples. No
    interpolation is involved.
    """
    fine = bpsk_waveform(bits, samples_per_symbol * oversample)
    start = int(round(delay * oversample))
    return fine[start::oversample]


def _boxcar_delay_block(bits, samples_per_symbol, delay, oversample=8):
    """As :func:`_fractional_delay_block`, but each sample is an average.

    Every returned sample averages one whole sample interval of the fine
    grid, so the sample covering a symbol boundary is a blend of the two
    symbols. This is outside the rectangular contract and is used only to
    observe what the search does with a smeared transition.
    """
    fine = bpsk_waveform(bits, samples_per_symbol * oversample)
    start = int(round(delay * oversample))
    usable = (fine.size - start) // oversample
    block = fine[start : start + usable * oversample]
    return block.reshape(usable, oversample).mean(axis=1)


@pytest.mark.parametrize("delay", [0.5, 1.25, 2.5, 3.75, 6.5, 7.75])
def test_fractional_offset_lands_on_the_next_integer_phase(delay):
    bits = _bits(N_SYMBOLS, 29)
    symbols = bpsk_modulate(bits)
    block = _fractional_delay_block(bits, SPS, delay)
    result = recover_symbol_timing(block, SPS)
    expected_offset = math.ceil((-delay) % SPS) % SPS
    first = int((delay + expected_offset) // SPS)
    assert result.timing_offset == expected_offset
    assert result.quality == 1.0
    np.testing.assert_array_equal(
        result.symbols, symbols[first : first + result.symbol_count]
    )


@pytest.mark.parametrize("delay", [2.5, 3.5, 5.25])
def test_smeared_fractional_transition_lands_on_a_straddling_phase(delay):
    bits = _bits(N_SYMBOLS, 30)
    symbols = bpsk_modulate(bits)
    block = _boxcar_delay_block(bits, SPS, delay)
    result = recover_symbol_timing(block, SPS)
    boundary = (-delay) % SPS
    straddling = {math.floor(boundary), math.ceil(boundary) % SPS}
    assert result.timing_offset in straddling
    # The interior midpoint stays clear of the smeared sample, so the
    # recovered symbols are still exact constellation points.
    np.testing.assert_allclose(
        result.symbols, symbols[1 : 1 + result.symbol_count]
    )
    recovered = bpsk_demodulate(result.symbols, samples_per_symbol=1)
    np.testing.assert_array_equal(recovered, bits[1 : 1 + result.symbol_count])


# ---------------------------------------------------------------------------
# Robustness within the controlled contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [30.0, 20.0, 10.0])
def test_noisy_block_still_recovers_the_phase(snr_db):
    bits = _bits(N_SYMBOLS, 31)
    waveform = bpsk_waveform(bits, SPS)[3:]
    noisy = add_awgn(waveform, snr_db, np.random.default_rng(4))
    result = recover_symbol_timing(noisy, SPS)
    assert result.timing_offset == _expected_offset(3, SPS)
    assert result.quality < 1.0
    assert result.margin > 0.0
    recovered = bpsk_demodulate(result.symbols, samples_per_symbol=1)
    np.testing.assert_array_equal(recovered, bits[1:])


@pytest.mark.parametrize("scale", [0.01, 0.5, 3.0, 1000.0])
def test_amplitude_scaling_does_not_change_the_search(scale):
    waveform = qpsk_waveform(_bits(2 * N_SYMBOLS, 32), SPS)[5:]
    plain = recover_symbol_timing(waveform, SPS)
    scaled = recover_symbol_timing(waveform * scale, SPS)
    assert scaled.timing_offset == plain.timing_offset
    assert scaled.phase_qualities == pytest.approx(plain.phase_qualities)
    np.testing.assert_allclose(scaled.symbols, plain.symbols * scale)


@pytest.mark.parametrize("phase_rad", [0.3, 1.0, -2.0, math.pi])
def test_constant_carrier_phase_rotation_does_not_change_the_search(phase_rad):
    waveform = qpsk_waveform(_bits(2 * N_SYMBOLS, 33), SPS)[6:]
    plain = recover_symbol_timing(waveform, SPS)
    rotated = recover_symbol_timing(apply_phase_offset(waveform, phase_rad), SPS)
    assert rotated.timing_offset == plain.timing_offset
    assert rotated.phase_qualities == pytest.approx(plain.phase_qualities)


@pytest.mark.parametrize("samples_per_symbol", [4, 8, 10, 16])
def test_worst_phase_sits_half_a_symbol_from_the_best(samples_per_symbol):
    # A window split evenly across a symbol boundary is the least
    # piecewise-constant one, which is what the criterion should punish
    # most.
    bits = _bits(4 * N_SYMBOLS, 34)
    waveform = bpsk_waveform(bits, samples_per_symbol)[3:]
    result = recover_symbol_timing(waveform, samples_per_symbol)
    qualities = np.asarray(result.phase_qualities)
    worst = int(np.argmin(qualities))
    half = (result.timing_offset + samples_per_symbol // 2) % samples_per_symbol
    assert worst == half


def test_selected_phase_is_the_highest_scoring_one():
    waveform = qpsk_waveform(_bits(2 * N_SYMBOLS, 35), SPS)[7:]
    result = recover_symbol_timing(waveform, SPS)
    qualities = np.asarray(result.phase_qualities)
    assert result.quality == qualities[result.timing_offset]
    assert result.quality == qualities.max()
    assert np.all(qualities <= 1.0) and np.all(qualities >= 0.0)
    others = np.delete(qualities, result.timing_offset)
    assert result.margin == pytest.approx(result.quality - others.max())


# ---------------------------------------------------------------------------
# Result object and dtypes
# ---------------------------------------------------------------------------


def test_result_exposes_exactly_the_documented_fields():
    result = recover_symbol_timing(bpsk_waveform(_bits(N_SYMBOLS, 36), SPS), SPS)
    assert isinstance(result, SymbolTimingRecovery)
    assert set(vars(result)) == {
        "symbols",
        "timing_offset",
        "first_symbol_index",
        "samples_per_symbol",
        "symbol_count",
        "scored_symbol_count",
        "quality",
        "margin",
        "phase_qualities",
        "tied_offsets",
    }


def test_result_is_frozen():
    result = recover_symbol_timing(bpsk_waveform(_bits(N_SYMBOLS, 37), SPS), SPS)
    with pytest.raises(FrozenInstanceError):
        result.timing_offset = 3


def test_result_field_types():
    result = recover_symbol_timing(qpsk_waveform(_bits(2 * N_SYMBOLS, 38), SPS), SPS)
    assert isinstance(result.symbols, np.ndarray)
    assert isinstance(result.timing_offset, int)
    assert isinstance(result.first_symbol_index, int)
    assert isinstance(result.samples_per_symbol, int)
    assert isinstance(result.symbol_count, int)
    assert isinstance(result.scored_symbol_count, int)
    assert isinstance(result.quality, float)
    assert isinstance(result.margin, float)
    assert isinstance(result.phase_qualities, tuple)
    assert all(isinstance(value, float) for value in result.phase_qualities)
    assert isinstance(result.tied_offsets, tuple)
    assert all(isinstance(value, int) for value in result.tied_offsets)


def test_complex_input_keeps_complex_symbols():
    result = recover_symbol_timing(qpsk_waveform(_bits(2 * N_SYMBOLS, 39), SPS), SPS)
    assert result.symbols.dtype == np.complex128
    assert np.iscomplexobj(result.symbols)


def test_real_input_keeps_real_symbols():
    symbols = np.where(_bits(N_SYMBOLS, 40) == 0, 1.0, -1.0)
    waveform = symbols_to_samples(symbols, SPS)[3:]
    result = recover_symbol_timing(waveform, SPS)
    assert result.symbols.dtype == np.float64
    assert result.timing_offset == _expected_offset(3, SPS)
    np.testing.assert_array_equal(result.symbols, symbols[1:])


def test_sequence_input_is_accepted():
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 41), SPS)
    result = recover_symbol_timing(list(waveform), SPS)
    assert result.timing_offset == 0
    np.testing.assert_array_equal(result.symbols, bpsk_modulate(_bits(N_SYMBOLS, 41)))


# ---------------------------------------------------------------------------
# Agreement with the neighbouring estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [0, 3, 5, 7])
def test_offset_matches_the_symbol_rate_estimator_boundary(drop):
    waveform = bpsk_waveform(_bits(N_SYMBOLS, 42), SPS)[drop:]
    recovered = recover_symbol_timing(waveform, SPS)
    estimated = estimate_symbol_rate(waveform, FS)
    assert estimated.samples_per_symbol == SPS
    assert recovered.timing_offset == estimated.boundary_offset


@pytest.mark.parametrize("drop", [0, 2, 6])
def test_recovered_offset_feeds_the_modulation_estimator(drop):
    bits = _bits(4 * N_SYMBOLS, 43)
    for waveform, expected in (
        (bpsk_waveform(bits, SPS)[drop:], "BPSK"),
        (qpsk_waveform(bits, SPS)[drop:], "QPSK"),
    ):
        recovered = recover_symbol_timing(waveform, SPS)
        estimate = estimate_modulation(
            waveform, SPS, boundary_offset=recovered.timing_offset
        )
        assert estimate.modulation == expected

