"""Unit tests for iqwav.estimation.modulation.estimate_modulation.

These cover the bounded, controlled-signal BPSK-vs-QPSK primitive only:
rectangular-pulse waveforms built by :mod:`iqwav.modulation` with a known
integer ``samples_per_symbol`` and a known symbol-boundary phase, so the
ground-truth constellation is exact by construction. No blind modulation
recognition, timing recovery, carrier recovery, or symbol-rate estimation is
involved.

The QPSK reference constellation is derived here from
:func:`iqwav.modulation.qpsk_modulate` rather than assumed, so the adversarial
subset cases below stay correct if that mapping ever changes.

Two conventions run through the whole file:

- ``estimate_modulation`` reports three outcomes, ``"BPSK"``, ``"QPSK"`` and
  ``"AMBIGUOUS"``. A :class:`ValueError` means a violated contract (bad shape,
  bad argument, too few symbols), never an inconvenient measurement, so a
  degenerate block is asserted to come back ``"AMBIGUOUS"`` rather than to
  raise.
- ``confidence`` is a classification margin, ``abs(bpsk_score - qpsk_score)``.
  It is never treated here as a probability or an error rate, and the
  adversarial cases assert only that a genuinely ambiguous observation gets a
  *small* margin, never that a margin implies correctness.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import ModulationEstimate, estimate_modulation
from iqwav.modulation import (
    bpsk_waveform,
    generate_iq_tone,
    qpsk_modulate,
    qpsk_waveform,
    symbols_to_samples,
)

FS = 48000.0
SPS = 8
N_SYMBOLS = 400
MIN_SYMBOLS = 16  # documented floor of estimate_modulation


def _bits(count: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-random bits."""
    return np.random.default_rng(seed).integers(0, 2, count)


def _bpsk(sps: int = SPS, *, n_symbols: int = N_SYMBOLS, seed: int = 1):
    return bpsk_waveform(_bits(n_symbols, seed), sps)


def _qpsk(sps: int = SPS, *, n_symbols: int = N_SYMBOLS, seed: int = 2):
    return qpsk_waveform(_bits(2 * n_symbols, seed), sps)


def _from_symbols(symbols: np.ndarray, sps: int = SPS) -> np.ndarray:
    return symbols_to_samples(np.asarray(symbols, dtype=np.complex128), sps)


def _qpsk_points() -> np.ndarray:
    """The four QPSK constellation points, read off the generator itself.

    ``qpsk_modulate`` maps the bit pair ``(b0, b1)`` to index ``2 * b0 + b1``,
    so feeding it ``00 01 10 11`` returns the constellation in index order.
    """
    return qpsk_modulate(np.array([0, 0, 0, 1, 1, 0, 1, 1]))


def _antipodal_pairs(points: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs of ``points`` that are 180 degrees apart."""
    return [
        (i, j)
        for i in range(points.size)
        for j in range(i + 1, points.size)
        if abs(points[i] + points[j]) < 1e-12
    ]


def _adjacent_pairs(points: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs of ``points`` that are 90 degrees apart."""
    antipodal = set(_antipodal_pairs(points))
    return [
        (i, j)
        for i in range(points.size)
        for j in range(i + 1, points.size)
        if (i, j) not in antipodal
    ]


def _two_point_stream(
    points: np.ndarray, pair: tuple[int, int], minority: int, total: int = 200
):
    """A ``total``-symbol two-point stream, ``minority`` on the second point."""
    first, second = pair
    return _from_symbols(
        np.concatenate(
            [
                np.full(total - minority, points[first]),
                np.full(minority, points[second]),
            ]
        )
    )


def _skewed_qpsk_stream(points: np.ndarray, majority: float, total: int = 400):
    """All four points, with ``majority`` of the probability on the first."""
    share = (1.0 - majority) / 3.0
    counts = [majority] + [share] * 3
    return _from_symbols(
        np.concatenate(
            [
                np.full(int(round(weight * total)), points[index])
                for index, weight in enumerate(counts)
            ]
        )
    )


# --------------------------------------------------------------------------
# Clean BPSK / QPSK across symbol periods
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sps", [1, 2, 3, 4, 5, 8, 10, 16, 32, 64])
def test_clean_bpsk_is_identified_at_every_symbol_period(sps):
    result = estimate_modulation(_bpsk(sps, seed=10 + sps), sps)
    assert result.modulation == "BPSK"
    # The margin is gated by the realized symbol balance of the draw, so a
    # random 400-symbol stream lands near but not always exactly at 1.0.
    assert result.confidence > 0.85
    assert result.bpsk_score > 0.85
    assert result.qpsk_score == pytest.approx(0.0)
    assert result.samples_per_symbol == sps
    assert result.symbol_count == N_SYMBOLS


def test_a_perfectly_balanced_bpsk_stream_scores_exactly_one():
    """Alternating symbols use both points equally, the ideal BPSK evidence."""
    result = estimate_modulation(bpsk_waveform(np.arange(N_SYMBOLS) % 2, SPS), SPS)
    assert result.modulation == "BPSK"
    assert result.bpsk_score == pytest.approx(1.0)
    assert result.qpsk_score == pytest.approx(0.0)
    assert result.confidence == pytest.approx(1.0)


@pytest.mark.parametrize("sps", [1, 2, 3, 4, 5, 8, 10, 16, 32, 64])
def test_clean_qpsk_is_identified_at_every_symbol_period(sps):
    result = estimate_modulation(_qpsk(sps, seed=100 + sps), sps)
    assert result.modulation == "QPSK"
    assert result.qpsk_score > result.bpsk_score
    # A random draw is never exactly balanced across the two antipodal
    # pairs, so the 2-fold score is small but not always exactly zero.
    assert result.bpsk_score < 0.1
    assert result.confidence > 0.8
    assert result.samples_per_symbol == sps
    assert result.symbol_count == N_SYMBOLS


def test_balanced_qpsk_cycle_is_perfectly_identified():
    """An exactly balanced four-point cycle gives the ideal QPSK scores."""
    result = estimate_modulation(_from_symbols(np.resize(_qpsk_points(), 400)), SPS)
    assert result.modulation == "QPSK"
    assert result.bpsk_score == pytest.approx(0.0)
    assert result.qpsk_score == pytest.approx(1.0)
    assert result.confidence == pytest.approx(1.0)


def test_real_valued_antipodal_stream_reads_as_bpsk():
    symbols = np.where(_bits(200, 5) == 0, 1.0, -1.0)
    result = estimate_modulation(symbols_to_samples(symbols, SPS), SPS)
    assert result.modulation == "BPSK"
    assert result.confidence == pytest.approx(1.0)


def test_scores_are_scale_invariant():
    """Amplitude carries no constellation information, at any magnitude."""
    reference = estimate_modulation(_qpsk(seed=55), SPS)
    for scale in (1e-300, 1e-9, 1e9, 1e300):
        scaled = estimate_modulation(scale * _qpsk(seed=55), SPS)
        assert scaled.modulation == reference.modulation
        assert scaled.bpsk_score == pytest.approx(reference.bpsk_score)
        assert scaled.qpsk_score == pytest.approx(reference.qpsk_score)


def test_symbols_of_zero_magnitude_do_not_break_the_statistic():
    """A gated stream keeps its constellation; blanked symbols just add none."""
    symbols = qpsk_modulate(_bits(400, 14))
    symbols[::3] = 0.0
    result = estimate_modulation(_from_symbols(symbols), SPS)
    assert result.modulation == "QPSK"
    assert result.confidence > 0.99


# --------------------------------------------------------------------------
# Known symbol-boundary phase
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trim", [0, 1, 2, 3, 4, 5, 6, 7])
def test_declared_boundary_offset_recovers_a_trimmed_block(trim):
    """The Phase 2E boundary-offset convention, applied to both classes."""
    offset = (-trim) % SPS
    bpsk = estimate_modulation(
        _bpsk(n_symbols=300, seed=1)[trim:], SPS, boundary_offset=offset
    )
    qpsk = estimate_modulation(
        _qpsk(n_symbols=300, seed=2)[trim:], SPS, boundary_offset=offset
    )
    assert bpsk.modulation == "BPSK"
    assert bpsk.confidence == pytest.approx(1.0)
    assert qpsk.modulation == "QPSK"
    assert qpsk.confidence == pytest.approx(1.0)
    assert bpsk.symbol_count == (300 if trim == 0 else 299)


def test_boundary_offset_shifts_the_symbol_count():
    samples = _bpsk(n_symbols=100, seed=56)
    assert estimate_modulation(samples, SPS).symbol_count == 100
    assert estimate_modulation(samples, SPS, boundary_offset=1).symbol_count == 99
    assert estimate_modulation(samples, SPS, boundary_offset=7).symbol_count == 99


def test_interior_sampling_survives_a_boundary_phase_off_by_one():
    """The interior midpoint is chosen so a one-sample error stays in-symbol."""
    samples = _qpsk(seed=57)
    for declared in (SPS - 1, 1):
        result = estimate_modulation(samples, SPS, boundary_offset=declared)
        assert result.modulation == "QPSK"
        assert result.confidence > 0.8


# --------------------------------------------------------------------------
# AWGN
# --------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0])
@pytest.mark.parametrize("sps", [4, 8, 16])
def test_noisy_bpsk_is_still_identified(snr_db, sps):
    noisy = add_awgn(_bpsk(sps, seed=7), snr_db=snr_db, rng=np.random.default_rng(11))
    result = estimate_modulation(noisy, sps)
    assert result.modulation == "BPSK"
    assert result.confidence > 0.5


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0])
@pytest.mark.parametrize("sps", [4, 8, 16])
def test_noisy_qpsk_is_still_identified(snr_db, sps):
    noisy = add_awgn(_qpsk(sps, seed=8), snr_db=snr_db, rng=np.random.default_rng(12))
    result = estimate_modulation(noisy, sps)
    assert result.modulation == "QPSK"
    assert result.confidence > 0.3


@pytest.mark.parametrize("seed", range(12))
def test_noise_at_five_db_does_not_flip_either_class(seed):
    """Independent draws at the documented 5 dB floor, both classes."""
    bpsk = add_awgn(
        _bpsk(seed=seed), snr_db=5.0, rng=np.random.default_rng(seed + 900)
    )
    qpsk = add_awgn(
        _qpsk(seed=seed), snr_db=5.0, rng=np.random.default_rng(seed + 950)
    )
    assert estimate_modulation(bpsk, SPS).modulation == "BPSK"
    assert estimate_modulation(qpsk, SPS).modulation == "QPSK"


@pytest.mark.parametrize("kind", ["BPSK", "QPSK"])
def test_confidence_decreases_as_noise_increases(kind):
    clean = _bpsk(seed=9) if kind == "BPSK" else _qpsk(seed=9)
    confidences = [
        estimate_modulation(
            add_awgn(clean, snr_db=snr, rng=np.random.default_rng(13)), SPS
        ).confidence
        for snr in (30.0, 20.0, 10.0, 5.0)
    ]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 < value < 1.0 for value in confidences)


# --------------------------------------------------------------------------
# Constant phase rotation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "degrees", [0.0, 13.0, 30.0, 45.0, 90.0, 135.0, 137.0, 180.0, 222.5]
)
def test_constant_phase_rotation_does_not_change_the_decision(degrees):
    radians = np.deg2rad(degrees)
    bpsk = estimate_modulation(apply_phase_offset(_bpsk(seed=60), radians), SPS)
    qpsk = estimate_modulation(apply_phase_offset(_qpsk(seed=61), radians), SPS)
    assert bpsk.modulation == "BPSK"
    assert bpsk.confidence == pytest.approx(1.0)
    assert qpsk.modulation == "QPSK"
    assert qpsk.confidence == pytest.approx(1.0)


@pytest.mark.parametrize("degrees", [13.0, 45.0, 90.0, 180.0, 231.7])
@pytest.mark.parametrize("kind", ["BPSK", "QPSK"])
def test_rotation_invariance_is_exact_even_under_noise(kind, degrees):
    """Rotating a block must leave both scores numerically unchanged."""
    clean = _bpsk(seed=1) if kind == "BPSK" else _qpsk(seed=2)
    noisy = add_awgn(clean, snr_db=10.0, rng=np.random.default_rng(6))
    reference = estimate_modulation(noisy, SPS)
    rotated = estimate_modulation(apply_phase_offset(noisy, np.deg2rad(degrees)), SPS)
    assert rotated.modulation == reference.modulation
    assert rotated.bpsk_score == pytest.approx(reference.bpsk_score, abs=1e-12)
    assert rotated.qpsk_score == pytest.approx(reference.qpsk_score, abs=1e-12)


@pytest.mark.parametrize("degrees", [0.0, 22.5, 45.0, 67.5])
def test_rotation_combined_with_five_db_noise(degrees):
    radians = np.deg2rad(degrees)
    bpsk = add_awgn(
        apply_phase_offset(_bpsk(seed=62), radians),
        snr_db=5.0,
        rng=np.random.default_rng(21),
    )
    qpsk = add_awgn(
        apply_phase_offset(_qpsk(seed=63), radians),
        snr_db=5.0,
        rng=np.random.default_rng(22),
    )
    assert estimate_modulation(bpsk, SPS).modulation == "BPSK"
    assert estimate_modulation(qpsk, SPS).modulation == "QPSK"


@pytest.mark.parametrize("degrees", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_rotation_does_not_disturb_the_adversarial_subsets(degrees):
    """The adversarial families keep their verdicts under a rotated carrier.

    A rotation by 45 degrees maps the QPSK constellation onto the BPSK axes
    and vice versa, so this is the case where an orientation-dependent
    statistic would flip. Both coherences are rotation-invariant by
    construction, so the classes are unchanged.
    """
    radians = np.deg2rad(degrees)
    points = _qpsk_points()
    antipodal = _two_point_stream(points, _antipodal_pairs(points)[0], 100)
    adjacent = _two_point_stream(points, _adjacent_pairs(points)[0], 100)
    skewed = _skewed_qpsk_stream(points, 0.7)

    assert estimate_modulation(
        apply_phase_offset(antipodal, radians), SPS
    ).modulation == "BPSK"
    assert estimate_modulation(
        apply_phase_offset(adjacent, radians), SPS
    ).modulation == "QPSK"
    assert estimate_modulation(
        apply_phase_offset(skewed, radians), SPS
    ).modulation == "QPSK"


@pytest.mark.parametrize("degrees", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_rotation_leaves_a_degenerate_block_ambiguous(degrees):
    """No rotation can turn a single-point block into a constellation."""
    points = _qpsk_points()
    single = _from_symbols(np.full(200, points[0]))
    result = estimate_modulation(
        apply_phase_offset(single, np.deg2rad(degrees)), SPS, min_confidence=0.0
    )
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence == 0.0


# --------------------------------------------------------------------------
# Small residual carrier offset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfo_fraction", [0.0, 1e-5, 5e-5, 1e-4])
def test_small_residual_cfo_leaves_both_classes_identified(cfo_fraction):
    """Up to ~15 degrees of total drift across the block is harmless."""
    cfo_hz = cfo_fraction * FS / SPS
    bpsk = estimate_modulation(
        apply_frequency_offset(_bpsk(seed=70), FS, cfo_hz), SPS
    )
    qpsk = estimate_modulation(
        apply_frequency_offset(_qpsk(seed=71), FS, cfo_hz), SPS
    )
    assert bpsk.modulation == "BPSK"
    assert qpsk.modulation == "QPSK"
    assert bpsk.confidence > 0.95
    assert qpsk.confidence > 0.95


def test_growing_residual_cfo_erodes_confidence():
    confidences = [
        estimate_modulation(
            apply_frequency_offset(_qpsk(seed=71), FS, fraction * FS / SPS), SPS
        ).confidence
        for fraction in (0.0, 1e-5, 5e-5, 1e-4, 2e-4)
    ]
    assert confidences == sorted(confidences, reverse=True)


@pytest.mark.parametrize("cfo_fraction", [5e-3, 1e-2])
def test_large_residual_cfo_is_ambiguous_rather_than_guessed(cfo_fraction):
    """A carrier spinning through many cycles destroys the constellation."""
    rotated = apply_frequency_offset(_qpsk(seed=71), FS, cfo_fraction * FS / SPS)
    result = estimate_modulation(rotated, SPS)
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence == pytest.approx(0.0)
    # Even with the floor removed there is nothing left to decide on: the
    # smeared constellation supports neither class at all.
    unrestricted = estimate_modulation(rotated, SPS, min_confidence=0.0)
    assert unrestricted.modulation == "AMBIGUOUS"
    assert unrestricted.bpsk_score == pytest.approx(0.0)
    assert unrestricted.qpsk_score == pytest.approx(0.0)


def test_out_of_contract_cfo_can_name_the_wrong_class_weakly():
    """A documented failure of an out-of-contract input, pinned deliberately.

    The estimator assumes the carrier phase is essentially constant across the
    block. Break that badly enough and a rotating QPSK constellation can alias
    into something the 2-fold coherence prefers: 432 degrees of total drift
    turns this draw into a ``"BPSK"`` reading. The margin stays small -- every
    wrong reading found in a sweep of 11 drifts x 40 seeds was under 0.12, and
    none went the other way -- but it does clear the default floor, which is
    why the floor must not be treated as CFO protection.
    """
    drifting = apply_frequency_offset(_qpsk(seed=26), FS, 3e-3 * FS / SPS)
    result = estimate_modulation(drifting, SPS)
    assert result.modulation == "BPSK"  # wrong, and out of contract
    assert result.confidence == pytest.approx(0.1122, abs=1e-4)
    # A caller who needs protection has to raise the floor, not trust 0.05.
    assert estimate_modulation(drifting, SPS, min_confidence=0.2).modulation == (
        "AMBIGUOUS"
    )


# --------------------------------------------------------------------------
# Short blocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_symbols", [16, 20, 24, 32, 48, 64, 128])
def test_short_blocks_are_accepted_down_to_the_documented_floor(n_symbols):
    bpsk = estimate_modulation(_bpsk(4, n_symbols=n_symbols, seed=300 + n_symbols), 4)
    qpsk = estimate_modulation(_qpsk(4, n_symbols=n_symbols, seed=400 + n_symbols), 4)
    assert bpsk.modulation == "BPSK"
    assert bpsk.symbol_count == n_symbols
    assert qpsk.modulation == "QPSK"
    assert qpsk.symbol_count == n_symbols


@pytest.mark.parametrize("sps", [1, 2, 3, 4, 8])
def test_shortest_accepted_block_at_several_symbol_periods(sps):
    bpsk = estimate_modulation(_bpsk(sps, n_symbols=MIN_SYMBOLS, seed=500 + sps), sps)
    qpsk = estimate_modulation(_qpsk(sps, n_symbols=MIN_SYMBOLS, seed=600 + sps), sps)
    assert bpsk.modulation == "BPSK"
    assert bpsk.symbol_count == MIN_SYMBOLS
    assert qpsk.modulation == "QPSK"
    assert qpsk.symbol_count == MIN_SYMBOLS


def test_short_block_qpsk_draw_is_not_confidently_claimed():
    """A documented limitation, not a claim: 16 symbols may not be QPSK-shaped.

    Seed 45 draws 16 QPSK symbols of which 14 land on the ``(45, -135)``
    antipodal pair, so the *observed* constellation really is close to
    antipodal. Both scores end up middling and nearly equal, so the block is
    reported ``"AMBIGUOUS"`` rather than claimed as either class. The same
    generator with 64 symbols is read correctly, which is why at least 64
    symbols are recommended.
    """
    short = estimate_modulation(_qpsk(4, n_symbols=MIN_SYMBOLS, seed=45), 4)
    assert short.modulation == "AMBIGUOUS"
    assert short.confidence == pytest.approx(0.0184, abs=1e-4)
    assert short.bpsk_score == pytest.approx(0.2881, abs=1e-4)
    assert short.qpsk_score == pytest.approx(0.2697, abs=1e-4)

    longer = estimate_modulation(_qpsk(4, n_symbols=64, seed=45), 4)
    assert longer.modulation == "QPSK"
    assert longer.confidence == pytest.approx(1.0)


def test_a_short_qpsk_draw_read_as_bpsk_really_is_antipodal():
    """The 16-symbol floor is numerical, and its failures are honest ones.

    Over 500 seeds at the minimum block length, 9 QPSK draws read ``"BPSK"``.
    Each one is checked here against its own symbol histogram: every such
    draw puts at least 81 per cent of its symbols on a single antipodal pair,
    so the estimator is describing the constellation it was given, not
    guessing the generator. This is exactly why 16 symbols is documented as a
    numerical floor and 64 as the recommendation.
    """
    points = _qpsk_points()
    pairs = _antipodal_pairs(points)
    readings = {"BPSK": 0, "QPSK": 0, "AMBIGUOUS": 0}
    for seed in range(500):
        bits = _bits(2 * MIN_SYMBOLS, seed)
        result = estimate_modulation(qpsk_waveform(bits, 4), 4)
        readings[result.modulation] += 1
        if result.modulation != "BPSK":
            continue
        indices = np.array(
            [int(np.argmin(np.abs(points - value))) for value in qpsk_modulate(bits)]
        )
        share = max(float(np.isin(indices, pair).mean()) for pair in pairs)
        assert share >= 0.8125, (seed, share, result)
    assert readings == {"BPSK": 9, "QPSK": 487, "AMBIGUOUS": 4}


@pytest.mark.parametrize("seed", range(20))
def test_sixty_four_symbols_is_enough_for_both_classes(seed):
    bpsk = estimate_modulation(_bpsk(4, n_symbols=64, seed=seed), 4)
    qpsk = estimate_modulation(_qpsk(4, n_symbols=64, seed=seed), 4)
    assert bpsk.modulation == "BPSK"
    assert qpsk.modulation == "QPSK"


def test_the_recommended_length_removes_the_short_block_errors():
    """Every draw that the 16-symbol floor gets wrong is right at 64."""
    worst_bpsk = worst_qpsk = 1.0
    for seed in range(500):
        bpsk = estimate_modulation(_bpsk(4, n_symbols=64, seed=seed), 4)
        qpsk = estimate_modulation(_qpsk(4, n_symbols=64, seed=seed), 4)
        assert bpsk.modulation == "BPSK", (seed, bpsk)
        assert qpsk.modulation == "QPSK", (seed, qpsk)
        worst_bpsk = min(worst_bpsk, bpsk.confidence)
        worst_qpsk = min(worst_qpsk, qpsk.confidence)
    # Margins are still only margins: correct at every seed, but not large.
    assert worst_bpsk == pytest.approx(0.5774, abs=1e-4)
    assert worst_qpsk == pytest.approx(0.2208, abs=1e-4)


def test_a_short_two_point_block_clears_the_floor_a_long_one_does_not():
    """The margin is not comparable across block lengths, by construction.

    A single minority symbol is 1/16 of a 16-symbol block but only 1/64 of a
    64-symbol one, and the margin tracks that fraction: the short block is
    named while the longer block with the *same* one-off symbol is withheld.
    This is why ``confidence`` is documented as a margin and not as a
    probability, and it is a second reason the 16-symbol floor is a numerical
    minimum rather than a recommendation.
    """
    points = _qpsk_points()
    antipodal = _antipodal_pairs(points)[0]
    adjacent = _adjacent_pairs(points)[0]

    short_anti = estimate_modulation(
        _two_point_stream(points, antipodal, 1, total=MIN_SYMBOLS), SPS
    )
    short_adj = estimate_modulation(
        _two_point_stream(points, adjacent, 1, total=MIN_SYMBOLS), SPS
    )
    assert short_anti.modulation == "BPSK"
    assert short_anti.confidence == pytest.approx(0.1340, abs=1e-4)
    assert short_adj.modulation == "QPSK"
    assert short_adj.confidence == pytest.approx(0.0780, abs=1e-4)

    long_anti = estimate_modulation(
        _two_point_stream(points, antipodal, 1, total=64), SPS
    )
    long_adj = estimate_modulation(
        _two_point_stream(points, adjacent, 1, total=64), SPS
    )
    assert long_anti.modulation == "AMBIGUOUS"
    assert long_anti.confidence == pytest.approx(0.0318, abs=1e-4)
    assert long_adj.modulation == "AMBIGUOUS"
    assert long_adj.confidence == pytest.approx(0.0165, abs=1e-4)


def test_a_short_block_of_all_four_points_is_still_read_as_qpsk():
    """The floor is usable when the constellation really is exercised."""
    points = _qpsk_points()
    for counts in ([4, 4, 4, 4], [7, 3, 3, 3], [10, 2, 2, 2], [13, 1, 1, 1]):
        assert sum(counts) == MIN_SYMBOLS
        symbols = np.concatenate(
            [np.full(count, points[index]) for index, count in enumerate(counts)]
        )
        result = estimate_modulation(_from_symbols(symbols), SPS)
        assert result.modulation == "QPSK", counts
        assert result.qpsk_score > result.bpsk_score


def test_a_single_point_is_ambiguous_even_at_the_minimum_length():
    points = _qpsk_points()
    result = estimate_modulation(
        _from_symbols(np.full(MIN_SYMBOLS, points[0])), SPS, min_confidence=0.0
    )
    assert result.modulation == "AMBIGUOUS"
    assert result.symbol_count == MIN_SYMBOLS
    assert result.confidence == 0.0


# --------------------------------------------------------------------------
# Unbalanced and adversarial symbol sequences
# --------------------------------------------------------------------------


def test_bpsk_never_flips_class_under_symbol_imbalance():
    """Squaring collapses an antipodal pair however unevenly it is used.

    The class never becomes QPSK, but the *margin* honestly tracks how much
    constellation diversity the block actually shows.
    """
    for every in (2, 5, 10, 20):
        bits = np.where(np.arange(N_SYMBOLS) % every == 0, 1, 0)
        result = estimate_modulation(
            bpsk_waveform(bits, SPS), SPS, min_confidence=0.0
        )
        assert result.modulation == "BPSK"
        assert result.qpsk_score == pytest.approx(0.0)


@pytest.mark.parametrize("minority", [1, 2, 5, 10, 20, 40, 80, 100, 200])
def test_the_bpsk_margin_tracks_twice_the_minority_fraction(minority):
    """The gate has a near-exact reading on a clean antipodal pair: ``2 * f``.

    ``C_1 = 1 - 2 * f`` for a two-point stream with minority fraction ``f``
    (up to the ``O(1 / n)`` debias term), so ``bpsk_score = C_2 * (1 - C_1)``
    is ``2 * f`` to within about 0.002 at 400 symbols. This is what makes the
    default floor a statement about constellation diversity: it withholds a
    decision below about 2.5 per cent minority symbols.
    """
    bits = np.zeros(N_SYMBOLS, dtype=int)
    bits[:minority] = 1
    fraction = minority / N_SYMBOLS
    samples = bpsk_waveform(bits, SPS)
    result = estimate_modulation(samples, SPS, min_confidence=0.0)
    assert result.bpsk_score == pytest.approx(2.0 * fraction, abs=2e-3)
    assert result.qpsk_score == pytest.approx(0.0)
    # At the default floor the same block is named only once that margin
    # clears 0.05, i.e. once the minority point carries about 2.5 per cent.
    expected = "BPSK" if result.bpsk_score >= 0.05 else "AMBIGUOUS"
    assert estimate_modulation(samples, SPS).modulation == expected


def test_a_lone_opposite_bpsk_symbol_is_ambiguous_not_confident_bpsk():
    """One flipped symbol in 400 is not a constellation, so nothing is claimed."""
    bits = np.zeros(N_SYMBOLS, dtype=int)
    bits[0] = 1
    result = estimate_modulation(bpsk_waveform(bits, SPS), SPS)
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence == pytest.approx(0.005, abs=1e-3)


def test_qpsk_skewed_three_to_one_still_reads_as_qpsk_with_a_small_margin():
    """Heavy skew erodes the margin instead of manufacturing a BPSK claim.

    Piling three quarters of the symbols onto one point raises ``C_2`` and
    lowers ``C_4`` together, so the two scores converge rather than crossing.
    """
    points = _qpsk_points()
    skewed = np.concatenate([np.full(150, points[0]), np.resize(points[1:], 50)])
    result = estimate_modulation(_from_symbols(skewed), SPS)
    assert result.modulation == "QPSK"
    assert result.confidence == pytest.approx(0.1237, abs=1e-4)
    assert result.bpsk_score < result.qpsk_score


@pytest.mark.parametrize(
    "majority", [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
)
def test_strongly_imbalanced_qpsk_is_never_confidently_called_bpsk(majority):
    """The adversarial sweep: no majority weight produces a BPSK claim.

    As one point takes over, the block degrades from a confident ``"QPSK"``
    to ``"AMBIGUOUS"``; it never crosses into a confident ``"BPSK"``, because
    the same concentration that raises the 2-fold coherence also raises the
    1-fold coherence that gates it.
    """
    points = _qpsk_points()
    result = estimate_modulation(_skewed_qpsk_stream(points, majority), SPS)
    assert result.modulation in {"QPSK", "AMBIGUOUS"}
    assert result.bpsk_score < result.qpsk_score
    assert result.bpsk_score < 0.3


def test_the_qpsk_margin_falls_monotonically_as_one_point_takes_over():
    points = _qpsk_points()
    margins = [
        estimate_modulation(
            _skewed_qpsk_stream(points, majority), SPS, min_confidence=0.0
        ).confidence
        for majority in (0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.95, 0.99)
    ]
    assert margins == sorted(margins, reverse=True)
    assert margins[0] == pytest.approx(1.0)
    assert margins[-1] < 0.001


@pytest.mark.parametrize("majority", [0.85, 0.90, 0.95, 0.99])
def test_an_overwhelming_qpsk_majority_becomes_ambiguous(majority):
    """Past about a 0.85 majority the observation supports neither class."""
    points = _qpsk_points()
    assert (
        estimate_modulation(_skewed_qpsk_stream(points, majority), SPS).modulation
        == "AMBIGUOUS"
    )


def test_qpsk_mildly_skewed_across_lobes_still_reads_as_qpsk():
    points = _qpsk_points()
    first, second = _antipodal_pairs(points)[0]
    third = next(k for k in range(points.size) if k not in (first, second))
    mild = np.concatenate(
        [
            np.resize(points[[first, second]], 140),
            np.full(60, points[third]),
        ]
    )
    result = estimate_modulation(_from_symbols(mild), SPS)
    assert result.modulation == "QPSK"
    assert result.confidence < 0.5  # honestly weaker than a balanced stream


def test_two_adjacent_qpsk_points_are_reported_as_qpsk():
    """Squaring a 90-degree pair gives an antipodal pair, so C_2 stays 0."""
    points = _qpsk_points()
    adjacent = _adjacent_pairs(points)
    assert len(adjacent) == 4
    for pair in adjacent:
        result = estimate_modulation(_two_point_stream(points, pair, 100), SPS)
        assert result.modulation == "QPSK"
        assert result.confidence == pytest.approx(1.0)
        assert result.bpsk_score == pytest.approx(0.0)


@pytest.mark.parametrize("minority", [10, 20, 50, 100])
def test_a_usable_adjacent_pair_reads_as_qpsk_at_every_skew(minority):
    """Two points 90 degrees apart are not antipodal, whatever the mix."""
    points = _qpsk_points()
    for pair in _adjacent_pairs(points):
        result = estimate_modulation(_two_point_stream(points, pair, minority), SPS)
        assert result.modulation == "QPSK"
        assert result.bpsk_score < result.qpsk_score


@pytest.mark.parametrize("minority", [1, 2, 5])
def test_an_adjacent_pair_becomes_ambiguous_as_one_point_vanishes(minority):
    """A 90-degree pair with a near-absent second point is not a constellation."""
    points = _qpsk_points()
    for pair in _adjacent_pairs(points):
        result = estimate_modulation(_two_point_stream(points, pair, minority), SPS)
        assert result.modulation == "AMBIGUOUS"
        assert result.confidence < 0.05


# --------------------------------------------------------------------------
# QPSK sequences using only a subset of the constellation
# --------------------------------------------------------------------------


def test_two_antipodal_qpsk_points_are_indistinguishable_from_bpsk():
    """A documented, irreducible ambiguity of the observed constellation.

    A QPSK generator restricted to one antipodal pair emits exactly a
    rotated BPSK constellation. The estimator reports BPSK, because that is
    what the symbols are; it cannot recover the generator's intent.
    """
    points = _qpsk_points()
    pairs = _antipodal_pairs(points)
    assert len(pairs) == 2
    for first, second in pairs:
        result = estimate_modulation(
            _from_symbols(np.resize(points[[first, second]], 200)), SPS
        )
        assert result.modulation == "BPSK"
        assert result.confidence == pytest.approx(1.0)


def test_an_antipodal_qpsk_subset_scores_identically_to_real_bpsk():
    """The same numbers as a real BPSK block: there is nothing left to tell apart.

    This is the mathematical statement of the limitation, not a tolerance:
    the restricted QPSK stream and a genuine BPSK stream with the same symbol
    pattern produce bit-identical scores.
    """
    points = _qpsk_points()
    first, second = _antipodal_pairs(points)[0]
    pattern = _bits(200, seed=41)
    restricted = np.where(pattern == 0, points[first], points[second])
    genuine = np.where(pattern == 0, 1.0 + 0.0j, -1.0 + 0.0j)
    from_qpsk = estimate_modulation(_from_symbols(restricted), SPS)
    from_bpsk = estimate_modulation(_from_symbols(genuine), SPS)
    assert from_qpsk.modulation == from_bpsk.modulation == "BPSK"
    assert from_qpsk.bpsk_score == from_bpsk.bpsk_score
    assert from_qpsk.qpsk_score == from_bpsk.qpsk_score
    assert from_qpsk.confidence == from_bpsk.confidence


@pytest.mark.parametrize("minority", [5, 10, 20, 50, 100])
def test_a_usable_antipodal_pair_reads_as_bpsk_at_every_skew(minority):
    """The ambiguity is stable: skew changes the margin, never the class."""
    points = _qpsk_points()
    for pair in _antipodal_pairs(points):
        result = estimate_modulation(_two_point_stream(points, pair, minority), SPS)
        assert result.modulation == "BPSK"
        assert result.qpsk_score == pytest.approx(0.0)
        assert result.confidence == pytest.approx(2.0 * minority / 200, abs=4e-3)


@pytest.mark.parametrize("minority", [1, 2])
def test_an_antipodal_pair_becomes_ambiguous_as_one_point_vanishes(minority):
    """Two points, but one of them almost never used: not enough diversity."""
    points = _qpsk_points()
    for pair in _antipodal_pairs(points):
        result = estimate_modulation(_two_point_stream(points, pair, minority), SPS)
        assert result.modulation == "AMBIGUOUS"
        assert result.confidence < 0.05


def test_three_qpsk_points_are_reported_as_qpsk_with_lower_confidence():
    points = _qpsk_points()
    result = estimate_modulation(_from_symbols(np.resize(points[:3], 201)), SPS)
    assert result.modulation == "QPSK"
    assert result.qpsk_score > result.bpsk_score
    assert 0.05 < result.confidence < 0.5


def test_a_single_qpsk_point_is_ambiguous_rather_than_a_class():
    """No diversity at all: every rotation leaves the observation unchanged.

    One point is 1-, 2- and 4-fold symmetric simultaneously, so both gated
    scores are exactly zero and neither class is claimed -- at any floor,
    including ``min_confidence=0.0``.
    """
    points = _qpsk_points()
    result = estimate_modulation(
        _from_symbols(np.full(200, points[0])), SPS, min_confidence=0.0
    )
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence == 0.0
    assert result.bpsk_score == 0.0
    assert result.qpsk_score == 0.0


def test_almost_a_single_qpsk_point_is_ambiguous_whichever_pair_is_used():
    """199 of 200 symbols on one point supports no claim about the pair."""
    points = _qpsk_points()
    for second in range(1, points.size):
        symbols = np.concatenate([np.full(199, points[0]), [points[second]]])
        result = estimate_modulation(_from_symbols(symbols), SPS)
        assert result.modulation == "AMBIGUOUS"
        assert result.confidence < 0.02


# --------------------------------------------------------------------------
# Out-of-scope input: reject or report low confidence, never a confident guess
# --------------------------------------------------------------------------


def test_pure_complex_noise_is_ambiguous():
    noise = np.random.default_rng(38).normal(size=6400) + 1j * np.random.default_rng(
        39
    ).normal(size=6400)
    assert estimate_modulation(noise, SPS).modulation == "AMBIGUOUS"
    unrestricted = estimate_modulation(noise, SPS, min_confidence=0.0)
    assert unrestricted.confidence < 0.05


def test_uniform_random_phase_symbols_are_ambiguous():
    phases = 2.0 * np.pi * np.random.default_rng(12).random(N_SYMBOLS)
    result = estimate_modulation(
        _from_symbols(np.exp(1j * phases)), SPS, min_confidence=0.0
    )
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence < 0.05


def test_eight_psk_is_not_confidently_claimed():
    """8-PSK has no net 2-fold or 4-fold asymmetry, so evidence stays tiny."""
    octal = np.random.default_rng(5).integers(0, 8, N_SYMBOLS)
    octal_rotated = np.random.default_rng(6).integers(0, 8, N_SYMBOLS)
    aligned = np.exp(1j * np.pi / 4 * octal)
    assert estimate_modulation(_from_symbols(aligned), SPS).modulation == "AMBIGUOUS"

    rotated = np.exp(1j * (np.pi / 8 + np.pi / 4 * octal_rotated))
    # This draw squeaks past the default floor; the point is that it can only
    # ever do so weakly, never with a confident decision.
    assert estimate_modulation(_from_symbols(rotated), SPS).confidence < 0.1


def test_sixteen_qam_is_a_documented_false_accept():
    """Out of scope: 16-QAM phases are 4-fold symmetric and read as QPSK."""
    grid = np.array([a + 1j * b for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)])
    symbols = grid[np.random.default_rng(8).integers(0, 16, N_SYMBOLS)]
    result = estimate_modulation(_from_symbols(symbols), SPS)
    assert result.modulation == "QPSK"
    assert result.confidence == pytest.approx(0.4427, abs=1e-4)
    # A caller who tightens the floor gets an ambiguous verdict rather than
    # protection: only a separate modulation-family test would give that.
    tightened = estimate_modulation(_from_symbols(symbols), SPS, min_confidence=0.5)
    assert tightened.modulation == "AMBIGUOUS"
    assert tightened.confidence == pytest.approx(result.confidence)


@pytest.mark.parametrize("divisor", [3.0, 8.0, 7.3])
def test_a_generic_unmodulated_tone_is_ambiguous(divisor):
    symbol_rate = FS / SPS
    _, tone = generate_iq_tone(
        fs=FS, freq=symbol_rate / divisor, duration=N_SYMBOLS * SPS / FS
    )
    assert estimate_modulation(tone, SPS).modulation == "AMBIGUOUS"


@pytest.mark.parametrize(
    ("divisor", "expected"), [(4.0, "QPSK"), (2.0, "BPSK")]
)
def test_a_tone_that_aliases_onto_the_symbol_grid_is_a_documented_false_accept(
    divisor, expected
):
    """One sample per symbol turns a tone at Rs/4 or Rs/2 into a constellation.

    The per-symbol phase step is ``360 / divisor`` degrees, so the
    representatives visit exactly four (or two) evenly spaced points. This is
    inherent to symbol-rate decimation, not a flaw in the statistic, and it is
    why this estimator must not be used as a signal detector.
    """
    symbol_rate = FS / SPS
    _, tone = generate_iq_tone(
        fs=FS, freq=symbol_rate / divisor, duration=N_SYMBOLS * SPS / FS
    )
    result = estimate_modulation(tone, SPS)
    assert result.modulation == expected
    assert result.confidence == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Constant and zero input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(256),
        np.zeros(256, dtype=np.complex128),
        np.full(256, 2.5),
        np.full(256, 1.0 + 1.0j),
        np.full(256, -3.0 - 0.5j),
    ],
    ids=[
        "zeros-real",
        "zeros-complex",
        "constant-real",
        "constant-complex",
        "constant-negative",
    ],
)
def test_constant_or_zero_input_is_ambiguous(samples):
    """A measurable block with no diversity: an outcome, not a contract breach.

    Both scores are exactly zero, so the verdict does not depend on the floor
    and callers never have to guard an ordinary measurement with try/except.
    """
    result = estimate_modulation(samples, SPS, min_confidence=0.0)
    assert result.modulation == "AMBIGUOUS"
    assert result.bpsk_score == 0.0
    assert result.qpsk_score == 0.0
    assert result.confidence == 0.0
    assert result.symbol_count == 32


def test_constant_representatives_under_a_varying_block_are_ambiguous():
    """Only the sampled representatives matter, not the rest of the block."""
    samples = np.arange(256, dtype=np.complex128)
    samples[SPS // 2 :: SPS] = 4.0 + 0.0j  # every representative identical
    result = estimate_modulation(samples, SPS, min_confidence=0.0)
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence == 0.0


# --------------------------------------------------------------------------
# Insufficient samples
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_symbols", [1, 2, 8, 15])
def test_too_few_complete_symbols_is_rejected(n_symbols):
    samples = _bpsk(4, n_symbols=n_symbols, seed=64)
    with pytest.raises(ValueError, match="at least 16 complete symbol periods"):
        estimate_modulation(samples, 4)


def test_boundary_offset_can_push_a_block_below_the_symbol_floor():
    samples = _bpsk(4, n_symbols=MIN_SYMBOLS, seed=65)
    assert estimate_modulation(samples, 4).symbol_count == MIN_SYMBOLS
    with pytest.raises(ValueError, match="at least 16 complete symbol periods"):
        estimate_modulation(samples, 4, boundary_offset=1)


def test_a_long_block_with_a_long_symbol_period_can_be_too_short():
    samples = _bpsk(64, n_symbols=15, seed=66)  # 960 samples, only 15 symbols
    assert samples.size == 960
    with pytest.raises(ValueError, match="at least 16 complete symbol periods"):
        estimate_modulation(samples, 64)


def test_the_too_short_message_states_the_floor_and_the_recommendation():
    """16 symbols is the numerical minimum; 64 is what the message advises.

    The contract must not read as though the minimum were also a reliability
    guarantee, so the rejection text carries both numbers.
    """
    samples = _bpsk(4, n_symbols=15, seed=68)
    with pytest.raises(ValueError) as excinfo:
        estimate_modulation(samples, 4)
    message = str(excinfo.value)
    assert "at least 16 complete symbol periods" in message
    assert "64 symbols or more are recommended" in message
    assert f"got {samples.size}." in message


def test_exactly_the_minimum_number_of_symbols_is_accepted():
    """The floor is inclusive: 16 complete symbols is measurable."""
    samples = _bpsk(4, n_symbols=MIN_SYMBOLS, seed=69)
    result = estimate_modulation(samples, 4)
    assert result.symbol_count == MIN_SYMBOLS
    assert result.modulation == "BPSK"


def test_empty_samples_are_rejected():
    with pytest.raises(ValueError, match="at least one value"):
        estimate_modulation(np.array([]), SPS)


@pytest.mark.parametrize(
    "samples",
    [np.zeros((4, 64)), np.zeros((64, 1)), np.zeros((2, 2, 64)), np.array(1.0)],
    ids=["2d", "column", "3d", "scalar"],
)
def test_non_one_dimensional_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_modulation(samples, SPS)


# --------------------------------------------------------------------------
# Non-finite samples
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_real_samples_are_rejected(bad):
    samples = _bpsk(seed=67).real.copy()
    samples[19] = bad
    with pytest.raises(ValueError, match="finite"):
        estimate_modulation(samples, SPS)


@pytest.mark.parametrize(
    "bad",
    [
        complex(float("nan"), 0.0),
        complex(0.0, float("nan")),
        complex(float("inf"), 0.0),
        complex(0.0, float("-inf")),
    ],
)
def test_non_finite_complex_samples_are_rejected(bad):
    samples = _qpsk(seed=68).astype(np.complex128)
    samples[23] = bad
    with pytest.raises(ValueError, match="finite"):
        estimate_modulation(samples, SPS)


def test_a_non_finite_sample_outside_the_representatives_still_rejects():
    """Validation covers the whole block, not only the sampled points."""
    samples = _qpsk(seed=69).astype(np.complex128)
    samples[SPS // 2 + 1] = np.nan  # never sampled, still invalid input
    with pytest.raises(ValueError, match="finite"):
        estimate_modulation(samples, SPS)


# --------------------------------------------------------------------------
# Invalid parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, -8])
def test_samples_per_symbol_below_one_is_rejected(value):
    with pytest.raises(ValueError, match="samples_per_symbol must be >= 1"):
        estimate_modulation(_bpsk(seed=72), value)


@pytest.mark.parametrize(
    "value",
    [8.0, 8.5, "8", None, True, False, np.float64(8.0), [8]],
    ids=["float", "fractional", "str", "none", "true", "false", "np-float", "list"],
)
def test_non_integer_samples_per_symbol_is_rejected(value):
    with pytest.raises(ValueError, match="samples_per_symbol must be an integer"):
        estimate_modulation(_bpsk(seed=73), value)


def test_numpy_integer_samples_per_symbol_is_accepted():
    result = estimate_modulation(_bpsk(seed=74), np.int64(SPS))
    assert result.modulation == "BPSK"
    assert result.samples_per_symbol == SPS
    assert isinstance(result.samples_per_symbol, int)


@pytest.mark.parametrize("value", [-1, -SPS])
def test_negative_boundary_offset_is_rejected(value):
    with pytest.raises(ValueError, match="boundary_offset must be >= 0"):
        estimate_modulation(_bpsk(seed=75), SPS, boundary_offset=value)


@pytest.mark.parametrize(
    "value",
    [1.0, 0.5, "0", None, True, [0]],
    ids=["float", "fractional", "str", "none", "true", "list"],
)
def test_non_integer_boundary_offset_is_rejected(value):
    with pytest.raises(ValueError, match="boundary_offset must be an integer"):
        estimate_modulation(_bpsk(seed=76), SPS, boundary_offset=value)


@pytest.mark.parametrize("value", [SPS, SPS + 1, 4 * SPS])
def test_boundary_offset_at_or_above_the_period_is_rejected(value):
    with pytest.raises(ValueError, match="boundary_offset must be <"):
        estimate_modulation(_bpsk(seed=77), SPS, boundary_offset=value)


def test_boundary_offset_must_be_zero_when_the_period_is_one():
    with pytest.raises(ValueError, match="boundary_offset must be <"):
        estimate_modulation(_bpsk(1, seed=78), 1, boundary_offset=1)


@pytest.mark.parametrize(
    "value", ["low", None, [0.5], {}, complex(0.5, 0.5)],
    ids=["str", "none", "list", "dict", "complex"],
)
def test_non_real_min_confidence_is_rejected(value):
    with pytest.raises(ValueError, match="min_confidence must be a real number"):
        estimate_modulation(_bpsk(seed=79), SPS, min_confidence=value)


@pytest.mark.parametrize(
    "value",
    [-0.1, -1.0, 1.0, 1.5, float("nan"), float("inf"), float("-inf")],
    ids=["small-neg", "neg-one", "one", "above-one", "nan", "inf", "-inf"],
)
def test_out_of_range_min_confidence_is_rejected(value):
    with pytest.raises(ValueError, match=r"min_confidence must lie within"):
        estimate_modulation(_bpsk(seed=80), SPS, min_confidence=value)


@pytest.mark.parametrize("value", [0, 0.0, 0.5, np.float64(0.25)])
def test_in_range_min_confidence_is_accepted(value):
    assert estimate_modulation(_bpsk(seed=81), SPS, min_confidence=value)


# --------------------------------------------------------------------------
# min_confidence semantics
# --------------------------------------------------------------------------


def test_returned_confidence_always_meets_the_requested_floor():
    """Whenever a class is named, its margin clears the caller's floor."""
    for samples, expected in ((_bpsk(seed=82), "BPSK"), (_qpsk(seed=83), "QPSK")):
        unrestricted = estimate_modulation(samples, SPS, min_confidence=0.0)
        assert unrestricted.modulation == expected
        for floor in (0.0, 0.05, 0.25, 0.5, min(unrestricted.confidence, 1.0 - 1e-9)):
            result = estimate_modulation(samples, SPS, min_confidence=floor)
            assert result == unrestricted
            assert result.modulation == expected
            assert result.confidence >= floor


def test_zero_min_confidence_names_a_class_on_any_positive_margin():
    """With the gate open, even structureless input is classified."""
    rng = np.random.default_rng(84)
    noise = rng.normal(size=64 * SPS) + 1j * rng.normal(size=64 * SPS)
    result = estimate_modulation(noise, SPS, min_confidence=0.0)
    assert result.confidence > 0.0
    assert result.modulation in {"BPSK", "QPSK"}
    assert 0.0 <= result.confidence <= 1.0
    # The floor is the only thing that converts a small margin into a
    # withheld verdict; the measurement itself is identical.
    gated = estimate_modulation(noise, SPS, min_confidence=0.5)
    assert gated.modulation == "AMBIGUOUS"
    assert gated.confidence == result.confidence


def test_a_short_structureless_block_can_clear_the_default_floor():
    """A documented limitation: 0.05 is not a noise test at 64 symbols.

    The margin of structureless input decays like ``1 / sqrt(n)``, so a short
    block of complex noise can post a small but above-floor margin. Length is
    what suppresses it -- another reason 64 symbols is a recommendation and not
    a guarantee.
    """
    rng = np.random.default_rng(84)
    short = estimate_modulation(
        rng.normal(size=64 * SPS) + 1j * rng.normal(size=64 * SPS), SPS
    )
    assert short.modulation == "QPSK"  # spurious, and honestly weak
    assert short.confidence == pytest.approx(0.0930, abs=1e-4)

    long_noise = np.random.default_rng(38).normal(
        size=800 * SPS
    ) + 1j * np.random.default_rng(39).normal(size=800 * SPS)
    assert estimate_modulation(long_noise, SPS).modulation == "AMBIGUOUS"


def test_zero_min_confidence_still_cannot_manufacture_a_class():
    """A zero margin is ambiguous at every floor, including zero.

    Opening the gate lets a weak margin through; it does not invent one where
    the observed constellation supports neither class.
    """
    result = estimate_modulation(np.full(256, 1.0 + 1.0j), SPS, min_confidence=0.0)
    assert result.modulation == "AMBIGUOUS"
    assert result.confidence == 0.0


def test_a_floor_above_the_margin_withholds_a_genuine_block():
    """The floor is a caller-set gate, so even clean QPSK can be gated out."""
    unrestricted = estimate_modulation(_qpsk(seed=85), SPS, min_confidence=0.0)
    assert unrestricted.modulation == "QPSK"
    floor = (unrestricted.confidence + 1.0) / 2.0  # above the margin, below 1
    assert unrestricted.confidence < floor < 1.0
    gated = estimate_modulation(_qpsk(seed=85), SPS, min_confidence=floor)
    assert gated.modulation == "AMBIGUOUS"
    # Withheld, not discarded: the measurement itself is unchanged.
    assert gated.confidence == unrestricted.confidence
    assert gated.bpsk_score == unrestricted.bpsk_score
    assert gated.qpsk_score == unrestricted.qpsk_score


def test_an_ambiguous_result_still_reports_the_evidence():
    """No exception, so the numbers have to travel on the result object.

    A caller who wants the old behaviour can raise on ``"AMBIGUOUS"`` and has
    both scores available for the message.
    """
    rng = np.random.default_rng(86)
    noise = rng.normal(size=200 * SPS) + 1j * rng.normal(size=200 * SPS)
    result = estimate_modulation(noise, SPS, min_confidence=0.5)
    assert result.modulation == "AMBIGUOUS"
    assert result.bpsk_score >= 0.0
    assert result.qpsk_score >= 0.0
    assert result.confidence == pytest.approx(
        abs(result.bpsk_score - result.qpsk_score)
    )
    assert result.symbol_count == 200
    assert result.samples_per_symbol == SPS


def test_confidence_is_a_margin_not_a_probability():
    """The documented meaning: a difference of scores, nothing more.

    Two blocks with wildly different reliability can share a margin, and the
    margin never behaves like an error rate -- so the test pins the identity
    rather than any probabilistic property.
    """
    for samples in (
        _bpsk(seed=92),
        _qpsk(seed=93),
        add_awgn(_qpsk(seed=94), snr_db=5.0, rng=np.random.default_rng(940)),
        np.full(256, 1.0 + 1.0j),
    ):
        result = estimate_modulation(samples, SPS, min_confidence=0.0)
        assert result.confidence == abs(result.bpsk_score - result.qpsk_score)
        assert result.confidence + min(result.bpsk_score, result.qpsk_score) == (
            pytest.approx(max(result.bpsk_score, result.qpsk_score))
        )


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "samples, expected",
    [(_bpsk(seed=87), "BPSK"), (_qpsk(seed=88), "QPSK")],
    ids=["bpsk", "qpsk"],
)
def test_repeated_calls_return_identical_results(samples, expected):
    first = estimate_modulation(samples, SPS)
    for _ in range(4):
        again = estimate_modulation(samples, SPS)
        assert again == first
    assert first.modulation == expected


def test_a_copy_of_the_block_gives_bit_identical_scores():
    samples = add_awgn(
        _qpsk(seed=89), snr_db=5.0, rng=np.random.default_rng(890)
    )
    original = estimate_modulation(samples, SPS)
    duplicate = estimate_modulation(samples.copy(), SPS)
    assert duplicate == original
    assert duplicate.confidence == original.confidence


def test_a_list_input_matches_the_array_input_exactly():
    samples = _bpsk(seed=90)
    as_list = estimate_modulation(samples.tolist(), SPS)
    assert as_list == estimate_modulation(samples, SPS)


def test_a_trailing_partial_symbol_does_not_change_the_result():
    """Only complete periods are used, so a truncated last symbol is ignored."""
    samples = _qpsk(seed=91)
    assert samples.size % SPS == 0
    full = estimate_modulation(samples, SPS)
    for extra in range(1, SPS):
        padded = np.concatenate([samples, samples[:extra]])
        assert estimate_modulation(padded, SPS) == full


# --------------------------------------------------------------------------
# Input non-mutation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "samples",
    [_bpsk(seed=92), _qpsk(seed=93), _bpsk(seed=94).real.copy()],
    ids=["bpsk", "qpsk", "real"],
)
def test_the_input_block_is_never_modified(samples):
    reference = samples.copy()
    estimate_modulation(samples, SPS)
    assert np.array_equal(samples, reference)
    assert samples.dtype == reference.dtype


def test_a_read_only_block_is_accepted():
    samples = _qpsk(seed=95)
    samples.setflags(write=False)
    assert estimate_modulation(samples, SPS).modulation == "QPSK"


def test_an_ambiguous_call_also_leaves_the_input_untouched():
    samples = _from_symbols(np.full(64, 1.0 + 1.0j))
    reference = samples.copy()
    assert estimate_modulation(samples, SPS).modulation == "AMBIGUOUS"
    assert np.array_equal(samples, reference)
    assert samples.dtype == reference.dtype


def test_a_rejected_call_also_leaves_the_input_untouched():
    samples = _bpsk(4, n_symbols=8, seed=101)
    reference = samples.copy()
    with pytest.raises(ValueError):
        estimate_modulation(samples, 4)
    assert np.array_equal(samples, reference)


# --------------------------------------------------------------------------
# Result object contract
# --------------------------------------------------------------------------


def test_result_is_a_frozen_modulation_estimate_with_the_documented_fields():
    result = estimate_modulation(_qpsk(seed=96), SPS)
    assert isinstance(result, ModulationEstimate)
    assert set(vars(result)) == {
        "modulation",
        "confidence",
        "bpsk_score",
        "qpsk_score",
        "symbol_count",
        "samples_per_symbol",
    }
    assert isinstance(result.modulation, str)
    assert isinstance(result.confidence, float)
    assert isinstance(result.bpsk_score, float)
    assert isinstance(result.qpsk_score, float)
    assert isinstance(result.symbol_count, int)
    assert isinstance(result.samples_per_symbol, int)
    assert not isinstance(result.symbol_count, np.integer)
    assert not isinstance(result.samples_per_symbol, np.integer)


@pytest.mark.parametrize(
    "field",
    [
        "modulation",
        "confidence",
        "bpsk_score",
        "qpsk_score",
        "symbol_count",
        "samples_per_symbol",
    ],
)
def test_result_fields_cannot_be_reassigned(field):
    result = estimate_modulation(_bpsk(seed=97), SPS)
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, 0)


@pytest.mark.parametrize(
    "sps, n_symbols, offset",
    [(1, 64, 0), (3, 64, 2), (SPS, N_SYMBOLS, 0), (SPS, 64, 5), (16, 40, 9)],
)
def test_reported_counts_match_the_analyzed_block(sps, n_symbols, offset):
    samples = _qpsk(sps, n_symbols=n_symbols, seed=98)[offset:]
    result = estimate_modulation(
        samples, sps, boundary_offset=(-offset) % sps, min_confidence=0.0
    )
    expected = (samples.size - (-offset) % sps) // sps
    assert result.symbol_count == expected
    assert result.samples_per_symbol == sps


@pytest.mark.parametrize(
    "samples", [_bpsk(seed=99), _qpsk(seed=100)], ids=["bpsk", "qpsk"]
)
def test_scores_and_confidence_stay_within_the_unit_interval(samples):
    result = estimate_modulation(samples, SPS)
    assert 0.0 <= result.bpsk_score <= 1.0
    assert 0.0 <= result.qpsk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == pytest.approx(
        abs(result.bpsk_score - result.qpsk_score)
    )
    winner = "BPSK" if result.bpsk_score >= result.qpsk_score else "QPSK"
    assert result.modulation == winner


def test_the_modulation_field_is_always_one_of_three_documented_strings():
    """The whole outcome vocabulary, exercised over the adversarial families."""
    points = _qpsk_points()
    rng = np.random.default_rng(102)
    blocks = [
        _bpsk(seed=103),
        _qpsk(seed=104),
        _from_symbols(np.full(200, points[0])),
        _two_point_stream(points, _antipodal_pairs(points)[0], 100),
        _two_point_stream(points, _adjacent_pairs(points)[0], 1),
        _skewed_qpsk_stream(points, 0.9),
        np.full(256, 1.0 + 1.0j),
        np.zeros(256, dtype=np.complex128),
        rng.normal(size=1600) + 1j * rng.normal(size=1600),
        _from_symbols(np.exp(2j * np.pi * rng.random(200))),
    ]
    seen = set()
    for block in blocks:
        for floor in (0.0, 0.05, 0.5, 0.95):
            result = estimate_modulation(block, SPS, min_confidence=floor)
            assert result.modulation in {"BPSK", "QPSK", "AMBIGUOUS"}
            if result.modulation == "AMBIGUOUS":
                assert result.confidence < max(floor, np.nextafter(0.0, 1.0))
            else:
                assert result.confidence >= floor
            seen.add(result.modulation)
    assert seen == {"BPSK", "QPSK", "AMBIGUOUS"}
