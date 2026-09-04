"""Bounded symbol-timing recovery for controlled rectangular waveforms.

This module provides a single, narrowly-scoped primitive:
:func:`recover_symbol_timing`, which recovers the best symbol sampling
phase of a rectangular-pulse BPSK or QPSK sample block, of the kind
produced by :func:`iqwav.modulation.bpsk_waveform`,
:func:`iqwav.modulation.qpsk_waveform`, and the sample-and-hold
:func:`iqwav.modulation.symbols_to_samples` they are built on, when the
integer ``samples_per_symbol`` is **already known**. It returns the
samples taken at that phase, one per symbol.

Convention
----------
The candidate timing phases are the ``samples_per_symbol`` integer
offsets ``0, 1, ..., samples_per_symbol - 1``: exactly one symbol period,
no wider and no finer.

The reported ``timing_offset`` is the recovered **symbol-boundary
phase**: symbols start at sample indices congruent to it modulo
``samples_per_symbol``, so the first complete symbol period of the block
is ``samples[timing_offset : timing_offset + samples_per_symbol]``. This
is the convention of
:attr:`iqwav.estimation.SymbolRateEstimate.boundary_offset` and of the
``boundary_offset`` argument of
:func:`iqwav.estimation.estimate_modulation`, so a recovered offset can
be handed straight to the latter.

Each recovered symbol is read at the **interior midpoint** of its period,
``timing_offset + k * samples_per_symbol + samples_per_symbol // 2`` for
``k = 0, 1, ..., symbol_count - 1``: the sample furthest from both symbol
edges, matching the representative-sample convention already used by
:func:`iqwav.estimation.estimate_modulation`. Those samples are returned
as they are, selected and copied, never averaged, filtered, or rescaled.

Concretely, discarding ``d`` leading samples from a generator output
whose symbols began at index 0 leaves the boundary phase at
``(-d) % samples_per_symbol``, which is what this function reports.

Timing-quality criterion
------------------------
For a candidate phase ``p`` the block is cut into consecutive windows of
``samples_per_symbol`` samples starting at ``p``. When ``p`` is the true
boundary phase every window lies inside one symbol and, for a rectangular
pulse, is exactly constant; at every other phase each window straddles a
symbol boundary, so every boundary whose two symbols differ makes its
window non-constant. The score is how much of the block's sample
dispersion the piecewise-constant symbol model at phase ``p`` explains::

    W(p)       = sum over the scored windows w at p, and the samples x
                 in each w, of |x - mean(w)|^2
    D          = sum over the scored region of |x - mean(region)|^2
    quality(p) = 1 - W(p) / D

That is the explained-dispersion fraction (a one-way ANOVA eta-squared)
of the symbol-window grouping. It lies in ``[0, 1]``, is unchanged by
amplitude scaling and by a constant carrier phase rotation, and is
exactly ``1.0`` when every scored window is constant, that is when the
windows coincide with symbol periods. The phase with the highest score
wins; when scores are exactly equal the smallest offset wins, which keeps
the result deterministic.

Every candidate phase is scored on the same number of windows,
``scored_symbol_count = (len(samples) - (samples_per_symbol - 1)) //
samples_per_symbol``, and against the same denominator ``D``, taken over
the union of all phases' windows, ``samples[:samples_per_symbol - 1 +
scored_symbol_count * samples_per_symbol]``. Both choices exist so the
``samples_per_symbol`` scores are directly comparable instead of each
being measured over a slightly different stretch of the block.

Identifiability
---------------
Two phases can both score ``1.0`` only if the scored region is constant,
because constant windows at two different phases overlap and the overlaps
chain across the block. One observed transition is therefore enough to
pin the boundary phase of a rectangular block exactly. When the block
never changes, an all-zero or constant block or a stream repeating one
symbol, ``D`` is zero, every phase is equally good, and the timing offset
genuinely carries no information: every offset is listed in
``tied_offsets``, ``margin`` is ``0.0``, and offset ``0`` is selected.
That is reported rather than raised, because a block that cannot separate
the phases is a measurement outcome; ``ValueError`` is reserved for a
violated contract.

Only transitions inside the commonly scored region are visible to the
search. A block whose sole transition falls in the leading
``samples_per_symbol - 1`` samples, or in the trailing partial window, is
reported as a tie instead of being resolved.

Scope
-----
This is a **bounded integer timing-phase search for controlled
rectangular signals, not a general-purpose timing-recovery system**. It
is explicitly not:

- sub-sample timing recovery: only the ``samples_per_symbol`` integer
  phases are searched and no interpolator is applied, so nothing finer
  than one sample is claimed or delivered. (For an ideally sampled
  rectangular pulse a constant fractional offset ``d`` yields the same
  sample stream as the integer offset ``ceil(d)``, so the integer grid is
  not an approximation of that model. A receiver that smears transitions
  is a different model, and there only the nearest integer phase is
  reported.)
- blind timing recovery: ``samples_per_symbol`` must already be known and
  exact,
- a timing loop of any kind: no PLL, no Gardner detector, no
  Mueller-and-Muller detector, no Costas loop, no feedback and no
  retained state. One phase is chosen for the whole block and nothing is
  tracked across it,
- carrier recovery, CFO estimation, or CFO correction (see
  :func:`iqwav.estimation.estimate_frequency_offset` and
  :func:`iqwav.dsp.correct_frequency_offset`),
- symbol-rate estimation (see
  :func:`iqwav.estimation.estimate_symbol_rate`, which reports both the
  integer period and a boundary phase for these waveforms),
- modulation recognition (see
  :func:`iqwav.estimation.estimate_modulation`): BPSK, QPSK, and any
  other rectangular symbol stream are treated identically here,
- support for FSK, QAM, or pulse-shaped (for example root-raised-cosine)
  waveforms, whose symbol periods are not piecewise constant,
- framing, FEC, de-interleaving, payload recovery, or any user interface.

The input array is never modified, filtered, resampled, or normalized.
There is no ``sample_rate`` argument: the search runs entirely in samples
and the recovered offset is a sample count, so the sample rate, known or
not, would not change any result.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["SymbolTimingRecovery", "recover_symbol_timing"]

# Fewest symbol windows every candidate phase is scored on. Four windows put
# at least three interior symbol boundaries inside the scored region at every
# candidate phase, which is what lets a stream that changes at some of its
# boundaries separate the phases. It is a numerical floor, not a reliability
# guarantee.
_MIN_SYMBOLS = 4

# Quality difference within which a rival phase counts as tied with the best.
# Scores lie in [0, 1] and an aligned phase scores exactly 1.0, so this
# default only absorbs floating-point dust; a genuine phase difference is
# orders of magnitude larger.
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, eq=False)
class SymbolTimingRecovery:
    """Outcome of a bounded symbol-timing-phase search.

    Attributes:
        symbols: The recovered symbol samples, one per complete symbol
            period at the selected phase, read at the interior midpoint
            ``timing_offset + k * samples_per_symbol +
            samples_per_symbol // 2``. These are input sample values,
            selected and copied, so the array has the same dtype as the
            input block, shares no memory with it, and holds no averaged
            or rescaled estimates.
        timing_offset: The selected symbol-boundary phase, in
            ``[0, samples_per_symbol)``: symbols start at sample indices
            congruent to this value modulo ``samples_per_symbol``.
        first_symbol_index: Index of the first returned symbol sample,
            ``timing_offset + samples_per_symbol // 2``, so callers need
            not re-derive the midpoint convention.
        samples_per_symbol: The samples-per-symbol value supplied by the
            caller, echoed back.
        symbol_count: Number of returned symbols,
            ``(len(samples) - timing_offset) // samples_per_symbol``.
        scored_symbol_count: Number of symbol windows every candidate
            phase was scored on. Never larger than ``symbol_count``.
        quality: Timing quality of the selected phase, in ``[0, 1]``.
            Exactly ``1.0`` when every scored window at that phase is
            constant, which is the clean rectangular case.
        margin: ``quality`` minus the best score of any other candidate
            phase, in ``[0, 1]``; at most ``tie_tolerance`` when another
            phase ties. With ``samples_per_symbol == 1`` there is no other
            phase and the margin is ``quality`` itself, that single phase
            being trivially correct.
        phase_qualities: Timing quality of every candidate phase, in
            offset order, so the whole one-symbol-period search is
            inspectable. Length ``samples_per_symbol``.
        tied_offsets: Every candidate phase whose quality is within
            ``tie_tolerance`` of the best, ascending, always including
            ``timing_offset``. Length 1 means the phase is uniquely
            identifiable; longer means the block cannot separate the
            listed phases and the smallest of them was selected.
    """

    symbols: npt.NDArray[np.generic]
    timing_offset: int
    first_symbol_index: int
    samples_per_symbol: int
    symbol_count: int
    scored_symbol_count: int
    quality: float
    margin: float
    phase_qualities: tuple[float, ...]
    tied_offsets: tuple[int, ...]


def _validate_samples(samples: object) -> npt.NDArray[np.generic]:
    """Validate the sample block itself, returning it as an ndarray."""
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    return values


def _validate_integer(value: object, *, name: str, minimum: int) -> int:
    """Validate one integer argument."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return int(value)


def _validate_tie_tolerance(value: object) -> float:
    """Validate the tie tolerance, which must lie in ``[0, 1)``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"tie_tolerance must be a real number, got {value!r}."
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number < 1.0:
        raise ValueError(
            f"tie_tolerance must lie within [0.0, 1.0), got {value!r}."
        )
    return number


def _phase_qualities(
    values: npt.NDArray[np.generic], period: int, windows: int
) -> npt.NDArray[np.float64]:
    """Score every candidate timing phase, in offset order.

    Implements the criterion documented in the module docstring: the
    fraction of the scored region's sample dispersion that a
    piecewise-constant symbol model at each phase explains.
    """
    region = values[: (period - 1) + windows * period]
    dispersion = float(np.sum(np.abs(region - np.mean(region)) ** 2))
    if dispersion <= 0.0:
        # A constant scored region leaves every phase perfectly
        # piecewise constant, equally good, and undividable below.
        return np.ones(period, dtype=np.float64)
    qualities = np.empty(period, dtype=np.float64)
    for phase in range(period):
        block = values[phase : phase + windows * period]
        block = block.reshape(windows, period)
        deviations = block - block.mean(axis=1, keepdims=True)
        within = float(np.sum(np.abs(deviations) ** 2))
        # A window mean minimises its own squared deviations and the
        # windows sit inside the scored region, so this ratio cannot
        # exceed 1 in exact arithmetic; the clip absorbs rounding.
        qualities[phase] = min(max(1.0 - within / dispersion, 0.0), 1.0)
    return qualities


def recover_symbol_timing(
    samples: npt.ArrayLike,
    samples_per_symbol: int,
    *,
    tie_tolerance: float = _TIE_TOLERANCE,
) -> SymbolTimingRecovery:
    """Recover the best symbol sampling phase of a rectangular block.

    Scores each of the ``samples_per_symbol`` integer timing phases, one
    full symbol period and nothing finer, with the explained-dispersion
    criterion documented in the module docstring; selects the best; and
    returns the input samples read at the interior midpoint of every
    complete symbol period at that phase.

    Assumes a controlled block: complex baseband IQ (real input is
    accepted and stays real), a known and exact integer
    ``samples_per_symbol``, a rectangular constant-envelope BPSK or QPSK
    symbol stream, one constant timing offset for the whole block, and no
    significant frequency drift across it. Nothing but the timing phase is
    estimated, and no loop, interpolator, or tracker is involved.

    Args:
        samples: 1-D sample block, complex IQ or real. Must be non-empty,
            finite, and hold at least ``5 * samples_per_symbol - 1``
            samples, so that every candidate phase can be scored on four
            symbol windows.
        samples_per_symbol: Known integer samples per symbol, ``>= 1``.
            A value of ``1`` leaves a single candidate phase, making the
            search trivial and the offset necessarily ``0``.
        tie_tolerance: Quality difference within which a rival phase
            counts as tied with the best and is reported in
            ``tied_offsets``. Must be a real number in ``[0.0, 1.0)``. The
            default only absorbs floating-point dust.

    Returns:
        A :class:`SymbolTimingRecovery` holding the recovered symbol
        samples, the selected boundary phase, and the score of every
        candidate phase. The input array is not modified and the returned
        symbols share no memory with it.

    Raises:
        ValueError: If ``samples`` is not a 1-D non-empty finite array, if
            ``samples_per_symbol`` is not an integer ``>= 1``, if
            ``tie_tolerance`` is outside ``[0.0, 1.0)``, or if the block is
            too short to score four symbol windows at every candidate
            phase. A block that cannot separate the phases does not raise:
            that is reported through ``tied_offsets`` and ``margin``.
    """
    period = _validate_integer(
        samples_per_symbol, name="samples_per_symbol", minimum=1
    )
    tolerance = _validate_tie_tolerance(tie_tolerance)
    values = _validate_samples(samples)
    required = (period - 1) + _MIN_SYMBOLS * period
    if values.size < required:
        raise ValueError(
            f"samples must contain at least {required} values to score "
            f"{_MIN_SYMBOLS} symbol windows at each of the {period} candidate "
            f"phases for samples_per_symbol={period}, got {values.size}."
        )
    windows = (values.size - (period - 1)) // period
    qualities = _phase_qualities(values, period, windows)
    # argmax reports the first maximum, so exactly equal scores select the
    # smallest offset and the result stays deterministic.
    offset = int(np.argmax(qualities))
    quality = float(qualities[offset])
    rivals = np.delete(qualities, offset)
    margin = quality - float(np.max(rivals)) if rivals.size else quality
    tied = tuple(
        int(phase) for phase in np.flatnonzero(qualities >= quality - tolerance)
    )
    count = (values.size - offset) // period
    indices = offset + np.arange(count, dtype=np.int64) * period + period // 2
    return SymbolTimingRecovery(
        symbols=values[indices],
        timing_offset=offset,
        first_symbol_index=offset + period // 2,
        samples_per_symbol=period,
        symbol_count=count,
        scored_symbol_count=windows,
        quality=quality,
        margin=margin,
        phase_qualities=tuple(float(value) for value in qualities),
        tied_offsets=tied,
    )
