"""Bounded BPSK-vs-QPSK modulation estimation for controlled symbol streams.

This module provides a single, narrowly-scoped estimator:
:func:`estimate_modulation`, which decides whether the symbols of a
**controlled, rectangular-pulse, symbol-synchronous** sampled waveform of
the kind produced by :mod:`iqwav.modulation`
(:func:`iqwav.modulation.bpsk_waveform`,
:func:`iqwav.modulation.qpsk_waveform`, and the underlying
:func:`iqwav.modulation.symbols_to_samples`) lie on the **BPSK** or the
**QPSK** constellation, given a known integer ``samples_per_symbol`` and a
known symbol-boundary phase.

Constellations
--------------
Both reference constellations are read off the actual IQWAV generators
rather than assumed:

- :func:`iqwav.modulation.bpsk_modulate` emits ``{+1 + 0j, -1 + 0j}``: two
  antipodal points on the real axis, phases ``{0, 180}`` degrees.
- :func:`iqwav.modulation.qpsk_modulate` emits
  ``{(+1 + 1j), (-1 + 1j), (+1 - 1j), (-1 - 1j)} / sqrt(2)``: four points
  on the diagonals, phases ``{45, 135, -45, -135}`` degrees. That 45-degree
  orientation matters when reasoning about subsets: the antipodal pairs are
  ``(45, -135)`` and ``(135, -45)``, while ``(45, -45)`` is an *adjacent*
  90-degree pair.

Only the *rotational symmetry order* of those two point sets is used, so
the statistic below is insensitive to the 45-degree offset and to any
other constant phase rotation of the analyzed block.

Method
------
The generators hold each symbol constant for exactly
``samples_per_symbol`` samples (``numpy.repeat``), so one sample taken from
the **interior** of each known symbol period is an exact, noise-free
representative of that symbol. The estimator takes the midpoint of every
complete period, ``boundary_offset + k * samples_per_symbol +
samples_per_symbol // 2``, which is the sample furthest from both symbol
edges and therefore the most tolerant of a boundary phase that is off by a
sample.

Raising every representative to the power ``m`` collapses a constellation
with ``m``-fold rotational symmetry onto a single point: a first power
leaves a single-point set unchanged, squaring maps two antipodal points onto
one point, and a fourth power maps any subset of a 90-degree-spaced
four-point set onto one point. The estimator therefore measures, for ``m``
in ``{1, 2, 4}``, how concentrated ``z ** m`` is:

    C_m = sqrt(max(0, (n * abs(mean(z ** m)) ** 2 - mean(abs(z) ** (2 * m)))
                  / (n - 1))) / mean(abs(z) ** m)

The numerator is the finite-sample-unbiased estimator of
``abs(E[z ** m]) ** 2``: because ``E[abs(mean(w)) ** 2] = abs(E[w]) ** 2 +
Var(w) / n``, subtracting the sample second moment removes the upward bias
that would otherwise make *any* short block look symmetric, and in
particular keeps a random imbalance in the symbol stream from being read as
structure. By Cauchy-Schwarz the result always lies in ``[0, 1]``; it is
exactly 1 for a clean constant-modulus block with the symmetry and about 0
without it, and it is invariant to amplitude scaling and to constant phase
rotation.

The three orders form a ladder, and each class must show its own symmetry
while *not* showing the lower-order symmetry that would make the observation
degenerate:

    bpsk_score = C_2 * (1 - C_1)
    qpsk_score = C_4 * (1 - C_2)

``C_2`` near 1 means the symbols occupy essentially one antipodal pair,
which is what BPSK looks like; ``C_4`` near 1 with a low ``C_2`` means
points on a 90-degree grid that squaring does *not* collapse, which is what
QPSK looks like. ``C_1`` measures how completely the observed symbols
concentrate on a *single* point, which is trivially symmetric under every
rotation and is therefore evidence for neither class: gating ``C_2`` by
``1 - C_1`` is what stops a nearly constant block from reading as confident
BPSK. No matching ``C_1`` term is needed for QPSK, because concentration
also drives ``C_2`` to 1 and the ``1 - C_2`` gate already covers it, while a
*balanced* 90-degree pair has a high ``C_1`` yet is unambiguously not
antipodal.

For a clean two-point antipodal observation the gate has an almost exact
reading: ``C_1 = 1 - 2 * f`` up to the ``O(1 / n)`` debias term, where ``f``
is the fraction of symbols on the minority point, so ``bpsk_score = 2 * f``
to within about 0.002 at 400 symbols. The default ``min_confidence`` of 0.05
therefore names BPSK only once the minority point carries about 2.5 per cent
of the symbols.

The decision is the larger score and ``confidence`` is the margin between
them. A block whose margin falls below ``min_confidence``, or that supports
neither class at all, is reported as ``"AMBIGUOUS"`` rather than forced into
one of the two classes; nothing about the observed symbols is treated as an
error, so no data-dependent exception is raised. Raw (rather than per-symbol
unit-normalized) moments are used deliberately: a circular complex Gaussian
noise term has zero non-conjugate moments, so
``E[(s + w) ** m] = E[s ** m]`` exactly, whereas normalizing each symbol to
unit magnitude first amplifies its phase noise ``m``-fold and costs several
dB of usable SNR.

What is classified
------------------
The **observed** constellation, never the transmitter's intent. ``"BPSK"``
means the analyzed symbols form an antipodal pair; it does **not** mean the
generator was a BPSK generator. A QPSK stream restricted to one of its
antipodal pairs emits exactly a rotated BPSK constellation, so ``"BPSK"``
should be read as *"antipodal observation, from either generator"*: with
respect to the transmitter that verdict is itself ambiguous, and this module
documents that ambiguity rather than pretending to resolve it. It cannot be
resolved, and not because of a weak statistic: the two blocks are bit-identical
inputs, so no function of the samples can separate them, and the only feature
that differs -- absolute phase -- is deliberately discarded so that a rotated
block classifies the same way. The one thing the estimator will not do is claim
``"QPSK"`` for such a block. Where the observed symbols carry too little
constellation diversity to support either class -- a single point, or a
two-point set so skewed that the minority point is nearly absent -- the result
is ``"AMBIGUOUS"`` instead of a guess.

How many symbols
----------------
16 complete symbols is the **numerical minimum**: below that the ``n - 1``
debiasing has too little to work with, and a shorter block is a contract
violation that raises :class:`ValueError`. It is a floor, not a guarantee of
anything -- a random 16-symbol QPSK draw can genuinely land mostly on one
antipodal pair and then correctly reads as BPSK. **At least 64 symbols is the
recommendation**, being the shortest length at which every draw tried was read
correctly; it is reported in the too-short message and documented, never
enforced.

Scope
-----
This is a **bounded, controlled-signal modulation primitive**, not blind
modulation recognition. It is explicitly **not**:

- blind automatic modulation recognition or classification (the symbol
  period and boundary phase must already be known, only two candidate
  constellations are considered, and no decision about whether the input is
  a PSK signal at all is made),
- a signal detector or a modulation-family test: ``"AMBIGUOUS"`` means the
  observed symbols do not favour BPSK over QPSK, not that the input is
  noise, and a low margin is not a detection threshold,
- symbol-timing recovery, interpolation, or a timing loop (nothing is
  tracked or corrected; a single known boundary phase is used),
- carrier recovery, CFO estimation, or CFO correction (see
  :func:`iqwav.estimation.estimate_peak_frequency` and
  :func:`iqwav.estimation.estimate_frequency_offset`),
- symbol-rate estimation (see
  :func:`iqwav.estimation.estimate_symbol_rate`, which supplies both the
  integer period and the boundary phase for these waveforms),
- support for FSK, QAM, higher-order PSK, or any other modulation family,
- framing, FEC, de-interleaving, payload recovery, machine learning, or any
  user interface.

The input samples are never modified, filtered, resampled, or corrected;
the measurement is the entire output.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["ModulationEstimate", "estimate_modulation"]

# Returned in place of a class name when the observed symbols do not support
# either candidate constellation by at least ``min_confidence``.
_AMBIGUOUS = "AMBIGUOUS"
# Fewest complete symbols for which the debiased coherences are meaningful.
# This is a numerical floor, not a reliability guarantee.
_MIN_SYMBOLS = 16
# Fewest complete symbols at which both classes were measured to be read
# correctly across every seed tried. Purely advisory: it is reported in the
# too-short message and documented, never enforced.
_RECOMMENDED_SYMBOLS = 64
# Symmetry orders probed: 1-fold detects concentration on a single point
# (evidence for neither class), 2-fold identifies an antipodal pair (BPSK),
# 4-fold identifies a 90-degree-spaced quaternary set (QPSK).
_DIVERSITY_ORDER = 1
_BPSK_ORDER = 2
_QPSK_ORDER = 4
# Distance from 1.0 within which a coherence is snapped to exactly 1.0. The
# ratio is algebraically <= 1 and exactly 1 for a set the rotation leaves
# unchanged, so this only repairs floating-point rounding: it makes a
# degenerate block score exactly 0 for both classes instead of leaving dust
# that a zero min_confidence would read as evidence.
_COHERENCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ModulationEstimate:
    """Result of a single bounded BPSK-vs-QPSK modulation estimate.

    Attributes:
        modulation: The observed constellation: ``"BPSK"``, ``"QPSK"``, or
            ``"AMBIGUOUS"`` when the block supports neither class by at
            least ``min_confidence``. This describes the constellation the
            analyzed symbols were *observed* to occupy, never the generator
            that produced them: ``"BPSK"`` is an antipodal observation, which
            a QPSK stream restricted to one antipodal pair also produces, so
            as a statement about the transmitter that verdict is ambiguous.
            See ``What is classified`` in the module docstring and the
            ``Limitations`` below.
        confidence: Classification margin ``abs(bpsk_score - qpsk_score)``,
            in ``[0, 1]``. **A margin, not a probability and not an error
            rate**: it says how much more one candidate constellation is
            supported than the other, and nothing about how often the
            decision is right. It is not calibrated, cannot be compared
            across block lengths (see ``Limitations``), and 1.0 means only
            that the observed constellation is clean and fully symmetric.
            Reported for an ``"AMBIGUOUS"`` result too, where it is below
            ``min_confidence`` by construction.
        bpsk_score: Evidence for BPSK in ``[0, 1]``: the debiased 2-fold
            symmetry coherence gated by the absence of 1-fold concentration,
            ``C_2 * (1 - C_1)``. It reaches 1 for a balanced antipodal pair
            and falls to 0 as the symbols concentrate on a single point,
            which is trivially symmetric under every rotation and therefore
            evidence for neither class.
        qpsk_score: Evidence for QPSK in ``[0, 1]``: the debiased 4-fold
            coherence gated by the absence of 2-fold structure,
            ``C_4 * (1 - C_2)``, which reaches 1 when the symbols occupy
            90-degree-spaced points that squaring does not collapse.
        symbol_count: Number of complete symbol periods analyzed, i.e. the
            number of representative samples taken:
            ``(len(samples) - boundary_offset) // samples_per_symbol``.
            Always ``>= 16``; at least 64 is recommended.
        samples_per_symbol: The known integer symbol period, in samples,
            that was used to place the representative samples. Echoed back
            unchanged.
    """

    modulation: str
    confidence: float
    bpsk_score: float
    qpsk_score: float
    symbol_count: int
    samples_per_symbol: int


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
    """Validate one integer estimator argument."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return int(value)


def _validate_min_confidence(value: object) -> float:
    """Validate the confidence floor, which must lie in ``[0, 1)``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"min_confidence must be a real number, got {value!r}."
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number < 1.0:
        raise ValueError(
            f"min_confidence must lie within [0.0, 1.0), got {value!r}."
        )
    return number


def _representative_symbols(
    values: np.ndarray, samples_per_symbol: int, boundary_offset: int
) -> npt.NDArray[np.complex128]:
    """Return one interior sample per complete symbol period.

    The midpoint ``samples_per_symbol // 2`` of each period is used: it is
    the sample furthest from both symbol edges, so a boundary phase that is
    off by a sample still lands inside the intended symbol. The input array
    is only read; a new complex array is returned.
    """
    count = (values.size - boundary_offset) // samples_per_symbol
    indices = (
        boundary_offset
        + np.arange(count, dtype=np.int64) * samples_per_symbol
        + samples_per_symbol // 2
    )
    return values[indices].astype(np.complex128)


def _normalized_symbols(
    symbols: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Scale the representatives so the largest magnitude is 1.

    Applied before taking eighth-order quantities, so a block recorded at any
    physical amplitude can neither overflow nor underflow. Every coherence is
    a ratio of equal-degree moments, so this cancels exactly and no score
    changes. An all-zero block has no peak to divide by and is returned
    unchanged; its coherences are all 0, which the caller reads as
    ``"AMBIGUOUS"``.
    """
    peak = float(np.max(np.abs(symbols)))
    if peak <= 0.0:
        return symbols
    return symbols / peak


def _symmetry_coherence(symbols: npt.NDArray[np.complex128], order: int) -> float:
    """Debiased ``order``-fold rotational-symmetry coherence, in ``[0, 1]``.

    Returns how completely ``symbols ** order`` collapses onto a single
    point, corrected for the upward bias that finite ``n`` would otherwise
    contribute. Exactly 1.0 for a clean constant-modulus block with the
    symmetry, about 0 for a block without it. Order 1 is a meaningful probe
    and not a degenerate case: raising to the first power changes nothing, so
    the result measures how far the symbols have collapsed onto a single
    constellation point.

    The ratio is provably in ``[0, 1]`` by Cauchy-Schwarz and provably equal
    to 1 for a set the rotation leaves unchanged, but evaluating it in
    floating point can land a few units in the last place either side of 1.
    Both directions are snapped back to the exact algebraic value, so a
    degenerate block yields gate factors of exactly ``1 - 1 == 0`` rather
    than rounding dust that would read as evidence.
    """
    count = symbols.size
    powered = symbols**order
    magnitudes = np.abs(powered)
    denominator = float(np.mean(magnitudes))
    if denominator <= 0.0:
        return 0.0
    biased = float(abs(np.mean(powered))) ** 2
    second = float(np.mean(magnitudes * magnitudes))
    unbiased = (count * biased - second) / (count - 1)
    coherence = float(np.sqrt(max(0.0, unbiased))) / denominator
    if coherence >= 1.0 - _COHERENCE_TOLERANCE:
        return 1.0
    return coherence


def estimate_modulation(
    samples: npt.ArrayLike,
    samples_per_symbol: int,
    *,
    boundary_offset: int = 0,
    min_confidence: float = 0.05,
) -> ModulationEstimate:
    """Decide whether a controlled symbol stream is BPSK, QPSK, or ambiguous.

    Takes one representative sample from the interior of every complete
    symbol period, measures the debiased 1-fold, 2-fold and 4-fold
    rotational symmetry of that observed constellation, and returns the
    better supported of the two candidate classes together with the margin
    between them -- or ``"AMBIGUOUS"`` when neither is supported by at least
    ``min_confidence``. See the module docstring for the statistic and for
    the two reference constellations, which are read off
    :mod:`iqwav.modulation` rather than assumed.

    Assumptions:
        - The block is a rectangular-pulse, symbol-synchronous waveform:
          each symbol is held constant for exactly ``samples_per_symbol``
          samples, as produced by
          :func:`iqwav.modulation.symbols_to_samples`.
        - ``samples_per_symbol`` and ``boundary_offset`` are already known
          and exact. :func:`iqwav.estimation.estimate_symbol_rate` reports
          both for these waveforms; nothing is searched or recovered here.
        - The symbols come from one of the two supported constellations,
          up to an arbitrary constant amplitude scaling and an arbitrary
          constant phase rotation, both of which the statistic ignores.
        - Any residual carrier offset is small enough that the carrier
          phase is essentially constant across the whole block.

    Limitations:
        - **The observed constellation is what is classified**, not the
          generator's intent, and no attempt is made to recover the intent.
          A QPSK stream restricted to one antipodal pair -- ``(45, -135)`` or
          ``(135, -45)`` degrees -- emits exactly a rotated BPSK
          constellation and is reported as ``"BPSK"`` at confidence 1.0.
          That is not a misclassification but an **irreducible ambiguity**,
          documented here rather than resolved: the two blocks are identical
          sample-for-sample, so no statistic can separate them, and the only
          feature that differs -- absolute phase -- is discarded on purpose so
          that rotation does not change the answer. Treat ``"BPSK"`` as
          ambiguous with respect to the transmitter. What the estimator does
          guarantee is that it never returns ``"QPSK"`` for such a block, and
          that as the minority point of the pair vanishes the result becomes
          ``"AMBIGUOUS"`` rather than a confident claim.
        - **A two-point subset is not distinguished from the full
          constellation.** Two QPSK points 90 degrees apart are not an
          antipodal pair, so they read ``"QPSK"`` at confidence 1.0 even
          though only half the constellation was used. ``"QPSK"`` means
          "90-degree-spaced", not "all four points were seen"; a
          three-point subset reads ``"QPSK"`` at about 0.45.
        - **Skew degrades a QPSK block gracefully into ambiguity, not into a
          false BPSK claim.** Piling probability onto one QPSK point raises
          ``C_2`` and lowers ``C_4`` together: a three-to-one skew still
          reads ``"QPSK"`` at about 0.11, and by a 0.85 majority the margin
          has fallen to about 0.04 and the block is ``"AMBIGUOUS"`` at the
          default floor. BPSK is immune to the corresponding skew, since
          squaring collapses an antipodal pair whatever the mix; a BPSK margin
          is twice the minority symbol fraction (up to the ``O(1 / n)``
          debias term), so the default floor withholds a decision below about
          2.5 percent minority symbols.
        - **Only these two classes are considered.** Another modulation is
          not detected and may be reported as whichever of the two it
          resembles: 16-QAM has strong 4-fold phase symmetry and reads
          ``"QPSK"`` at about 0.44, and 8-PSK rotated by ``pi / 8`` squeaks
          past the default floor as ``"BPSK"`` at about 0.06. Unrotated
          8-PSK and uniformly random phases score exactly 0 for both classes
          and are ``"AMBIGUOUS"``. ``min_confidence`` is a sanity floor, not
          a modulation detector, and this estimator performs no blind
          modulation recognition.
        - **An unmodulated carrier is not rejected when it aliases onto the
          symbol grid.** Taking one sample per symbol decimates a tone to a
          per-symbol phase step of ``2 * pi * f / symbol_rate``: at exactly a
          quarter of the symbol rate that step is 90 degrees and the
          representatives form a perfect four-point constellation, reported
          as ``"QPSK"`` at confidence 1.0; at half the symbol rate it is 180
          degrees and they form a perfect antipodal pair, reported as
          ``"BPSK"`` at confidence 1.0. Every other tone frequency measured
          spreads the phase and scores 0 for both classes. This is inherent
          to one-sample-per-symbol decimation, not a property of the
          statistic.
        - **Confidence is a classification margin, not a probability and not
          an error rate.** Its floor for structureless input falls only as
          about ``1 / sqrt(symbol_count)``: over 300 complex-Gaussian blocks
          the largest margin was 0.82 at 16 symbols and 0.16 at 800, while a
          genuine 5 dB block can sit as low as 0.14. There is no single
          threshold that separates the two, and margins are not comparable
          across block lengths; use more symbols when that matters.
        - **Short blocks are unreliable.** 16 symbols is the numerical floor
          for the debiasing, not a reliability guarantee: over 500 seeds at
          16 symbols, 9 QPSK blocks read ``"BPSK"`` (worst margin 0.73) and 4
          were ``"AMBIGUOUS"``, because a short random draw genuinely can
          land mostly on one antipodal pair. Errors disappear by 64 symbols,
          which is the recommended minimum; BPSK was read correctly at every
          length tried.
        - **Residual CFO must stay small, and the floor is no protection
          against it.** Up to about 145 degrees of total carrier drift across
          the block every draw measured was still read correctly: 14 degrees
          costs a few hundredths of margin, and 72 degrees leaves QPSK at
          about 0.19 while BPSK still reads 0.70. Beyond that the behaviour is
          erratic rather than gracefully ambiguous, because a rotating
          constellation aliases onto the symbol grid -- at 180 degrees of
          drift no draw was named at all, while at 288 degrees BPSK was read
          correctly again and QPSK was not. **A wrong class can win**: over a
          sweep of 11 drifts beyond 2 cycles times 40 seeds, 57 QPSK blocks
          read ``"BPSK"`` at the default floor, the strongest at 0.11 (no
          block ever read ``"QPSK"`` wrongly). Margins that small are the only
          symptom, so raise ``min_confidence`` -- or estimate and correct the
          offset first -- rather than trusting the default on a block with
          unknown CFO.
        - Pulse-shaped (for example root-raised-cosine) signals violate the
          constant-within-a-symbol assumption and are out of scope.

    Args:
        samples: One-dimensional, finite, real or complex sample block. Real
            input is treated as a complex block with zero imaginary part, so
            a real antipodal stream reads as BPSK. Never modified.
        samples_per_symbol: Known integer number of samples per symbol,
            ``>= 1``. Must be exact; a wrong value samples across symbol
            boundaries and the result is meaningless. At least 16 complete
            symbol periods must follow ``boundary_offset``, and at least 64
            are recommended.
        boundary_offset: Known index of the first sample of the first
            complete symbol, in ``[0, samples_per_symbol)``. This is the
            convention :class:`~iqwav.estimation.SymbolRateEstimate` reports
            in its own ``boundary_offset`` field. Defaults to 0.
        min_confidence: Smallest margin at which a class is named. A block
            whose two class scores are closer than this is reported as
            ``"AMBIGUOUS"`` rather than decided, so a named class always
            satisfies ``result.confidence >= min_confidence``. Must lie in
            ``[0.0, 1.0)``. Defaults to 0.05, which admits every clean or
            AWGN-impaired block measured down to 5 dB SNR (worst margin
            0.14) while withholding a decision on the degenerate families
            above. Passing 0.0 names a class on any strictly positive
            margin, however weakly supported; a block with no constellation
            diversity at all still comes back ``"AMBIGUOUS"``, because its
            margin is exactly zero.

    Returns:
        A frozen :class:`ModulationEstimate`. Its ``modulation`` is
        ``"BPSK"``, ``"QPSK"``, or ``"AMBIGUOUS"``; the scores and
        ``confidence`` are populated in all three cases.

    Raises:
        ValueError: Only for a violated contract, never for an inconvenient
            measurement. If ``samples`` is not one-dimensional, is empty, or
            contains a non-finite value; if ``samples_per_symbol`` is not an
            integer ``>= 1``; if ``boundary_offset`` is not an integer in
            ``[0, samples_per_symbol)``; if ``min_confidence`` is not a real
            number in ``[0.0, 1.0)``; or if fewer than 16 complete symbol
            periods follow ``boundary_offset``. A block that simply does not
            favour either constellation -- including a constant or all-zero
            block -- returns ``"AMBIGUOUS"`` instead of raising.
    """
    values = _validate_samples(samples)
    period = _validate_integer(samples_per_symbol, name="samples_per_symbol", minimum=1)
    offset = _validate_integer(boundary_offset, name="boundary_offset", minimum=0)
    if offset >= period:
        raise ValueError(
            f"boundary_offset must be < samples_per_symbol={period}, "
            f"got {boundary_offset!r}."
        )
    floor = _validate_min_confidence(min_confidence)

    required = offset + _MIN_SYMBOLS * period
    if values.size < required:
        raise ValueError(
            f"samples must contain at least {_MIN_SYMBOLS} complete symbol "
            f"periods after boundary_offset={offset} at "
            f"samples_per_symbol={period} ({required} values; "
            f"{_RECOMMENDED_SYMBOLS} symbols or more are recommended), "
            f"got {values.size}."
        )

    symbols = _normalized_symbols(_representative_symbols(values, period, offset))

    diversity = _symmetry_coherence(symbols, _DIVERSITY_ORDER)
    antipodal = _symmetry_coherence(symbols, _BPSK_ORDER)
    quaternary = _symmetry_coherence(symbols, _QPSK_ORDER)

    # Each class must show its own symmetry and *not* the lower-order
    # symmetry that would make the observation degenerate: symbols piled onto
    # a single point are symmetric under every rotation and so are evidence
    # for neither class, and symbols that squaring collapses are an antipodal
    # pair rather than a quaternary set.
    bpsk_score = antipodal * (1.0 - diversity)
    qpsk_score = quaternary * (1.0 - antipodal)
    confidence = abs(bpsk_score - qpsk_score)

    if confidence <= 0.0 or confidence < floor:
        # Not an error: the block was measured successfully and simply does
        # not favour one observed constellation over the other. A zero margin
        # is the exact outcome for a block with no constellation diversity at
        # all, so that case is ambiguous even at min_confidence=0.0.
        modulation = _AMBIGUOUS
    else:
        modulation = "BPSK" if bpsk_score > qpsk_score else "QPSK"

    return ModulationEstimate(
        modulation=modulation,
        confidence=confidence,
        bpsk_score=bpsk_score,
        qpsk_score=qpsk_score,
        symbol_count=int(symbols.size),
        samples_per_symbol=period,
    )
