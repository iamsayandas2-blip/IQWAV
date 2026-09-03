"""Bounded symbol-rate estimation for controlled rectangular-pulse signals.

This module provides a single, narrowly-scoped estimator:
:func:`estimate_symbol_rate`, which recovers the symbol (baud) rate of a
**controlled, rectangular-pulse, symbol-synchronous** sampled waveform of
the kind produced by :mod:`iqwav.modulation`
(:func:`iqwav.modulation.bpsk_waveform`,
:func:`iqwav.modulation.qpsk_waveform`, and the underlying
:func:`iqwav.modulation.symbols_to_samples`), given a known sample rate.

Method
------
Those generators hold each symbol constant for exactly
``samples_per_symbol`` samples (``numpy.repeat``), so the first difference
of the waveform is exactly zero *inside* a symbol and non-zero only where
consecutive symbols differ. The transition profile
``d[n] = abs(x[n] - x[n - 1])`` is therefore an impulse train whose
impulses all land on a single residue class of the sample index modulo the
symbol period.

The estimator exploits exactly that structure. For every candidate integer
period ``P`` in a bounded search range it measures the fraction of the
total ``d`` mass that falls on the best single residue class modulo ``P``,
corrects that fraction for the ``1 / P`` level expected by chance, and
returns the **largest** candidate period that is nearly as well supported
as the best one: integer sub-multiples of the true period concentrate the
same impulses just as perfectly (they tie), while multiples of it split
the impulses across several classes and score at most ``1 / m`` of the
best. The largest well-supported period is thus the symbol period, and
the symbol rate is ``sample_rate / P``.

Scope
-----
This is a **bounded, controlled-signal symbol-rate primitive**, not a
blind receiver component. It is explicitly **not**:

- a blind or general-purpose RF baud-rate estimator (it assumes
  rectangular, unshaped pulses and a constant integer number of samples
  per symbol; pulse-shaped signals such as root-raised-cosine are out of
  scope),
- symbol-timing recovery or a timing loop (a single boundary phase is
  reported for the analyzed block; nothing is tracked, interpolated, or
  corrected),
- carrier recovery, CFO estimation or CFO correction (see Phase 2A/2D
  :func:`iqwav.estimation.estimate_peak_frequency` and
  :func:`iqwav.estimation.estimate_frequency_offset` for the frequency-side
  primitives),
- modulation recognition (nothing is inferred about the constellation;
  BPSK, QPSK and any other constant-period symbol stream are treated
  identically),
- framing, FEC, de-interleaving, payload recovery, or activity detection.

The input samples are never modified, filtered, resampled, or corrected;
the measurement is the entire output.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["SymbolRateEstimate", "estimate_symbol_rate"]

# _MIN_PERIODS periods of the shortest supported symbol period (2 samples),
# plus the one sample consumed by the first difference.
_MIN_SAMPLES = 9
# Minimum number of candidate symbol periods that must fit in the analyzed
# block for a periodicity claim to be made at all.
_MIN_PERIODS = 4
# A period supported by essentially a single observed transition is not
# identifiable: every candidate period explains one lone impulse equally
# well. Expressed as a participation ratio, see ``effective_transitions``.
_MIN_EFFECTIVE_TRANSITIONS = 2.0
# Reject a candidate when a proper divisor explains it substantially better.
_DIVISOR_DOMINANCE_MARGIN = 0.05


@dataclass(frozen=True)
class SymbolRateEstimate:
    """Result of a single controlled-signal symbol-rate estimate.

    Attributes:
        symbol_rate_hz: Estimated symbol rate in symbols per second,
            equal to ``sample_rate_hz / samples_per_symbol`` exactly.
        samples_per_symbol: Estimated symbol period in samples. Always an
            integer >= 2: the analysis grid of this estimator is the
            integer-period grid, which is exactly the grid the IQWAV
            rectangular-pulse waveform generators produce.
        sample_rate_hz: The caller-supplied known sample rate in Hz,
            echoed back for traceability.
        symbol_rate_resolution_hz: Spacing between the returned rate and
            the next *lower* achievable rate on the integer-period grid,
            ``sample_rate / P - sample_rate / (P + 1)``. The next higher
            achievable rate, ``sample_rate / (P - 1)``, is farther away.
            This bounds the estimator's quantization, not its statistical
            error.
        quality: Chance-corrected concentration of the transition profile
            on the selected boundary phase, ``(concentration - 1 / P) /
            (1 - 1 / P)``, at most 1.0. It is the fraction of total
            first-difference magnitude attributable to boundary-aligned
            symbol transitions rather than to intra-symbol variation
            (noise, residual carrier rotation, pulse shaping). 1.0 means a
            noiseless rectangular-pulse waveform; values near 0 mean no
            periodic transition structure was found and the rate must not
            be trusted. May be negative for structureless input.
        concentration: Raw, uncorrected fraction of total
            first-difference magnitude falling on the selected boundary
            phase, in ``[0, 1]``. For random input this tends to
            ``1 / samples_per_symbol`` rather than 0, which is why
            ``quality`` is the chance-corrected figure.
        boundary_offset: Estimated symbol-boundary phase: symbols start at
            sample indices congruent to this value modulo
            ``samples_per_symbol``. 0 for the canonical
            :func:`iqwav.modulation.symbols_to_samples` output. This is a
            single block-level phase measurement, not timing recovery.
        symbol_count: Number of complete symbol periods the block spans
            at the estimated period and phase,
            ``(len(samples) - boundary_offset) // samples_per_symbol``.
        effective_transitions: Participation ratio ``(sum d) ** 2 /
            sum(d ** 2)`` of the boundary-aligned transition magnitudes,
            i.e. a threshold-free effective count of how many symbol
            boundaries carried observable transition energy. It guards
            against the degenerate case of a block containing a single
            transition, where every candidate period fits equally well.
            For noisy input it grows toward the number of candidate
            boundaries because noise contributes at every boundary, so it
            is a degeneracy guard, not a signal-quality metric.
        searched_samples_per_symbol: The ``(min, max)`` candidate symbol
            periods actually searched, after the requested maximum was
            capped to what the block length can support.
    """

    symbol_rate_hz: float
    samples_per_symbol: int
    sample_rate_hz: float
    symbol_rate_resolution_hz: float
    quality: float
    concentration: float
    boundary_offset: int
    symbol_count: int
    effective_transitions: float
    searched_samples_per_symbol: tuple[int, int]


def _validate_samples(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Validate the block and sample rate, returning samples as an array."""
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(
            f"sample_rate must be positive and finite, got {sample_rate!r}."
        )
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError("samples must contain at least one value.")
    if values.size < _MIN_SAMPLES:
        raise ValueError(
            f"samples must contain at least {_MIN_SAMPLES} values to observe "
            f"{_MIN_PERIODS} periods of the shortest supported symbol period, "
            f"got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    return values


def _validate_integer_period(value: object, *, name: str, minimum: int) -> int:
    """Validate one integer candidate-period bound."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return int(value)


def _validate_unit_fraction(
    value: object, *, name: str, lower: float, upper: float,
    include_lower: bool, include_upper: bool,
) -> float:
    """Validate one finite float estimator parameter within a range."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    low_ok = number >= lower if include_lower else number > lower
    high_ok = number <= upper if include_upper else number < upper
    if not (low_ok and high_ok):
        left = "[" if include_lower else "("
        right = "]" if include_upper else ")"
        raise ValueError(
            f"{name} must lie within {left}{lower}, {upper}{right}, "
            f"got {value!r}."
        )
    return number


def _transition_profile(
    values: np.ndarray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], float]:
    """Return the transition magnitudes, their sample indices, and the total.

    ``magnitudes[i] = abs(values[i + 1] - values[i])`` is indexed by the
    sample that *starts* the potential new symbol, ``indices[i] = i + 1``,
    so a symbol boundary at sample ``n`` contributes to residue class
    ``n % P``. The input array is only read; the returned arrays are new.
    """
    magnitudes = np.abs(np.diff(values)).astype(np.float64)
    indices = np.arange(1, values.size, dtype=np.int64)
    return magnitudes, indices, float(np.sum(magnitudes))


def _score_period(
    magnitudes: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    total: float,
    period: int,
) -> tuple[float, float, int, float]:
    """Score one candidate symbol period.

    Bins the transition magnitudes by sample index modulo ``period`` and
    measures the best single bin. Returns ``(quality, concentration,
    boundary_offset, effective_transitions)`` as documented on
    :class:`SymbolRateEstimate`.
    """
    classes = indices % period
    binned = np.bincount(classes, weights=magnitudes, minlength=period)
    offset = int(np.argmax(binned))
    # A fraction of a non-negative total cannot exceed 1; clamp against
    # summation round-off so quality stays inside its documented range.
    concentration = min(float(binned[offset]) / total, 1.0)
    chance = 1.0 / period
    quality = (concentration - chance) / (1.0 - chance)
    aligned = magnitudes[classes == offset]
    energy = float(np.sum(aligned * aligned))
    effective = float(binned[offset]) ** 2 / energy if energy > 0.0 else 0.0
    return quality, concentration, offset, effective


def estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    *,
    min_samples_per_symbol: int = 2,
    max_samples_per_symbol: int = 64,
    quality_ratio: float = 0.75,
    min_quality: float = 0.02,
) -> SymbolRateEstimate:
    """Estimate the symbol rate of a controlled rectangular-pulse waveform.

    Definition:
        The symbol period is estimated as the largest integer number of
        samples ``P`` in the searched range on which the waveform's
        first-difference magnitudes concentrate essentially maximally, and
        the reported rate is::

            symbol_rate_hz = sample_rate / P

        Concretely, with ``d[n] = abs(x[n] - x[n - 1])`` for
        ``n = 1 .. N - 1``:

        1. For each candidate ``P``, ``d`` is binned by ``n % P`` and the
           largest bin's share of the total is taken as
           ``concentration``.
        2. ``quality = (concentration - 1 / P) / (1 - 1 / P)`` corrects
           that share for the ``1 / P`` level a structureless signal
           reaches by chance, so candidates of different ``P`` are
           directly comparable.
        3. Every candidate whose ``quality`` is at least
           ``quality_ratio`` times the best ``quality`` observed is
           considered equally well supported, and the **largest** such
           candidate is returned.

        Step 3 is what makes the answer the symbol period rather than one
        of its sub-multiples: for a rectangular-pulse waveform every
        integer divisor of the true period concentrates the same
        transitions perfectly and therefore ties at the same quality,
        while an ``m``-fold multiple splits them across ``m`` classes and
        reaches only about ``1 / m`` of it.

    Assumptions (all satisfied by the IQWAV waveform generators):
        - Rectangular, unshaped pulses: the waveform is piecewise constant
          over each symbol, as produced by
          :func:`iqwav.modulation.symbols_to_samples` and therefore by
          :func:`iqwav.modulation.bpsk_waveform` and
          :func:`iqwav.modulation.qpsk_waveform`.
        - A single constant symbol period throughout the block, equal to
          an integer number of samples of at least 2. This is the only
          model the IQWAV generators support (``samples_per_symbol`` is
          validated there as an integer >= 1), so the API is deliberately
          constrained to integer periods; fractional or drifting
          samples-per-symbol is out of scope and is not estimated.
        - The known ``sample_rate`` corresponds to those samples.
        - The symbol sequence actually changes at a representative subset
          of symbol boundaries (pseudo-random data does). Boundaries where
          consecutive symbols are equal produce no transition and are
          invisible to any method of this kind.
        - The true symbol period lies inside the searched range
          ``[min_samples_per_symbol, max_samples_per_symbol]``.

    Limitations:
        - Not blind and not general: pulse-shaped (for example
          root-raised-cosine) signals, multipath, timing jitter,
          fractional resampling and multi-rate captures violate the
          piecewise-constant model, and no claim is made for them.
        - If the true period exceeds the searched maximum, the returned
          period is one of its divisors inside the range rather than an
          error: divisors are genuinely consistent with the observed
          transitions. Bound the search with knowledge of the signal.
        - Data-dependent ambiguity: if the symbol stream only ever changes
          every ``m``-th boundary, the observable period really is
          ``m * P`` and that is what is reported.
        - A residual carrier offset makes the waveform non-constant within
          a symbol and lowers ``quality``; small offsets do not move the
          estimate, large ones eventually destroy the structure. No
          carrier or timing correction is performed here.
        - One symbol per ``P`` samples is the finest structure resolvable:
          ``samples_per_symbol = 1`` (symbol rate equal to the sample
          rate) has no intra-symbol structure at all and is rejected by
          the search range.
        - A single FFT-free time-domain block is analyzed; no averaging
          across blocks and no confidence interval is produced.
          ``quality`` is a descriptive statistic of this block only.

    Signal is never modified:
        ``samples`` is read only. No filtering, resampling, timing
        correction, or normalization is applied and no waveform is
        returned.

    Args:
        samples: 1-D real or complex sampled waveform with at least 9
            finite values, and not constant (a constant or all-zero block
            has no transitions and raises ``ValueError``). Complex
            baseband IQ, as produced by the BPSK/QPSK waveform helpers, is
            the intended input; real-valued rectangular waveforms work
            identically.
        sample_rate: Sampling frequency in Hz. Must be positive and
            finite.
        min_samples_per_symbol: Smallest candidate symbol period in
            samples. Must be an integer >= 2. Defaults to 2.
        max_samples_per_symbol: Largest candidate symbol period in
            samples. Must be an integer >= ``min_samples_per_symbol``.
            Defaults to 64. It is capped to ``(N - 1) // 4`` so that at
            least four candidate periods fit in the block; the range
            actually searched is reported in the result.
        quality_ratio: How close to the best observed quality a longer
            candidate period must come to be preferred over it, in
            ``(0, 1]``. Defaults to 0.75, which accepts the exact ties
            produced by sub-multiples of the true period while rejecting
            multiples of it (they reach at most about half). Lower values
            make the estimator prefer multiples of the true period;
            1.0 demands an exact tie.
        min_quality: Smallest returned ``quality`` accepted, in
            ``[0, 1)``; ``result.quality >= min_quality`` therefore always
            holds. Defaults to 0.02: blocks below that show no usable
            periodic transition structure (for reference, pure Gaussian
            noise reaches roughly ``0.55 / sqrt(N)``, and rectangular
            BPSK/QPSK at 0 dB SNR with 16 samples per symbol reaches about
            0.03). Because the noise level of ``quality`` depends on the
            block length, a short structureless block can still pass a
            fixed threshold; pass 0.0 to disable the check and inspect
            ``quality`` directly instead.

    Returns:
        A :class:`SymbolRateEstimate` with the estimated rate, the integer
        symbol period, the echoed sample rate, the integer-grid
        resolution, the quality and raw concentration statistics, the
        symbol-boundary phase, the number of complete symbol periods, the
        effective transition count, and the searched period range.

    Raises:
        ValueError: If ``sample_rate`` is not positive and finite; if
            ``samples`` is not a one-dimensional array of at least 9
            finite values; if the estimator parameters are not integers or
            floats in their documented ranges; if the block is too short
            for the requested candidate range; if the block is constant or
            all zero; if no candidate period concentrates transitions above
            chance; if the estimated period's ``quality`` is below
            ``min_quality``; or if the block contains effectively a single
            transition, which leaves the period unidentifiable.
    """
    values = _validate_samples(samples, sample_rate)
    period_min = _validate_integer_period(
        min_samples_per_symbol, name="min_samples_per_symbol", minimum=2
    )
    period_max = _validate_integer_period(
        max_samples_per_symbol,
        name="max_samples_per_symbol",
        minimum=period_min,
    )
    ratio = _validate_unit_fraction(
        quality_ratio,
        name="quality_ratio",
        lower=0.0,
        upper=1.0,
        include_lower=False,
        include_upper=True,
    )
    quality_floor = _validate_unit_fraction(
        min_quality,
        name="min_quality",
        lower=0.0,
        upper=1.0,
        include_lower=True,
        include_upper=False,
    )

    supported_max = (values.size - 1) // _MIN_PERIODS
    searched_max = min(period_max, supported_max)
    if searched_max < period_min:
        raise ValueError(
            f"samples contains {values.size} values, which supports candidate "
            f"symbol periods of at most {supported_max} sample(s) while "
            f"observing {_MIN_PERIODS} periods, but min_samples_per_symbol is "
            f"{period_min}; supply a longer block or lower "
            "min_samples_per_symbol."
        )

    magnitudes, indices, total = _transition_profile(values)
    if total <= 0.0:
        raise ValueError(
            "samples never change value (constant or all-zero block), so the "
            "block contains no symbol transitions and no symbol rate can be "
            "estimated."
        )

    scores = [
        _score_period(magnitudes, indices, total, period)
        for period in range(period_min, searched_max + 1)
    ]
    qualities = [score[0] for score in scores]
    best_quality = max(qualities)
    if best_quality <= 0.0:
        raise ValueError(
            "no candidate symbol period concentrates transitions above the "
            "level expected by chance; the block shows no rectangular-pulse "
            "symbol structure in the searched range "
            f"[{period_min}, {searched_max}] samples per symbol."
        )

    threshold = ratio * best_quality

    # Finite transition sequences can make a multiple of the true period look
    # spuriously strong. Reject such a candidate when a proper divisor explains
    # the transitions substantially better. Nearly tied divisors remain
    # eligible because those cases are genuinely ambiguous.
    eligible = []
    for index, quality in enumerate(qualities):
        period_candidate = period_min + index
        if quality < threshold:
            continue
        dominated = any(
            period_candidate % divisor == 0
            and divisor >= period_min
            and qualities[divisor - period_min]
            > quality + _DIVISOR_DOMINANCE_MARGIN
            for divisor in range(period_min, period_candidate)
        )
        if not dominated:
            eligible.append(index)

    selected = max(eligible)
    quality, concentration, offset, effective = scores[selected]
    period = period_min + selected
    if quality < quality_floor:
        raise ValueError(
            f"the estimated symbol period of {period} sample(s) reaches "
            f"quality {quality:.4g}, below min_quality={quality_floor!r}; the "
            "block shows no usable periodic symbol-transition structure in "
            f"[{period_min}, {searched_max}] samples per symbol."
        )
    if effective < _MIN_EFFECTIVE_TRANSITIONS:
        raise ValueError(
            f"the selected symbol period of {period} sample(s) is supported by "
            f"only {effective:.3g} effective transition(s); a block with a "
            "single observable transition is explained equally well by every "
            "candidate period, so the symbol period is not identifiable."
        )

    rate = float(sample_rate) / period
    return SymbolRateEstimate(
        symbol_rate_hz=rate,
        samples_per_symbol=period,
        sample_rate_hz=float(sample_rate),
        symbol_rate_resolution_hz=float(sample_rate) / (period * (period + 1)),
        quality=quality,
        concentration=concentration,
        boundary_offset=offset,
        symbol_count=(values.size - offset) // period,
        effective_transitions=effective,
        searched_samples_per_symbol=(period_min, searched_max),
    )
