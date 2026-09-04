"""Controlled automatic parameter-estimation pipeline for BPSK/QPSK.

This module provides :func:`run_parameter_estimation_pipeline`, which
chains the already-implemented controlled estimation and synchronization
primitives into a fixed-order analysis pipeline for rectangular-pulse
BPSK or QPSK signals with an integer samples-per-symbol.

This extends Phase 2I (:mod:`iqwav.pipeline.receiver`), where the
modulation, CFO, symbol rate, and timing parameters were all supplied by
the caller. Here, everything except the sample rate and (optionally) a
known CFO is estimated from the signal itself using existing project
estimators.

Processing order
-----------------
1. Validate the input IQ samples and the caller-supplied sample rate.
2. Estimate the dominant frequency with
   :func:`iqwav.estimation.estimate_peak_frequency` (Phase 2A).
3. Estimate occupied bandwidth with
   :func:`iqwav.estimation.estimate_occupied_bandwidth` (Phase 2B).
4. Estimate noise floor / SNR with :func:`iqwav.estimation.estimate_snr`
   (Phase 2C), using the Phase 2B occupied band as the signal region and
   the remaining spectral edge as the noise region, when those regions
   are available.
5. Estimate the symbol rate and integer samples-per-symbol with
   :func:`iqwav.estimation.estimate_symbol_rate` (Phase 2E).
6. Recover the integer timing phase with
   :func:`iqwav.synchronization.recover_symbol_timing` (Phase 2H), using
   the Phase 2E samples-per-symbol.
7. Classify BPSK versus QPSK with
   :func:`iqwav.estimation.estimate_modulation` (Phase 2F), on the
   Phase 2H timing-recovered symbols.
8. If a known CFO was supplied, apply
   :func:`iqwav.dsp.correct_frequency_offset` (Phase 2G) before timing
   recovery, and demodulate the recovered symbols with the existing
   known-timing demodulator matching the Phase 2F verdict, but only when
   every required controlled parameter (samples-per-symbol, timing, and
   an unambiguous BPSK/QPSK verdict) is available.

This module performs no new estimation algorithm: every step below
directly reuses an existing, already-implemented primitive. Sample rate
is never estimated -- it is always the caller-supplied value, echoed
back unchanged. Any estimator that fails or returns an ambiguous result
short-circuits the later stages that depend on it rather than
substituting a guessed value; the result's ``status`` field and the
``None``-valued fields explain what was and was not available.

Scope
-----
This is a **thin orchestration layer** over already-implemented
primitives, not a new DSP algorithm, and not a blind receiver. It is
explicitly not:

- blind carrier recovery, a PLL, or a Costas loop (only the constant CFO
  the caller supplies, if any, is removed; nothing is tracked or
  recovered from the signal itself),
- an interpolating or fractional-sample timing recovery (Phase 2H's
  integer-phase search only),
- pulse-shaping support (rectangular, unshaped pulses only),
- FEC, deinterleaving, framing, or payload recovery (only raw
  demodulated bits, when available, are returned),
- GUI integration.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..demod import bpsk_demodulate, qpsk_demodulate
from ..dsp import correct_frequency_offset
from ..estimation import (
    ModulationEstimate,
    NoiseFloorEstimate,
    OccupiedBandwidthEstimate,
    PeakFrequencyEstimate,
    SNREstimate,
    SymbolRateEstimate,
    estimate_modulation,
    estimate_occupied_bandwidth,
    estimate_peak_frequency,
    estimate_snr,
    estimate_symbol_rate,
)
from ..synchronization import SymbolTimingRecovery, recover_symbol_timing

__all__ = ["ParameterEstimationResult", "run_parameter_estimation_pipeline"]

# Fraction of the occupied bandwidth's own width used as a guard gap
# before selecting a noise-reference region at the spectral edge, so the
# noise region does not abut the signal region's transition band.
_NOISE_GUARD_FRACTION = 0.5


@dataclass(frozen=True, eq=False)
class ParameterEstimationResult:
    """Outcome of running the controlled parameter-estimation pipeline.

    Every field is populated with the best available result from the
    corresponding stage, or left ``None`` when that stage could not run
    or its output was too ambiguous to build on. ``status`` and
    ``failure_reason`` summarize which case occurred; no field is ever a
    fabricated or guessed value.

    Attributes:
        status: One of ``"complete"`` (bits were recovered),
            ``"partial"`` (some estimates succeeded but bits were not
            recovered), or ``"failed"`` (an early, required stage
            raised). See ``failure_reason`` for detail in the latter two
            cases.
        failure_reason: Human-readable explanation of why ``status`` is
            not ``"complete"``, or ``None`` when it is.
        sample_rate: The caller-supplied sample rate in Hz, echoed back
            unchanged. This is never estimated.
        peak_frequency: Phase 2A dominant-frequency estimate, or ``None``
            if that stage failed.
        occupied_bandwidth: Phase 2B occupied-bandwidth estimate, or
            ``None`` if that stage failed.
        snr: Phase 2C SNR estimate, or ``None`` if the occupied-bandwidth
            estimate (needed to build signal/noise regions) was
            unavailable or if the SNR estimator itself failed.
        noise_floor: Phase 2C noise-floor estimate over the same
            noise-reference region as ``snr``, or ``None`` under the same
            conditions.
        symbol_rate: Phase 2E symbol-rate estimate, or ``None`` if that
            stage failed or returned below-threshold quality.
        samples_per_symbol: Integer samples-per-symbol from
            ``symbol_rate``, echoed here for convenience, or ``None`` if
            unavailable.
        timing: Phase 2H symbol-timing-recovery result, or ``None`` if
            ``samples_per_symbol`` was unavailable or the block was too
            short for the timing search.
        modulation: Phase 2F modulation estimate (BPSK, QPSK, or
            AMBIGUOUS), or ``None`` if ``timing`` was unavailable.
        frequency_offset_hz: The known CFO supplied by the caller and
            removed before timing recovery, if any; ``None`` if the
            caller did not supply one (no CFO correction was applied).
        cfo_corrected_samples: Samples after CFO correction, if a CFO was
            supplied; ``None`` otherwise. Same length as the input,
            complex128.
        bits: Recovered bits, or ``None`` unless ``samples_per_symbol``,
            ``timing``, and an unambiguous (BPSK or QPSK, not AMBIGUOUS)
            ``modulation`` verdict were all available, in which case
            demodulation is performed with the existing known-timing
            demodulator matching the verdict.
    """

    status: str
    failure_reason: str | None
    sample_rate: float
    peak_frequency: PeakFrequencyEstimate | None
    occupied_bandwidth: OccupiedBandwidthEstimate | None
    snr: SNREstimate | None
    noise_floor: NoiseFloorEstimate | None
    symbol_rate: SymbolRateEstimate | None
    samples_per_symbol: int | None
    timing: SymbolTimingRecovery | None
    modulation: ModulationEstimate | None
    frequency_offset_hz: float | None
    cfo_corrected_samples: npt.NDArray[np.complex128] | None
    bits: npt.NDArray[np.int64] | None


def _validate_samples(samples: object) -> npt.NDArray[np.complexfloating]:
    """Validate the input IQ sample block."""
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    if not np.iscomplexobj(values):
        raise ValueError("samples must be complex-valued IQ data.")
    return values


def _validate_sample_rate(sample_rate: object) -> float:
    """Validate the caller-supplied sample rate."""
    try:
        value = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sample_rate must be a real number, got {sample_rate!r}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"sample_rate must be positive and finite, got {sample_rate!r}."
        )
    return value


def _validate_frequency_offset_hz(value: object) -> float:
    """Validate an optional known CFO."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"frequency_offset_hz must be a real number, got {value!r}."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"frequency_offset_hz must be finite, got {value!r}."
        )
    return number


def _snr_regions(
    bandwidth: OccupiedBandwidthEstimate, sample_rate: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Build disjoint (signal, noise) regions from a Phase 2B estimate.

    The signal region is the occupied band itself. The noise region is a
    guard-separated slice of unused spectrum between the occupied band's
    upper edge and Nyquist. Returns ``None`` if no such noise region
    exists (the occupied band already reaches the edge of the spectrum
    with no room for a separated reference slice).
    """
    nyquist = sample_rate / 2.0
    lower = bandwidth.lower_frequency_hz
    upper = bandwidth.upper_frequency_hz
    width = max(upper - lower, 0.0)
    guard = _NOISE_GUARD_FRACTION * width if width > 0.0 else 0.0

    noise_lower = upper + guard
    noise_upper = nyquist
    if noise_lower < noise_upper:
        return (lower, upper), (noise_lower, noise_upper)

    # Not enough room above the band; try the lower edge instead, using
    # the appropriate floor for complex (signed) vs. real (non-negative)
    # spectra.
    floor = -nyquist
    noise_upper = lower - guard
    if floor < noise_upper:
        return (lower, upper), (floor, noise_upper)
    return None


def run_parameter_estimation_pipeline(
    samples: npt.ArrayLike,
    sample_rate: float,
    *,
    frequency_offset_hz: float | None = None,
) -> ParameterEstimationResult:
    """Run the controlled automatic parameter-estimation pipeline.

    Chains, in this fixed order, the existing Phase 2A/2B/2C/2E/2H/2F
    estimators over a controlled rectangular-pulse BPSK or QPSK block,
    and demodulates with the existing known-timing demodulators only when
    every parameter that stage needs was itself successfully and
    unambiguously determined. No parameter is guessed: an estimator
    failure or an ambiguous verdict is represented explicitly by leaving
    the dependent result fields ``None`` and setting ``status`` /
    ``failure_reason`` accordingly, rather than substituting a default.

    Args:
        samples: 1-D complex IQ sample block. Must be non-empty, finite,
            and complex-valued. Read-only: never modified.
        sample_rate: Known sampling frequency in Hz, supplied by the
            caller. Must be positive and finite. This value is echoed
            back unchanged; it is never estimated by this pipeline.
        frequency_offset_hz: An optional known, constant CFO in Hz to
            remove before timing recovery and demodulation, via
            :func:`iqwav.dsp.correct_frequency_offset`. If ``None`` (the
            default), no CFO correction is applied and the pipeline
            assumes the block is already at baseband; this is a
            controlled parameter, never blindly estimated.

    Returns:
        A :class:`ParameterEstimationResult` with every stage's outcome,
        or ``None`` for stages that could not run or were ambiguous.

    Raises:
        ValueError: If ``samples`` is not a 1-D, non-empty, finite,
            complex array, if ``sample_rate`` is not positive and finite,
            or if ``frequency_offset_hz`` is supplied but not finite.
    """
    values = _validate_samples(samples)
    sample_rate = _validate_sample_rate(sample_rate)
    if frequency_offset_hz is not None:
        frequency_offset_hz = _validate_frequency_offset_hz(frequency_offset_hz)

    peak_frequency: PeakFrequencyEstimate | None = None
    occupied_bandwidth: OccupiedBandwidthEstimate | None = None
    snr: SNREstimate | None = None
    noise_floor: NoiseFloorEstimate | None = None
    symbol_rate: SymbolRateEstimate | None = None
    samples_per_symbol: int | None = None
    timing: SymbolTimingRecovery | None = None
    modulation: ModulationEstimate | None = None
    cfo_corrected_samples: npt.NDArray[np.complex128] | None = None
    bits: npt.NDArray[np.int64] | None = None
    failure_reason: str | None = None

    # Step 2: dominant frequency (Phase 2A).
    try:
        peak_frequency = estimate_peak_frequency(values, sample_rate)
    except ValueError as exc:
        failure_reason = f"peak frequency estimation failed: {exc}"

    # Step 3: occupied bandwidth (Phase 2B).
    try:
        occupied_bandwidth = estimate_occupied_bandwidth(values, sample_rate)
    except ValueError as exc:
        if failure_reason is None:
            failure_reason = f"occupied bandwidth estimation failed: {exc}"

    # Step 4: noise floor / SNR (Phase 2C), only if regions are available.
    if occupied_bandwidth is not None:
        regions = _snr_regions(occupied_bandwidth, sample_rate)
        if regions is not None:
            signal_region, noise_region = regions
            try:
                snr = estimate_snr(values, sample_rate, signal_region, noise_region)
                noise_floor_val = snr.noise_power_density
                noise_floor = NoiseFloorEstimate(
                    noise_power_density=noise_floor_val,
                    noise_floor_db=10.0 * math.log10(noise_floor_val),
                    noise_power=snr.estimated_noise_power,
                    noise_bandwidth_hz=snr.noise_bandwidth_hz,
                    lower_frequency_hz=snr.noise_lower_frequency_hz,
                    upper_frequency_hz=snr.noise_upper_frequency_hz,
                    resolution_hz=snr.resolution_hz,
                )
            except ValueError:
                # SNR is not a required parameter for later stages; leave
                # it unavailable rather than fabricating a value.
                pass

    # Step 5: symbol rate / samples-per-symbol (Phase 2E).
    try:
        symbol_rate = estimate_symbol_rate(values, sample_rate)
        samples_per_symbol = symbol_rate.samples_per_symbol
    except ValueError as exc:
        if failure_reason is None:
            failure_reason = f"symbol rate estimation failed: {exc}"

    # Step 1 (CFO correction ahead of timing, if a known CFO was given).
    working_samples = values
    if frequency_offset_hz is not None:
        cfo_corrected_samples = correct_frequency_offset(
            values, sample_rate, frequency_offset_hz
        )
        working_samples = cfo_corrected_samples

    # Step 6: integer timing-phase recovery (Phase 2H).
    if samples_per_symbol is not None:
        try:
            timing = recover_symbol_timing(working_samples, samples_per_symbol)
        except ValueError as exc:
            if failure_reason is None:
                failure_reason = f"timing recovery failed: {exc}"
    elif failure_reason is None:
        failure_reason = "symbol rate unavailable; timing recovery skipped."

    # Step 7: BPSK vs. QPSK classification (Phase 2F).
    if timing is not None and samples_per_symbol is not None:
        try:
            modulation = estimate_modulation(timing.symbols, 1)
        except ValueError as exc:
            if failure_reason is None:
                failure_reason = f"modulation classification failed: {exc}"

    # Step 8: demodulate only when every controlled parameter is known
    # and the modulation verdict is unambiguous.
    if (
        timing is not None
        and modulation is not None
        and modulation.modulation in ("BPSK", "QPSK")
    ):
        if modulation.modulation == "BPSK":
            bits = bpsk_demodulate(timing.symbols, 1)
        else:
            bits = qpsk_demodulate(timing.symbols, 1)

    if bits is not None:
        status = "complete"
        failure_reason = None
    elif failure_reason is not None:
        status = "failed"
    elif modulation is not None and modulation.modulation == "AMBIGUOUS":
        status = "partial"
        failure_reason = (
            "modulation classification was ambiguous "
            f"(confidence {modulation.confidence:.4f} below threshold); "
            "bits were not recovered."
        )
    else:
        status = "partial"
        failure_reason = (
            "one or more required stages did not produce a usable result; "
            "bits were not recovered."
        )

    return ParameterEstimationResult(
        status=status,
        failure_reason=failure_reason,
        sample_rate=sample_rate,
        peak_frequency=peak_frequency,
        occupied_bandwidth=occupied_bandwidth,
        snr=snr,
        noise_floor=noise_floor,
        symbol_rate=symbol_rate,
        samples_per_symbol=samples_per_symbol,
        timing=timing,
        modulation=modulation,
        frequency_offset_hz=frequency_offset_hz,
        cfo_corrected_samples=cfo_corrected_samples,
        bits=bits,
    )