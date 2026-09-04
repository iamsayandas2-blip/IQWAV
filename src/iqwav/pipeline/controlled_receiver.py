"""Controlled receiver workflow integrating Phases 2I and 2J.

This module provides :func:`run_controlled_receiver_pipeline`, a single
higher-level workflow that combines the controlled end-to-end receiver
pipeline (:mod:`iqwav.pipeline.receiver`, Phase 2I) with the controlled
parameter-estimation pipeline (:mod:`iqwav.pipeline.analysis`, Phase 2J).

It supports controlled rectangular BPSK/QPSK signals with:

- a known sample rate (always supplied);
- a known integer samples-per-symbol, when the caller has it -- otherwise
  it is estimated by Phase 2E, exactly as in Phase 2J;
- a known modulation, when the caller has it -- otherwise it is
  classified by Phase 2F, exactly as in Phase 2J;
- an optional known constant CFO, corrected by Phase 2G if supplied;
- a controlled integer timing offset, always recovered by Phase 2H.

No new DSP or estimation algorithm is implemented here. This module is a
**thin orchestration layer** that reuses the existing Phase
2A/2B/2C/2E/2F/2G/2H primitives (via Phase 2J's estimation stages) and
the existing known-timing demodulators (via Phase 2I's demodulation
step), and reports one unified, structured result describing every
stage's outcome.

Scope
-----
This module does not add:

- blind CFO estimation, blind timing recovery, or blind modulation
  recognition (samples-per-symbol and modulation are either supplied by
  the caller or estimated by the already-implemented Phase 2E/2F
  primitives; timing is recovered by the already-implemented, bounded
  Phase 2H integer-phase search);
- pulse-shape estimation or carrier-recovery loops;
- FEC, deinterleaving, framing, or payload recovery;
- GUI integration.
"""

import math
from dataclasses import dataclass, field

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
from .analysis import _snr_regions

__all__ = ["ControlledReceiverResult", "run_controlled_receiver_pipeline"]

_SUPPORTED_MODULATIONS = ("bpsk", "qpsk")

# Stage names used as keys in ``stage_status`` and ``failure_reasons``.
_STAGES = (
    "peak_frequency",
    "occupied_bandwidth",
    "snr",
    "symbol_rate",
    "cfo_correction",
    "timing_recovery",
    "modulation_classification",
    "demodulation",
)


@dataclass(frozen=True, eq=False)
class ControlledReceiverResult:
    """Outcome of running the controlled receiver workflow.

    Every estimation field is populated with the best available result
    from the corresponding stage, or ``None`` when that stage did not
    run, was skipped, or failed. ``stage_status`` and ``failure_reasons``
    give a per-stage account of what happened; no field is ever a
    fabricated or guessed value.

    Attributes:
        status: One of ``"complete"`` (bits were recovered), ``"partial"``
            (some stages succeeded but bits were not recovered), or
            ``"failed"`` (a required estimation stage raised and no
            usable samples-per-symbol was available).
        failure_reason: Human-readable top-level explanation of why
            ``status`` is not ``"complete"``, or ``None`` when it is.
        sample_rate: The caller-supplied sample rate in Hz, echoed back
            unchanged. This is never estimated.
        peak_frequency: Phase 2A dominant-frequency estimate, or ``None``
            if that stage failed.
        occupied_bandwidth: Phase 2B occupied-bandwidth estimate, or
            ``None`` if that stage failed.
        snr: Phase 2C SNR estimate, or ``None`` if unavailable.
        noise_floor: Phase 2C noise-floor estimate over the same region
            as ``snr``, or ``None`` under the same conditions.
        symbol_rate: Phase 2E symbol-rate estimate, or ``None`` if that
            stage failed. Computed regardless of whether
            ``samples_per_symbol`` was supplied by the caller, purely as
            reported metadata.
        samples_per_symbol: The effective samples-per-symbol used by the
            pipeline: the caller-supplied value when given, otherwise the
            Phase 2E estimate, otherwise ``None``.
        samples_per_symbol_known: ``True`` if ``samples_per_symbol`` was
            supplied by the caller rather than estimated.
        timing: Phase 2H symbol-timing-recovery result, or ``None`` if
            ``samples_per_symbol`` was unavailable or the search failed.
        timing_offset: The integer symbol-boundary phase selected by
            Phase 2H, or ``None`` if timing recovery did not run.
        modulation_estimate: Phase 2F modulation estimate (BPSK, QPSK, or
            AMBIGUOUS) computed from the recovered symbols, or ``None``
            if timing recovery did not run or succeed. Computed
            regardless of whether ``modulation`` was supplied by the
            caller, purely as reported metadata.
        modulation: The effective modulation used for demodulation,
            ``"bpsk"`` or ``"qpsk"``: the caller-supplied value when
            given, otherwise the Phase 2F verdict when unambiguous,
            otherwise ``None``.
        modulation_known: ``True`` if ``modulation`` was supplied by the
            caller rather than estimated.
        frequency_offset_hz: The known CFO supplied by the caller, if
            any; ``None`` if the caller did not supply one.
        cfo_corrected_samples: Samples after CFO correction, if a CFO was
            supplied; ``None`` otherwise. Same length as the input,
            complex128.
        bits: Recovered bits, or ``None`` unless ``samples_per_symbol``,
            timing recovery, and an unambiguous ``modulation`` were all
            available.
        stage_status: Mapping from stage name to one of ``"success"``,
            ``"failed"``, or ``"skipped"`` for each of: ``peak_frequency``,
            ``occupied_bandwidth``, ``snr``, ``symbol_rate``,
            ``cfo_correction``, ``timing_recovery``,
            ``modulation_classification``, ``demodulation``.
        failure_reasons: Mapping from stage name to a human-readable
            reason, present only for stages whose status is ``"failed"``.
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
    samples_per_symbol_known: bool
    timing: SymbolTimingRecovery | None
    timing_offset: int | None
    modulation_estimate: ModulationEstimate | None
    modulation: str | None
    modulation_known: bool
    frequency_offset_hz: float | None
    cfo_corrected_samples: npt.NDArray[np.complex128] | None
    bits: npt.NDArray[np.int64] | None
    stage_status: dict = field(default_factory=dict)
    failure_reasons: dict = field(default_factory=dict)


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


def _validate_samples_per_symbol(samples_per_symbol: object) -> int:
    """Validate an optional known samples_per_symbol as an integer >= 1."""
    if isinstance(samples_per_symbol, bool) or not isinstance(
        samples_per_symbol, (int, np.integer)
    ):
        raise ValueError(
            f"samples_per_symbol must be an integer, got {samples_per_symbol!r}."
        )
    if int(samples_per_symbol) < 1:
        raise ValueError(
            f"samples_per_symbol must be >= 1, got {samples_per_symbol!r}."
        )
    return int(samples_per_symbol)


def _validate_modulation(modulation: object) -> str:
    """Validate an optional known modulation type."""
    if not isinstance(modulation, str) or modulation not in _SUPPORTED_MODULATIONS:
        raise ValueError(
            "modulation must be one of "
            f"{_SUPPORTED_MODULATIONS!r}, got {modulation!r}."
        )
    return modulation


def _validate_frequency_offset_hz(value: object) -> float:
    """Validate an optional known CFO."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"frequency_offset_hz must be a real number, got {value!r}."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"frequency_offset_hz must be finite, got {value!r}.")
    return number


def run_controlled_receiver_pipeline(
    samples: npt.ArrayLike,
    sample_rate: float,
    *,
    samples_per_symbol: int | None = None,
    modulation: str | None = None,
    frequency_offset_hz: float | None = None,
) -> ControlledReceiverResult:
    """Run the integrated controlled receiver workflow.

    Combines the Phase 2I controlled receiver chain with the Phase 2J
    parameter-estimation chain into one workflow: any of
    ``samples_per_symbol``, ``modulation``, and ``frequency_offset_hz``
    may be supplied as known controlled parameters, in which case they
    are used directly (as in Phase 2I); any left unsupplied are estimated
    from the signal using the existing Phase 2E/2F estimators (as in
    Phase 2J). Diagnostic spectral metadata (Phase 2A/2B/2C) is always
    computed. Timing phase is always recovered by the existing, bounded
    Phase 2H search. No parameter is ever guessed: a failed or skipped
    stage is reported explicitly via ``stage_status`` /
    ``failure_reasons`` rather than substituted with a default.

    Args:
        samples: 1-D complex IQ sample block. Must be non-empty, finite,
            and complex-valued. Read-only: never modified.
        sample_rate: Known sampling frequency in Hz. Must be positive and
            finite. Always echoed back unchanged; never estimated.
        samples_per_symbol: Known integer samples per symbol, >= 1, if
            available. If ``None`` (the default), it is estimated via
            Phase 2E.
        modulation: Known modulation, ``"bpsk"`` or ``"qpsk"``, if
            available. If ``None`` (the default), it is classified via
            Phase 2F, and used for demodulation only when unambiguous.
        frequency_offset_hz: Known constant CFO in Hz to remove before
            timing recovery, if any. If ``None`` (the default), no CFO
            correction is applied.

    Returns:
        A :class:`ControlledReceiverResult` with every stage's outcome.
        The input array is not modified.

    Raises:
        ValueError: If ``samples`` is not a 1-D, non-empty, finite,
            complex array; if ``sample_rate`` is not positive and finite;
            if ``samples_per_symbol`` is supplied but is not an integer
            >= 1; if ``modulation`` is supplied but is not ``"bpsk"`` or
            ``"qpsk"``; or if ``frequency_offset_hz`` is supplied but not
            finite.
    """
    values = _validate_samples(samples)
    sample_rate = _validate_sample_rate(sample_rate)
    sps_known = samples_per_symbol is not None
    if sps_known:
        samples_per_symbol = _validate_samples_per_symbol(samples_per_symbol)
    modulation_known = modulation is not None
    if modulation_known:
        modulation = _validate_modulation(modulation)
    if frequency_offset_hz is not None:
        frequency_offset_hz = _validate_frequency_offset_hz(frequency_offset_hz)

    stage_status: dict[str, str] = {stage: "skipped" for stage in _STAGES}
    failure_reasons: dict[str, str] = {}

    peak_frequency: PeakFrequencyEstimate | None = None
    occupied_bandwidth: OccupiedBandwidthEstimate | None = None
    snr: SNREstimate | None = None
    noise_floor: NoiseFloorEstimate | None = None
    symbol_rate: SymbolRateEstimate | None = None
    timing: SymbolTimingRecovery | None = None
    modulation_estimate: ModulationEstimate | None = None
    cfo_corrected_samples: npt.NDArray[np.complex128] | None = None
    bits: npt.NDArray[np.int64] | None = None
    top_level_failure: str | None = None

    # Stage: dominant frequency (Phase 2A). Always computed as metadata.
    try:
        peak_frequency = estimate_peak_frequency(values, sample_rate)
        stage_status["peak_frequency"] = "success"
    except ValueError as exc:
        reason = f"peak frequency estimation failed: {exc}"
        stage_status["peak_frequency"] = "failed"
        failure_reasons["peak_frequency"] = reason
        top_level_failure = top_level_failure or reason

    # Stage: occupied bandwidth (Phase 2B). Always computed as metadata.
    try:
        occupied_bandwidth = estimate_occupied_bandwidth(values, sample_rate)
        stage_status["occupied_bandwidth"] = "success"
    except ValueError as exc:
        reason = f"occupied bandwidth estimation failed: {exc}"
        stage_status["occupied_bandwidth"] = "failed"
        failure_reasons["occupied_bandwidth"] = reason
        top_level_failure = top_level_failure or reason

    # Stage: noise floor / SNR (Phase 2C), only if regions are available.
    if occupied_bandwidth is not None:
        regions = _snr_regions(occupied_bandwidth, sample_rate)
        if regions is not None:
            signal_region, noise_region = regions
            try:
                snr = estimate_snr(values, sample_rate, signal_region, noise_region)
                noise_floor = NoiseFloorEstimate(
                    noise_power_density=snr.noise_power_density,
                    noise_floor_db=10.0 * math.log10(snr.noise_power_density),
                    noise_power=snr.estimated_noise_power,
                    noise_bandwidth_hz=snr.noise_bandwidth_hz,
                    lower_frequency_hz=snr.noise_lower_frequency_hz,
                    upper_frequency_hz=snr.noise_upper_frequency_hz,
                    resolution_hz=snr.resolution_hz,
                )
                stage_status["snr"] = "success"
            except ValueError as exc:
                reason = f"SNR/noise estimation failed: {exc}"
                stage_status["snr"] = "failed"
                failure_reasons["snr"] = reason

    # Stage: symbol rate (Phase 2E). Always computed as metadata,
    # regardless of whether samples_per_symbol was supplied.
    estimated_sps: int | None = None
    try:
        symbol_rate = estimate_symbol_rate(values, sample_rate)
        estimated_sps = symbol_rate.samples_per_symbol
        stage_status["symbol_rate"] = "success"
    except ValueError as exc:
        reason = f"symbol rate estimation failed: {exc}"
        stage_status["symbol_rate"] = "failed"
        failure_reasons["symbol_rate"] = reason
        if not sps_known:
            top_level_failure = top_level_failure or reason

    effective_sps: int | None = samples_per_symbol if sps_known else estimated_sps

    # Stage: CFO correction (Phase 2G), only if a known CFO was supplied.
    working_samples = values
    if frequency_offset_hz is not None:
        try:
            cfo_corrected_samples = correct_frequency_offset(
                values, sample_rate, frequency_offset_hz
            )
            working_samples = cfo_corrected_samples
            stage_status["cfo_correction"] = "success"
        except ValueError as exc:
            reason = f"CFO correction failed: {exc}"
            stage_status["cfo_correction"] = "failed"
            failure_reasons["cfo_correction"] = reason
            top_level_failure = top_level_failure or reason

    # Stage: integer timing-phase recovery (Phase 2H).
    if effective_sps is not None:
        try:
            timing = recover_symbol_timing(working_samples, effective_sps)
            stage_status["timing_recovery"] = "success"
        except ValueError as exc:
            reason = f"timing recovery failed: {exc}"
            stage_status["timing_recovery"] = "failed"
            failure_reasons["timing_recovery"] = reason
            top_level_failure = top_level_failure or reason
    else:
        # samples_per_symbol is unavailable, so timing recovery is
        # skipped rather than failed: it never ran, and per the
        # documented contract, failure_reasons holds only stages whose
        # stage_status is "failed".
        reason = "samples_per_symbol unavailable; timing recovery skipped."
        top_level_failure = top_level_failure or reason

    # Stage: BPSK vs. QPSK classification (Phase 2F). Always computed as
    # metadata when timing succeeded, regardless of whether modulation
    # was supplied.
    if timing is not None:
        try:
            modulation_estimate = estimate_modulation(timing.symbols, 1)
            stage_status["modulation_classification"] = "success"
        except ValueError as exc:
            reason = f"modulation classification failed: {exc}"
            stage_status["modulation_classification"] = "failed"
            failure_reasons["modulation_classification"] = reason

    effective_modulation: str | None = None
    if modulation_known:
        effective_modulation = modulation
    elif modulation_estimate is not None and modulation_estimate.modulation in (
        "BPSK",
        "QPSK",
    ):
        effective_modulation = modulation_estimate.modulation.lower()

    # Stage: demodulate, only when every parameter needed is known.
    # demod_skip_reason is a local fallback for the top-level
    # failure_reason only; it is not stored in failure_reasons when the
    # stage was merely skipped (never attempted) rather than failed.
    demod_skip_reason: str | None = None
    if timing is not None and effective_modulation is not None:
        try:
            if effective_modulation == "bpsk":
                bits = bpsk_demodulate(timing.symbols, 1)
            else:
                bits = qpsk_demodulate(timing.symbols, 1)
            stage_status["demodulation"] = "success"
        except ValueError as exc:
            reason = f"demodulation failed: {exc}"
            stage_status["demodulation"] = "failed"
            failure_reasons["demodulation"] = reason
            bits = None
    else:
        reasons = []
        if timing is None:
            reasons.append("timing recovery unavailable")
        if effective_modulation is None:
            if modulation_estimate is not None and modulation_estimate.modulation == (
                "AMBIGUOUS"
            ):
                reasons.append(
                    "modulation classification was ambiguous "
                    f"(confidence {modulation_estimate.confidence:.4f} "
                    "below threshold)"
                )
            else:
                reasons.append("modulation unavailable")
        demod_skip_reason = "bits not recovered: " + "; ".join(reasons) + "."

    if bits is not None:
        status = "complete"
        failure_reason = None
    elif top_level_failure is not None:
        status = "failed"
        failure_reason = top_level_failure
    else:
        status = "partial"
        failure_reason = failure_reasons.get(
            "demodulation",
            demod_skip_reason
            or "one or more required stages did not produce a usable result; "
            "bits were not recovered.",
        )

    return ControlledReceiverResult(
        status=status,
        failure_reason=failure_reason,
        sample_rate=sample_rate,
        peak_frequency=peak_frequency,
        occupied_bandwidth=occupied_bandwidth,
        snr=snr,
        noise_floor=noise_floor,
        symbol_rate=symbol_rate,
        samples_per_symbol=effective_sps,
        samples_per_symbol_known=sps_known,
        timing=timing,
        timing_offset=timing.timing_offset if timing is not None else None,
        modulation_estimate=modulation_estimate,
        modulation=effective_modulation,
        modulation_known=modulation_known,
        frequency_offset_hz=frequency_offset_hz,
        cfo_corrected_samples=cfo_corrected_samples,
        bits=bits,
        stage_status=stage_status,
        failure_reasons=failure_reasons,
    )