"""Controlled end-to-end BPSK/QPSK receiver pipeline.

This module provides a single, narrowly-scoped primitive:
:func:`run_receiver_pipeline`, which chains the already-implemented
controlled DSP primitives into a fixed-order receive chain for
rectangular-pulse BPSK or QPSK signals with a known, constant carrier
frequency offset (CFO) and an unknown but constant integer symbol-timing
phase.

Processing order
-----------------
1. Correct the known CFO with
   :func:`iqwav.dsp.correct_frequency_offset` (Phase 2G).
2. Recover the best integer timing phase with
   :func:`iqwav.synchronization.recover_symbol_timing` (Phase 2H).
3. The one-sample-per-symbol block returned by step 2 already *is* the
   extracted symbol samples.
4. Demodulate those symbols with the existing known-timing demodulator,
   :func:`iqwav.demod.bpsk_demodulate` or
   :func:`iqwav.demod.qpsk_demodulate`, called with
   ``samples_per_symbol=1`` since step 2 already reduced each symbol
   period to its single representative sample.
5. Return the recovered bits plus CFO and timing metadata in a frozen
   result dataclass.

This module performs no blind estimation of any kind: the CFO,
modulation type, sample rate, and samples-per-symbol are all supplied by
the caller. Only the integer timing phase is genuinely recovered, by
Phase 2H's bounded search.

Scope
-----
This is a **thin orchestration layer**, not a new DSP algorithm. It is
explicitly not:

- blind CFO estimation, blind timing recovery, symbol-rate estimation,
  modulation classification, carrier recovery, a PLL, or a Costas loop
  (the CFO, modulation, sample rate, and samples-per-symbol are all
  known and supplied; only the integer timing phase is searched, by the
  already-implemented Phase 2H primitive),
- Gardner or Mueller-and-Muller timing recovery, interpolation,
  filtering, resampling, or pulse-shaping support (rectangular pulses
  and integer-sample timing only, exactly as required by the underlying
  primitives),
- FEC, deinterleaving, framing, or payload recovery (only raw
  demodulated bits are returned),
- GUI integration.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import correct_frequency_offset
from iqwav.synchronization import SymbolTimingRecovery, recover_symbol_timing

__all__ = ["ReceiverPipelineResult", "run_receiver_pipeline"]

_SUPPORTED_MODULATIONS = ("bpsk", "qpsk")


@dataclass(frozen=True, eq=False)
class ReceiverPipelineResult:
    """Outcome of running the controlled receiver pipeline.

    Attributes:
        bits: Recovered bits, one per symbol for BPSK or two per symbol
            (bit-pair per symbol, matching :func:`iqwav.demod.qpsk_demodulate`)
            for QPSK. int64 array.
        modulation: The modulation type supplied by the caller, echoed
            back, always ``"bpsk"`` or ``"qpsk"``.
        sample_rate: The sample rate supplied by the caller, echoed back.
        samples_per_symbol: The samples-per-symbol value supplied by the
            caller, echoed back.
        frequency_offset_hz: The known CFO supplied by the caller and
            removed in step 1, echoed back.
        cfo_corrected_samples: The samples after CFO correction, before
            timing recovery. Same length as the input, complex128.
        timing_offset: The integer symbol-boundary phase selected by
            Phase 2H, in ``[0, samples_per_symbol)``.
        timing_recovery: The full :class:`iqwav.synchronization.SymbolTimingRecovery`
            result from step 2, for inspection of timing quality metrics.
        symbol_count: Number of recovered symbols, echoed from
            ``timing_recovery.symbol_count``.
    """

    bits: npt.NDArray[np.int64]
    modulation: str
    sample_rate: float
    samples_per_symbol: int
    frequency_offset_hz: float
    cfo_corrected_samples: npt.NDArray[np.complex128]
    timing_offset: int
    timing_recovery: SymbolTimingRecovery
    symbol_count: int


def _validate_modulation(modulation: object) -> str:
    """Validate the modulation type."""
    if not isinstance(modulation, str) or modulation not in _SUPPORTED_MODULATIONS:
        raise ValueError(
            "modulation must be one of "
            f"{_SUPPORTED_MODULATIONS!r}, got {modulation!r}."
        )
    return modulation


def _validate_sample_rate(sample_rate: object) -> float:
    """Validate the sample rate."""
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
    """Validate samples_per_symbol as an integer >= 1."""
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


def _validate_samples(samples: object) -> npt.NDArray[np.complexfloating]:
    """Validate the received IQ sample block."""
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


def run_receiver_pipeline(
    samples: npt.ArrayLike,
    sample_rate: float,
    samples_per_symbol: int,
    modulation: str,
    frequency_offset_hz: float,
) -> ReceiverPipelineResult:
    """Run the controlled BPSK/QPSK receiver pipeline.

    Applies, in this fixed order: known-CFO correction (Phase 2G),
    integer symbol-timing-phase recovery (Phase 2H), symbol extraction
    from the timing-recovery result, and known-timing demodulation. No
    blind estimation of any kind is performed; the CFO, modulation, and
    both rate parameters must all be supplied and correct.

    Args:
        samples: 1-D complex IQ sample block. Must be non-empty, finite,
            and long enough for Phase 2H to score four symbol windows
            per candidate phase (see
            :func:`iqwav.synchronization.recover_symbol_timing`).
        sample_rate: Sampling frequency in Hz. Must be positive and
            finite.
        samples_per_symbol: Known integer samples per symbol, >= 1.
        modulation: Either ``"bpsk"`` or ``"qpsk"``.
        frequency_offset_hz: Known constant CFO to remove, in Hz. May be
            positive, negative, or zero. Must be finite.

    Returns:
        A :class:`ReceiverPipelineResult` with the recovered bits and
        processing metadata. The input array is not modified.

    Raises:
        ValueError: If ``modulation`` is not ``"bpsk"`` or ``"qpsk"``, if
            ``sample_rate`` is not positive and finite, if
            ``samples_per_symbol`` is not an integer >= 1, if
            ``frequency_offset_hz`` is not finite, if ``samples`` is not
            a 1-D non-empty finite complex array, or if ``samples`` is
            too short for Phase 2H's timing search.
    """
    modulation = _validate_modulation(modulation)
    sample_rate = _validate_sample_rate(sample_rate)
    samples_per_symbol = _validate_samples_per_symbol(samples_per_symbol)
    if not math.isfinite(frequency_offset_hz):
        raise ValueError(
            f"frequency_offset_hz must be finite, got {frequency_offset_hz!r}."
        )
    values = _validate_samples(samples)

    # Step 1: correct the known CFO (Phase 2G).
    cfo_corrected = correct_frequency_offset(
        values, sample_rate, frequency_offset_hz
    )

    # Step 2: recover the best integer timing phase (Phase 2H). This also
    # performs step 3, extraction of one representative sample per symbol.
    timing = recover_symbol_timing(cfo_corrected, samples_per_symbol)

    # Step 4: demodulate with the existing known-timing demodulators.
    # samples_per_symbol=1 because timing recovery already reduced each
    # symbol period to a single representative sample.
    if modulation == "bpsk":
        bits = bpsk_demodulate(timing.symbols, 1)
    else:
        bits = qpsk_demodulate(timing.symbols, 1)

    return ReceiverPipelineResult(
        bits=bits,
        modulation=modulation,
        sample_rate=sample_rate,
        samples_per_symbol=samples_per_symbol,
        frequency_offset_hz=frequency_offset_hz,
        cfo_corrected_samples=cfo_corrected,
        timing_offset=timing.timing_offset,
        timing_recovery=timing,
        symbol_count=timing.symbol_count,
    )