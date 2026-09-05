"""Active-region analysis orchestration for full IQ captures.

This module provides :func:`analyze_active_regions`, which runs the
existing activity detector (:mod:`iqwav.detection.activity`) over a full
IQ capture and, for each detected active region, independently runs the
existing integrated controlled receiver workflow
(:func:`iqwav.pipeline.controlled_receiver.run_controlled_receiver_pipeline`)
on just that region's samples.

This module performs no new detection, estimation, CFO correction,
timing recovery, or demodulation algorithm: it only slices the input
around the already-detected active regions and dispatches each slice to
the already-implemented, already-tested per-block pipeline. It does not
assume that each active region belongs to a different transmitter, and
it does not attempt multi-signal separation, clustering, or blind
recognition of any kind -- an active region may itself contain zero,
one, or several transmissions; that judgment is left to the caller.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..detection import ActivityDetectionResult, detect_activity
from .controlled_receiver import ControlledReceiverResult, run_controlled_receiver_pipeline

__all__ = [
    "RegionAnalysisResult",
    "ActiveRegionAnalysisResult",
    "analyze_active_regions",
]


@dataclass(frozen=True, eq=False)
class RegionAnalysisResult:
    """Analysis outcome for one active region.

    Attributes:
        start_sample: Region start index in the original, full capture
            (inclusive), exactly as reported by activity detection.
        end_sample: Region end index in the original, full capture
            (exclusive), exactly as reported by activity detection.
        duration_samples: Number of samples in the region
            (``end_sample - start_sample``).
        status: One of ``"analyzed"`` (the per-region pipeline ran to
            completion, though ``receiver`` may still show ``"partial"``
            or ``"failed"`` internally -- see ``receiver.status``) or
            ``"error"`` (the per-region pipeline itself raised and could
            not run at all, e.g. because the region was too short).
        failure_reason: Human-readable explanation when ``status`` is
            ``"error"``, or ``None`` otherwise. This is distinct from
            ``receiver.failure_reason``, which explains a partial or
            failed *result* from a pipeline call that itself ran
            successfully.
        receiver: The :class:`~iqwav.pipeline.controlled_receiver.ControlledReceiverResult`
            for this region -- covering both parameter estimation and,
            when applicable, demodulation -- or ``None`` if ``status``
            is ``"error"`` and the pipeline could not be run.
    """

    start_sample: int
    end_sample: int
    duration_samples: int
    status: str
    failure_reason: str | None
    receiver: ControlledReceiverResult | None


@dataclass(frozen=True, eq=False)
class ActiveRegionAnalysisResult:
    """Outcome of running active-region analysis over a full IQ capture.

    Attributes:
        status: ``"no_activity"`` if activity detection found no active
            regions, or ``"analyzed"`` if one or more regions were
            analyzed (individual regions may still have failed or
            partial results -- see each entry in ``regions``).
        activity: The full :class:`~iqwav.detection.activity.ActivityDetectionResult`
            from activity detection, preserved unchanged.
        regions: Per-region analysis results, one per detected active
            region, ordered deterministically by ascending
            ``start_sample`` (the same order activity detection already
            produces). Empty when ``status`` is ``"no_activity"``.
    """

    status: str
    activity: ActivityDetectionResult
    regions: tuple[RegionAnalysisResult, ...]


def analyze_active_regions(
    samples: npt.ArrayLike,
    sample_rate: float,
    *,
    window_size: int = 256,
    threshold_db: float = 6.0,
    merge_gap_samples: int = 0,
    noise_percentile: float = 50.0,
) -> ActiveRegionAnalysisResult:
    """Detect active regions and analyze each one independently.

    Runs :func:`iqwav.detection.detect_activity` over the full capture,
    then, for each detected region (in ascending ``start_sample`` order),
    extracts that region's samples from the original array -- without
    modifying it -- and runs
    :func:`iqwav.pipeline.controlled_receiver.run_controlled_receiver_pipeline`
    on the extracted slice alone. Every parameter that pipeline can
    estimate (samples-per-symbol, modulation, CFO diagnostics, timing) is
    estimated independently per region; no parameter or transmitter
    identity is assumed to be shared across regions.

    If a region's per-region pipeline call itself raises (for example,
    because the region is too short for timing recovery), that failure is
    captured as an ``"error"`` :class:`RegionAnalysisResult` for that
    region only; it does not abort analysis of the other regions.

    Args:
        samples: 1-D complex IQ sample block for the full capture. Must
            be non-empty, finite, and complex-valued. Not modified.
        sample_rate: Sample rate in Hz. Must be positive and finite.
        window_size: Passed through to :func:`iqwav.detection.detect_activity`.
        threshold_db: Passed through to :func:`iqwav.detection.detect_activity`.
        merge_gap_samples: Passed through to :func:`iqwav.detection.detect_activity`.
        noise_percentile: Passed through to :func:`iqwav.detection.detect_activity`.

    Returns:
        An :class:`ActiveRegionAnalysisResult` describing detection
        metadata and, when regions were found, one
        :class:`RegionAnalysisResult` per region.

    Raises:
        ValueError: If any argument fails validation, propagated
            directly from :func:`iqwav.detection.detect_activity`.
    """
    values = np.asarray(samples)

    activity = detect_activity(
        values,
        sample_rate,
        window_size=window_size,
        threshold_db=threshold_db,
        merge_gap_samples=merge_gap_samples,
        noise_percentile=noise_percentile,
    )

    if not activity.regions:
        return ActiveRegionAnalysisResult(
            status="no_activity",
            activity=activity,
            regions=(),
        )

    ordered_regions = sorted(activity.regions, key=lambda region: region.start_sample)

    region_results: list[RegionAnalysisResult] = []
    for region in ordered_regions:
        # Slice, never mutate, the original capture. A view is fine here
        # since run_controlled_receiver_pipeline never writes to its
        # input.
        region_samples = values[region.start_sample : region.end_sample]
        duration_samples = region.end_sample - region.start_sample

        try:
            receiver_result = run_controlled_receiver_pipeline(
                region_samples, activity.sample_rate
            )
        except ValueError as exc:
            region_results.append(
                RegionAnalysisResult(
                    start_sample=region.start_sample,
                    end_sample=region.end_sample,
                    duration_samples=duration_samples,
                    status="error",
                    failure_reason=f"per-region pipeline failed: {exc}",
                    receiver=None,
                )
            )
            continue

        region_results.append(
            RegionAnalysisResult(
                start_sample=region.start_sample,
                end_sample=region.end_sample,
                duration_samples=duration_samples,
                status="analyzed",
                failure_reason=None,
                receiver=receiver_result,
            )
        )

    return ActiveRegionAnalysisResult(
        status="analyzed",
        activity=activity,
        regions=tuple(region_results),
    )