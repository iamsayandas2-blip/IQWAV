"""Signal activity detection and active-region extraction for IQWAV."""

from .activity import ActiveRegion, ActivityDetectionResult, detect_activity

__all__ = ["ActiveRegion", "ActivityDetectionResult", "detect_activity"]