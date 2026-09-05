"""General DSP operations for IQWAV."""

from .band_extraction import BandExtractionResult, extract_band
from .filters import (
    apply_fir_filter,
    design_bandpass_fir,
    design_highpass_fir,
    design_lowpass_fir,
)
from .frequency_correction import correct_frequency_offset
from .impairments import apply_frequency_offset, apply_phase_offset
from .noise import add_awgn, signal_power
from .psd import periodogram_psd, welch_psd
from .spectrogram import spectrogram_data
from .spectrum import magnitude_spectrum

__all__ = [
    "BandExtractionResult",
    "add_awgn",
    "apply_fir_filter",
    "apply_frequency_offset",
    "apply_phase_offset",
    "correct_frequency_offset",
    "design_bandpass_fir",
    "design_highpass_fir",
    "design_lowpass_fir",
    "extract_band",
    "magnitude_spectrum",
    "periodogram_psd",
    "signal_power",
    "spectrogram_data",
    "welch_psd",
]