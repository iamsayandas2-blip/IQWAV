# IQWAV Development Log

This file records meaningful development progress in chronological order.

It answers:

- What was done?
- What currently works?
- What was tested?
- What decisions were made?
- What remains incomplete?
- What should happen next?

Newest entries should be added at the top below this introduction.

## Phase 2G — Controlled Carrier-Frequency-Offset Correction

### Scope

Implemented a bounded carrier-frequency-offset correction primitive for
complex IQ samples.

The correction requires:

- known sample rate;
- known frequency offset;
- complex IQ input;
- a constant frequency offset across the analyzed block.

This is not blind CFO estimation or carrier recovery.

### Implementation

- Added `src/iqwav/dsp/frequency_correction.py`.
- Added `correct_frequency_offset()`.
- Exported the function through `iqwav.dsp`.
- Uses a vectorized complex conjugate-phasor multiplication.
- Uses sample indices `0, 1, ..., N-1`.
- Returns a new `complex128` array.
- Preserves sample count and signal magnitude.
- Does not modify the input array.
- Does not use FFT, filtering, resampling, or amplitude normalization.

### Correction Convention

If the received signal is:

    r[n] = s[n] * exp(+j * 2π * f_offset * n / sample_rate)

then the corrected signal is:

    s_hat[n] = r[n] * exp(-j * 2π * f_offset * n / sample_rate)

The correction therefore applies the conjugate of the phasor used by
`apply_frequency_offset()`.

### Validation

Tested:

- zero frequency offset;
- positive frequency offset;
- negative frequency offset;
- multiple sample rates;
- multiple samples-per-symbol values;
- controlled BPSK waveforms;
- controlled QPSK waveforms;
- offsets larger than the symbol rate;
- amplitude preservation;
- sample-count preservation;
- output allocation;
- deterministic behavior;
- input non-mutation;
- output dtype;
- invalid sample rates;
- invalid frequency offsets;
- empty input;
- non-one-dimensional input;
- non-finite input;
- real-valued input rejection;
- recovery through the existing known-timing demodulators.

Focused tests: 32 passed.

Full regression: 927 passed.

### Limitations

The frequency offset must be supplied by the caller.

This primitive does not perform:

- blind CFO estimation;
- spectral peak searching;
- carrier recovery;
- PLL or Costas-loop tracking;
- timing recovery;
- symbol-rate estimation;
- modulation classification;
- filtering;
- resampling;
- FEC;
- de-interleaving;
- framing;
- payload recovery;
- GUI processing.

Only a known, constant frequency offset is corrected.

### Status

Phase 2G implemented and validated as a controlled CFO-correction
primitive.

Blind carrier and timing recovery remain out of scope.
----------------

## Phase 2F — Controlled BPSK/QPSK Modulation Estimation

### Scope

Implemented a bounded modulation-estimation primitive for controlled,
rectangular-pulse IQWAV waveforms.

The estimator requires:

- known integer samples-per-symbol;
- known symbol-boundary offset;
- symbol-synchronous sampling;
- controlled IQWAV-generated BPSK/QPSK waveforms.

This is not blind automatic modulation recognition.

### Implementation

- Added `src/iqwav/estimation/modulation.py`.
- Added `ModulationEstimate`.
- Added `estimate_modulation()`.
- Exported the estimator through `iqwav.estimation`.
- Uses interior samples from complete symbol periods.
- Uses second- and fourth-order rotational-symmetry statistics.
- Supports constant amplitude scaling and constant phase rotation.
- Does not modify, filter, resample, or synchronize the input.

### Supported Classes

- BPSK
- QPSK

Unsupported:

- FSK
- QAM
- higher-order PSK;
- AM/FM classification;
- blind AMR;
- timing recovery;
- carrier recovery;
- FEC;
- framing;
- payload recovery;
- machine learning.

### Ambiguity and Limitations

The estimator classifies the observed constellation, not transmitter intent.

A QPSK stream using only one antipodal pair is mathematically
indistinguishable from genuine BPSK. Therefore, an antipodal observation
is reported as BPSK in the observed-constellation sense, but this does
not establish the transmitter's intended modulation.

Confidence is a classification margin, not a probability, error rate,
or guarantee of modulation identity.

Sixteen complete symbols is the numerical minimum. At least 64 complete
symbols are recommended for more reliable classification.

Short blocks, imbalanced constellations, unsupported modulations,
unmodulated tones, and residual carrier drift can produce misleading
classifications.

### Validation

Tested:

- clean BPSK and QPSK;
- multiple samples-per-symbol values;
- boundary offsets;
- amplitude scaling;
- constant phase rotations;
- AWGN;
- residual CFO;
- short blocks;
- imbalanced and subset constellations;
- invalid inputs;
- constant and zero signals;
- determinism;
- input non-mutation.

Focused tests: 304 passed.

Full regression: 895 passed using a writable local pytest
base directory.

The ordinary full run also produced 17 pre-existing Windows
temporary-directory permission errors. Re-running with
`--basetemp=.pytest_tmp` removed those unrelated collection errors.

### Status

Phase 2F implemented and validated as a controlled BPSK/QPSK
modulation-estimation primitive.

Blind modulation recognition remains out of scope.

# LOGS entry — Phase 2E

Append the block below to the top of `LOGS.md`, directly under the
introduction (newest entry first), preserving the existing `---` separators.

---
2026-09-03 — Estimation Phase 2E: bounded symbol/baud-rate estimation for controlled rectangular-pulse signals implemented and verified

### Implementation

Added:

- `src/iqwav/estimation/symbol_rate.py`

Modified:

- `src/iqwav/estimation/__init__.py`

Public API:

- `estimate_symbol_rate(samples, sample_rate, *, min_samples_per_symbol=2,
  max_samples_per_symbol=64, quality_ratio=0.75, min_quality=0.02) ->
  SymbolRateEstimate`
- `SymbolRateEstimate` (frozen dataclass): `symbol_rate_hz`,
  `samples_per_symbol`, `sample_rate_hz`, `symbol_rate_resolution_hz`,
  `quality`, `concentration`, `boundary_offset`, `symbol_count`,
  `effective_transitions`, `searched_samples_per_symbol`

### Why this estimator

The existing waveform model was inspected first. `symbols_to_samples` (and
therefore `bpsk_waveform` and `qpsk_waveform`) builds waveforms with
`numpy.repeat`, holding each symbol constant for exactly
`samples_per_symbol` samples, and `samples_per_symbol` is validated there as
an **integer** `>= 1`. The generated signal is thus piecewise constant with
an integer symbol period and no pulse shaping. That model does support a
defensible estimator, so no stop was required — but it also means integer
periods are the only thing the model can produce, so the API is deliberately
constrained to the integer-period grid rather than pretending to estimate
fractional samples-per-symbol.

An FFT-of-`|diff|` spectral-line approach was rejected: for a perfectly
rectangular pulse train the harmonics of the symbol-rate line are
equal-magnitude in the ideal case, so peak-picking cannot separate the
fundamental from its harmonics without an arbitrary threshold. The
time-domain concentration search below is exact on the grid the generators
actually produce, needs no transition-detection threshold, and yields a
quality figure with a real interpretation.

### Method

Let `d[n] = abs(x[n] - x[n - 1])` for `n = 1 .. N - 1`. For a
rectangular-pulse waveform `d` is exactly zero inside a symbol and non-zero
only where consecutive symbols differ, so it is an impulse train whose
impulses all fall on one residue class of `n` modulo the symbol period.

For each candidate integer period `P` in the searched range:

1. bin `d` by `n % P` (`numpy.bincount`) and take the largest bin's share of
   the total as `concentration`;
2. correct it for the `1 / P` level a structureless signal reaches by
   chance:

       quality = (concentration - 1 / P) / (1 - 1 / P)

   which makes candidates of different `P` directly comparable, and equals
   `T / (T + W)` — boundary-aligned transition mass over total mass —
   independently of `P` for a correct period;
3. keep every candidate whose `quality` is at least
   `quality_ratio * best_quality` and return the **largest** of them.

Step 3 is the crux. Every integer *divisor* of the true period concentrates
the same impulses just as perfectly and therefore ties at the same quality,
while an `m`-fold *multiple* splits them across `m` classes and reaches only
about `best / m` (measured for a clean `sps = 8` block: `q(8) = 1.000`,
`q(16) = 0.487`, `q(24) = 0.334`, `q(32) = 0.258`, `q(64) = 0.132`). Taking
the largest well-supported candidate therefore lands on the symbol period
rather than a sub-multiple, and the default ratio of 0.75 sits inside the
wide gap to the nearest multiple.

The ratio rule is not cosmetic: on a 20 dB SNR block with `sps = 8` the true
period scored `quality = 0.4662` while its divisors scored `0.4698` (`P = 2`)
and `0.4685` (`P = 4`), so an exact-maximum rule would have returned a
divisor. Noise perturbs the tie; the ratio absorbs it.

`symbol_rate_hz = sample_rate / P`, and `symbol_rate_resolution_hz` reports
the grid spacing `sample_rate / P - sample_rate / (P + 1) =
sample_rate / (P * (P + 1))`: the estimator's quantization, not its
statistical error. `boundary_offset` is the residue class that won, i.e. the
symbol-boundary phase within the block (0 for canonical generator output,
`(-trim) mod P` after trimming a partial leading symbol).

### Assumptions

- Rectangular, unshaped pulses: the waveform is piecewise constant over each
  symbol, exactly as `iqwav.modulation.symbols_to_samples` produces.
- One constant symbol period for the whole block, an integer `>= 2` samples.
  This is the only model the generators support, so the API is constrained to
  it; fractional and drifting samples-per-symbol are not estimated.
- The sample rate is known and supplied.
- The symbol sequence actually changes at a representative subset of
  boundaries (pseudo-random data does). Boundaries between equal symbols
  produce no transition and are invisible to any method of this kind.
- The true period lies inside `[min_samples_per_symbol,
  max_samples_per_symbol]`.

### Limitations

- **Not blind, not general.** Pulse-shaped (e.g. root-raised-cosine)
  signals, multipath, timing jitter, fractional resampling, and multi-rate
  captures all violate the piecewise-constant model. No claim is made for
  them, and none is made for real over-the-air captures.
- **True period above the search bound returns a divisor, not an error.** A
  divisor is genuinely consistent with the observed transitions
  (`sps = 32` searched to a maximum of 8 returns 8). The search range must be
  bounded with knowledge of the signal, and it is echoed in the result.
- **Data-dependent ambiguity.** If the stream only ever changes every `m`-th
  boundary, the observable period is `m * P` and that is what is reported.
- **Degeneracy is refused, not guessed.** A block whose transition mass sits
  on effectively one impulse is explained equally well by every candidate
  period, so `effective_transitions` — the threshold-free participation ratio
  `(sum d)^2 / sum(d^2)` over the winning class — must reach 2.0 or a
  `ValueError` is raised. This was found empirically: a 5-symbol,
  `sps = 8` block with one observable transition returned period 9 before the
  guard existed.
- **`min_quality` is length-dependent.** Pure Gaussian noise reaches roughly
  `0.55 / sqrt(N)`: measured best quality `0.0093` at `N = 4096`, `0.0186` at
  `N = 1024`, `~0.030` at `N = 256`. The 0.02 default therefore rejects long
  structureless blocks (and `sps = 1`, which is indistinguishable from a
  random sequence) while still accepting 0 dB BPSK/QPSK at `sps <= 16`
  (`q ~ 0.028-0.10`), but a *short* structureless block can still pass it.
  The contract is only that `result.quality >= min_quality`; the caller must
  read `quality` and decide.
- **`quality` is descriptive, not a confidence interval.** It is a statistic
  of the one analyzed block: no averaging across blocks, no probability
  attached. Clean input gives exactly 1.0; 20 dB BPSK at `sps = 8` gives
  `~0.47`; 0 dB at `sps = 16` gives `~0.028`.
- **Residual carrier offset lowers quality without biasing the estimate.**
  Rotation makes the waveform non-constant inside a symbol: at `0.01 * Rs`
  the period is still exact with `q ~ 0.95`, at `0.25 * Rs` still exact with
  `q ~ 0.40`. No carrier or timing correction is performed here.
- **`samples_per_symbol = 1` is out of range by construction** — a symbol
  rate equal to the sample rate has no intra-symbol structure to measure.

The input samples are never modified: no filtering, resampling, timing
correction, or normalization, and no waveform is returned.

### Validation

Validation order is `sample_rate` (positive, finite), then `samples` (1-D,
non-empty, at least 9 finite values — four periods of the shortest supported
period plus the sample consumed by the first difference), then the estimator
parameters (`min_samples_per_symbol` an integer `>= 2`,
`max_samples_per_symbol` an integer `>= min_samples_per_symbol`,
`quality_ratio` a finite float in `(0, 1]`, `min_quality` a finite float in
`[0, 1)`; `bool` is rejected as a period bound). `max_samples_per_symbol` is
then capped to `(N - 1) // 4`, and a range left empty by that cap raises
rather than silently shrinking to nothing.

Degenerate signals raise `ValueError` instead of returning a fabricated
answer: a constant or all-zero block ("never change value"), a block where no
candidate beats chance (`quality <= 0`, e.g. a strict ramp), a block whose
estimated period falls below `min_quality` (pure noise, `sps = 1`), and a
block with effectively a single transition.

### Tests

Added:

- `tests/unit/test_symbol_rate_estimation.py`

Focused Phase 2E result: `136 passed`.

Pre-change repository baseline: `452 passed`.

Full regression result after Phase 2E: `588 passed` (452 + 136, zero
regressions).

Coverage includes clean BPSK at `sps` 2/3/4/5/8/10/16/25/32/50/64 and clean
QPSK at 2/4/8/10/16/32, each checked against the known ground truth
`FS / sps`; symbol rate scaling across four sample rates; exactly zero error
against ground truth; the clean invariants `quality == concentration == 1.0`,
`boundary_offset == 0`, `symbol_count == n_symbols`; a real-valued
rectangular waveform; the integer-grid resolution identity; noisy BPSK at
20/15/10/5 dB and noisy QPSK at 20/10/5 dB across three `sps` values, all
still exact; monotonically decreasing quality with decreasing SNR; the frozen
dataclass (`FrozenInstanceError`, exact documented field set, field types);
determinism across repeated calls including a copied array; proof the input
array is left untouched; list input; the echoed and length-capped search
range; the leading-partial-symbol phase (trim 3 of `sps = 8` gives
`boundary_offset == 5` and `symbol_count == 299`) and every trim 0-7; a true
period above the search bound returning a divisor; a single-value search
range; the 12-sample minimum block; `quality_ratio` at 1.0 (still exact) and
at 0.05 (documented multiple-of-the-period hazard); `min_quality` tightened
(raises), zeroed (returns an honest low-quality result), and the
`result.quality >= min_quality` contract; constant and all-zero blocks in
both real and complex dtype; a single-transition step; pure complex noise at
`N = 4096`; `sps = 1`; a monotonic ramp; and the full invalid-input set (six
bad sample rates, empty, too-short, 2-D/column/3-D/scalar, non-finite real
and complex samples, non-integer and out-of-range period bounds, non-numeric
and out-of-range `quality_ratio` and `min_quality`, a block too short for the
requested range, and validation order).

Test command used: `PYTHONPATH=src python -m pytest -q --basetemp=.pytest_tmp`
— the package is not installed in this environment, and the explicit
`basetemp` avoids a pre-existing `PermissionError` on the default Windows temp
directory in `tests/unit/test_io.py` that is unrelated to this change.

### Repository state

Nothing was committed or pushed. The working tree holds the new estimator,
the new test module, the `estimation/__init__.py` export update, and this log
entry file. `README.md`, `LOGS.md`, and every pre-existing module and test are
unchanged.

### Next

Deferred and still explicitly out of scope for this module: blind or
pulse-shaped baud-rate estimation, symbol-timing recovery and timing loops,
carrier recovery, modulation classification (AMR), framing, FEC, payload
recovery, activity detection, and GUI work.


---
2026-09-03 — Estimation Phase 2D: known-reference frequency-offset (CFO) estimation implemented and verified

### Implementation

Added:

- `src/iqwav/estimation/frequency_offset.py`

Modified:

- `src/iqwav/estimation/__init__.py`

Public API:

- `estimate_frequency_offset(samples, sample_rate, reference_frequency_hz, *, refine=True) -> FrequencyOffsetEstimate`
- `FrequencyOffsetEstimate` (frozen dataclass): `observed_frequency_hz`,
  `reference_frequency_hz`, `frequency_offset_hz`, `bin_frequency_hz`,
  `resolution_hz`, `refined`

The reference frequency is a caller-supplied input, never inferred. The
estimator reuses the Phase 2A peak-frequency estimator
(`estimate_peak_frequency`) for the observation instead of duplicating FFT,
peak-selection, or refinement logic; Phase 2D contributes only the
reference-relative arithmetic and the reference-specific validation.

### Mathematical definition

    frequency_offset_hz = observed_frequency_hz - reference_frequency_hz

`observed_frequency_hz` is Phase 2A's dominant-component frequency for the
same `refine` setting, and the offset is the plain arithmetic difference: no
modular reduction, unwrapping, scaling, or extra sign convention. A positive
offset means the observed component lies above the reference.

Conventions are inherited unchanged from Phase 2A. Complex IQ input is
searched over the full two-sided spectrum, so both the observation and the
reference are signed frequencies within `[-sample_rate/2, +sample_rate/2]`.
Real input follows the non-negative convention: the observation is folded
onto `[0, sample_rate/2]`, so the reference must also be non-negative and
within that range (pass `abs(f)`).

`resolution_hz` is the raw FFT bin spacing `sample_rate / N`, i.e. the
frequency quantum of the analysis and not an error bound. With `refine=False`
the observation, and therefore the offset, is quantized to that grid (error
bounded by `resolution_hz / 2` for an isolated tone). With `refine=True`
Phase 2A's three-point parabolic log-magnitude interpolation is applied
unchanged. `bin_frequency_hz` always carries the unrefined peak-bin center, so
the raw and refined observations remain distinguishable in one result.

The input signal is never modified: no mixing, de-rotation, resampling,
filtering, or CFO correction is performed, and no corrected signal is
returned. The measured offset is the entire output.

### Validation

Validation order is `sample_rate` (positive, finite), then
`reference_frequency_hz` (finite, and inside the range observable at that
sample rate for the given input kind — the range depends on both), then
`samples`, whose validation is delegated to Phase 2A (1-D, at least 4 finite
values, not constant or all-zero). A reference outside the observable range
raises `ValueError` rather than producing a meaningless difference.

### Boundary behavior

A reference of exactly `0` (DC) or exactly `±sample_rate/2` (Nyquist) is
accepted. A real Nyquist tone measures at `+sample_rate/2` and gives a zero
offset against that reference. For complex input at the Nyquist boundary the
two-sided axis wraps: `numpy.fft.fftfreq` labels the Nyquist bin
`-sample_rate/2`, so an unrefined observation there differenced against a
`+sample_rate/2` reference shows the full `-sample_rate` wrap, while
refinement resolves the same tone back to just below `+sample_rate/2`.
Phase 2D does not attempt to resolve that inherent ambiguity; it is
documented and pinned by a test.

### Tests

Added:

- `tests/unit/test_frequency_offset.py`

Focused Phase 2D result: `71 passed`.

Pristine pre-change repository baseline: `381 passed`.

Full regression result after Phase 2D: `452 passed` (381 + 71, zero
regressions).

Coverage includes zero, positive, and negative CFO; the
`offset == observed - reference` identity across both refine settings;
off-bin observations and off-bin references; `refine=True` versus
`refine=False`, with refinement shown to beat the raw bin and the raw error
shown to stay within half a bin; signed positive and negative complex
frequencies, including an offset that crosses 0 Hz; real input under the
non-negative convention, for both signs of the generated tone and for
negative offsets; noisy complex tones at 20/10/6 dB SNR plus a noisy real
tone; dominant-component selection between two simultaneous tones;
`resolution_hz` tracking the analysis length; the four-sample minimum block;
proof the input array is left untouched; DC and Nyquist boundary references;
the near-Nyquist wrap; integer argument normalization; the full invalid-input
set (bad sample rate, non-finite reference, out-of-range reference for both
the real and complex conventions, empty/too-short/non-1-D/non-finite samples,
and constant or zero signals); frozen-dataclass immutability; and
determinism across repeated calls.

Test command used: `PYTHONPATH=src python -m pytest` — the package is not
installed in this environment.

### Limitations and deferred work

A single FFT block is analyzed, with no windowing, Welch averaging, leakage
correction, or confidence interval. The measurement describes whichever
component is strongest in the block, so a spur or a stronger neighbouring
signal is reported instead; the offset is only meaningful when the dominant
component is the one the reference describes. Sub-bin accuracy relies on
Phase 2A's isolated-sinusoid assumption, and near-Nyquist results carry the
wrap ambiguity described above.

Explicitly out of scope for this module and still deferred: blind carrier
detection, automatic reference-frequency discovery, CFO correction, carrier
recovery, PLL/Costas-loop synchronization, symbol-timing recovery, baud-rate
estimation, AMR/modulation classification, FEC, framing, payload recovery,
and GUI work.

### Repository state

Nothing was committed or pushed. The working tree holds the new estimator,
the new test module, the `estimation/__init__.py` export update, and this log
entry file. `README.md`, `LOGS.md`, and every pre-existing module and test are
unchanged.


---
2026-09-03 — Estimation Phase 2C: explicit noise-floor and SNR estimation implemented and verified

### Implementation

Added:

- `src/iqwav/estimation/noise.py`

Modified:

- `src/iqwav/estimation/__init__.py`

Public API:

- `estimate_noise_floor(samples, sample_rate, noise_region_hz) -> NoiseFloorEstimate`
- `estimate_snr(samples, sample_rate, signal_region_hz, noise_region_hz) -> SNREstimate`
- `NoiseFloorEstimate` and `SNREstimate` are frozen dataclasses.

The caller must explicitly supply a noise-reference region. `estimate_snr`
also requires an explicit signal region. Neither function performs blind noise
detection, activity detection, signal-band selection, or classification.

### Mathematical definition

The implementation reuses IQWAV's existing FFT spectrum utility. For an N
sample block, each FFT bin has power `abs(FFT)**2 / N**2`, so the sum of all
two-sided bin powers equals mean-square sample power by Parseval's identity.
The noise power density is selected-region bin power divided by the selected
physical bin width, in sample-units-squared/Hz. Its dB value is
`10*log10(density)`, referenced to 1 sample-unit-squared/Hz.

Frequency regions are `(lower_hz, upper_hz)`. They select FFT-bin centers
with `lower_hz <= f < upper_hz`. The one exception is the real, even-N
Nyquist endpoint: it is included when `upper_hz == sample_rate/2`, so it can
be measured from the valid real-valued axis. Region precision is limited to
the FFT-bin spacing `sample_rate / N`.

For SNR, measured signal-region power includes signal plus noise. Reference
noise density is multiplied by the selected signal-region bandwidth to yield
expected noise inside that region. Residual signal power is measured total
minus expected noise, and `snr_db = 10*log10(residual_signal_power /
estimated_noise_power)`. This normalizes for different signal/noise-region
bandwidths. Zero reference noise or non-positive residual signal power raises
`ValueError`; no NaN or infinite SNR is returned.

Complex input retains independent signed two-sided bins over
`[-sample_rate/2, sample_rate/2]`. Real input follows Phase 2B: conjugate
bins are folded onto a non-negative axis, while DC and even-N Nyquist bins
are counted once and have half-bin physical width at the axis boundary.

### Tests

Added:

- `tests/unit/test_noise_estimation.py`

Focused Phase 2C result: `28 passed`.

Pristine pre-change repository baseline: `352 passed`.

Full regression result after Phase 2C: `380 passed`.

The focused tests cover deterministic AWGN at several powers, known tone plus
AWGN SNR, unequal-bandwidth normalization, caller-selected regions, positive
and negative complex frequency regions, real folding, DC/Nyquist boundaries,
invalid inputs and regions, degenerate noise/signal cases, frozen dataclasses,
repeatability, and finite-sample tolerances.

### Limitations and deferred work

These estimators depend on caller-provided representative regions and a known
sample rate. They use a single FFT block (no windowing, Welch averaging,
confidence interval, or leakage correction) and do not infer RF semantics
from IQ or WAV samples. Blind noise-floor/SNR estimation, activity detection,
carrier/CFO estimation or correction, synchronization, modulation
classification, baud-rate estimation, FEC, framing, payload recovery, and GUI
work remain deferred.


2026-09-03 — Estimation Phase 2B: occupied-bandwidth estimation implemented and verified

### Implementation

Added a second primitive under the estimation subsystem:

- `src/iqwav/estimation/bandwidth.py`

Modified:

- `src/iqwav/estimation/__init__.py` (export the new public API)

Public API:

- `estimate_occupied_bandwidth(samples, sample_rate, percent_power=99.0) -> OccupiedBandwidthEstimate`
- `OccupiedBandwidthEstimate` (frozen dataclass): `bandwidth_hz`,
  `lower_frequency_hz`, `upper_frequency_hz`, `percent_power`

Reuses the existing `iqwav.dsp.magnitude_spectrum` FFT utility rather than
duplicating FFT logic; the estimator is a higher-level interpretation layer
over that spectrum, in the same style as `estimate_peak_frequency`.

Definition used: **cumulative-power, bin-based occupied bandwidth.** The
smallest contiguous run of FFT bins whose summed power reaches
`percent_power` percent of the total measured spectral power is located via
an exact two-pointer minimum-window search (correct because bin powers are
non-negative). Each bin of width `resolution_hz = sample_rate / N` is
treated as covering `[bin_freq - resolution_hz/2, bin_freq + resolution_hz/2)`;
the reported edges are the outer edges of the first/last included bin, so
`bandwidth_hz` is `(bins in the run) * resolution_hz` except where clamped
at the real-signal boundaries described below.

Behavior:

- Complex IQ input: full two-sided spectrum, bins ordered ascending from
  approximately `-sample_rate/2` to `+sample_rate/2`; a returned interval
  may include negative frequencies, DC, positive frequencies, or any mix.
  Mirror-image tones at `+f`/`-f` are independent physical content and are
  **not** folded together.
- Real-valued input: the FFT spectrum is conjugate-symmetric, so `+f` and
  `-f` bins represent the *same* physical component. These are folded onto
  the non-negative axis before the search (`power[+k] + power[-k]` for
  `0 < k < N//2`; DC and, for even `N`, Nyquist are self-conjugate and used
  unmodified). Folded total power equals the original two-sided total, so
  no energy is discarded or double-counted. Boundaries are clamped to
  `[0, sample_rate/2]`.
- `percent_power` is monotonic: a larger value can never produce a smaller
  bandwidth for the same signal (proven by the two-pointer search: any
  window satisfying a higher power threshold also satisfies a lower one).
- Constant (including all-zero) input raises `ValueError` rather than
  fabricating a bandwidth, since a DC-only signal has no meaningful
  occupied bandwidth under this definition.
- Requires `sample_rate` positive and finite, `percent_power` finite and
  in `(0, 100]`, and at least 4 samples.

This is explicitly a cumulative-power bin-based estimator, not a
noise-floor-aware bandwidth estimator and not an automatic RF
signal-activity detector: it does not separate signal power from noise
power and does not decide whether a signal is present. It is not a
general-purpose blind RF bandwidth estimator.

### Tests

Added:

- `tests/unit/test_bandwidth_estimation.py`

Focused estimation tests: `36 passed`.

Complete project test suite: `352 passed` (316 baseline + 36 new, zero
regressions).

Coverage includes a narrowband on-bin complex tone, two separated complex
tones (including a much weaker second tone at high `percent_power`),
broadband/noise-like signals versus a tone-plus-moderate-noise case,
monotonicity of bandwidth in `percent_power` across a full sweep including
100%, real-valued input (including near-Nyquist and pure-Nyquist-tone
folding correctness), positive/negative complex-frequency content shown
not to cancel or fold, even and odd sample counts for both real and
complex input, DC- and Nyquist-adjacent edge cases, and the full set of
invalid inputs (bad sample rate, empty/too-few/non-1D/non-finite samples,
zero/constant signals, and invalid `percent_power`).

### Next

Per milestone scope, stopping after Phase 2B. CFO, SNR, noise-floor
estimation, activity detection, AMR, synchronization, modulation
classification, baud-rate estimation, FEC, framing, and GUI work remain
deferred to later milestones.


2026-09-02 — Estimation Phase 2A: spectral peak frequency estimation implemented and verified

### Implementation

Added the first primitive under the previously-reserved estimation subsystem:

- `src/iqwav/estimation/spectral.py`
- `src/iqwav/estimation/__init__.py`

Public API:

- `estimate_peak_frequency(samples, fs, *, refine=True) -> PeakFrequencyEstimate`
- `PeakFrequencyEstimate` (frozen dataclass): `frequency_hz`, `bin_frequency_hz`,
  `resolution_hz`, `bin_index`, `refined`

Reuses the existing `iqwav.dsp.magnitude_spectrum` FFT utility rather than
duplicating FFT logic; the estimator is a higher-level interpretation layer
over that spectrum.

Behavior:

- Complex IQ input: full two-sided spectrum search, **signed** frequency
  (a tone at `+f` estimates `+f`; a tone at `-f` estimates `-f`).
- Real-valued input: search restricted to the non-negative half of the
  spectrum (spectrum is conjugate-symmetric for real signals), so the
  returned frequency is always **non-negative**; sign is not meaningful
  for a real tone.
- Sub-bin refinement (default on): standard three-point parabolic
  interpolation of the local log-magnitude around the peak bin, using
  circular neighbors since the FFT spectrum is periodic. `refine=False`
  returns the raw FFT bin center with no interpolation.
- Constant/zero-variance input (including all-zero) raises `ValueError`
  rather than fabricating a frequency, since there is no oscillating
  component to localize.
- Requires `fs` positive and finite, and at least 4 samples.

This is explicitly a spectral peak estimator, not a blind RF carrier
estimator: it assumes `fs` is already known and does not attempt occupied
bandwidth, noise-floor/SNR estimation, activity detection, carrier
recovery, CFO correction, timing recovery, baud-rate estimation, or
modulation classification. Those remain later milestones.

### Tests

Added:

- `tests/unit/test_spectral_estimation.py`

Focused estimation tests: `18 passed`.

Complete project test suite: `316 passed` (298 baseline + 18 new, zero
regressions).

Coverage includes signed complex positive/negative-frequency tones, the
real-signal non-negative convention, off-bin tones with refinement shown
to reduce error versus the raw bin, noisy-tone recovery at moderate SNR,
strongest-tone selection among multiple simultaneous tones, and invalid
inputs (bad sample rate, empty/too-few samples, non-finite samples, and
constant/zero signals).

### Next

Per milestone scope, stopping after Phase 2A. Occupied bandwidth, noise
floor, SNR estimation, activity detection, CFO, and other estimation
tasks are deferred to later milestones.

2026-09-01 — Phase 1 correlation primitives implemented and verified

### Implementation

Added reusable finite-length correlation support under:

- `src/iqwav/correlation/core.py`
- `src/iqwav/correlation/__init__.py`

Public functions:

- `autocorrelation(samples)`
- `normalized_autocorrelation(samples)`
- `cross_correlation(first, second)`
- `normalized_cross_correlation(first, second)`
- `find_correlation_peaks(correlation, lags, ...)`

Cross-correlation uses the explicit convention:

`r_xy[lag] = sum_n x[n + lag] * conj(y[n])`

with full finite-length lags ordered from `-(len(y)-1)` through `len(x)-1`.
Consequently, a delayed copy in the first input yields a peak at positive
delay. Normalized variants use the energy of the exact overlapping samples at
each lag and reject whole-input zero-energy cases.

### Tests

Added:

- `tests/unit/test_correlation.py`

Focused correlation tests: `19 passed`.

Complete project test suite: `298 passed`.

Coverage includes known real and complex sequences, documented complex
conjugation behavior, normalization bounds and zero-energy rejection, expected
delayed-copy peak locations, complex-magnitude peak detection, and invalid
array/option inputs.


## 2026-08-30 — Wideband OTA FM channelization and demodulation verified

### Experiment

Extended real-world FM validation using a wideband Mumbai broadcast-FM IQ capture:

- center frequency: 92.3 MHz
- sample rate: 10 MS/s
- file size: approximately 880 MB
- duration: approximately 11 seconds
- format: complex64
- capture contained multiple broadcast-FM stations

The external IQ recording remains under `data/external/` and is not committed to Git.

### Wideband analysis

A wideband PSD covering approximately 87.3–97.3 MHz showed multiple distinct FM broadcast stations.

A strong station around 92.7 MHz was selected for further processing.

### Channelization

The selected station was approximately +400 kHz relative to the 92.3 MHz recording center.

Processing performed:

`wideband IQ`
→ complex-IQ DC removal
→ frequency translation by approximately -400 kHz
→ target station centered near 0 Hz
→ anti-alias filtering and 40× decimation
→ 10 MS/s reduced to 250 kS/s

The resulting PSD confirmed that the selected FM channel remained while neighboring wideband stations were removed.

### FM demodulation

The isolated channel was processed using the production:

`fm_demodulate()`

The demodulated multiplex spectrum showed structure consistent with broadcast FM:

- strong 0–15 kHz program audio,
- clear ~19 kHz stereo pilot,
- energy in the 23–53 kHz stereo-difference region,
- a feature near the ~57 kHz RDS region.

No stereo or RDS decoding was performed.

### Audio recovery

Mono-compatible audio was recovered using:

`FM multiplex`
→ 15 kHz low-pass filtering
→ demodulated DC removal
→ 50 µs FM de-emphasis
→ resampling from 250 kS/s to 50 kS/s
→ normalization
→ 16-bit WAV

The complete capture produced approximately 11 seconds of clear, intelligible broadcast audio.

Processing of the large recording was performed in chunks rather than loading the entire capture into expanded complex arrays.

### Result

PASS — IQWAV successfully processed a genuine wideband multi-station OTA capture, selected and channelized one FM station, demodulated it with the production FM discriminator, identified expected multiplex structure, and recovered clear audio.


## 2026-08-30 — FM demodulation productionized and real-data verified

### Implementation
Added reusable FM phase-discriminator support:

- `src/iqwav/demod/analog.py`
  - `fm_demodulate(samples)`
- exported through `src/iqwav/demod/__init__.py`
- added focused unit tests in:
  - `tests/unit/test_analog_demodulation.py`

The discriminator computes:

`angle(samples[1:] * conj(samples[:-1]))`

and returns phase increment in radians/sample.

It intentionally does not perform:
- `Fs/(2π)` scaling,
- DC removal,
- filtering,
- resampling,
- normalization,
- de-emphasis,
- stereo decoding,
- carrier/CFO estimation.

### Automated verification
- Focused FM-demodulation tests: 11 passed.
- Full project suite: 279 passed.

Tests cover:
- output shape and dtype,
- positive and negative phase increments,
- wrapped phase differences,
- amplitude invariance,
- invalid real/multidimensional input,
- insufficient samples,
- NaN/Inf rejection.

### Real OTA integration verification
The production `fm_demodulate()` function replaced the manual discriminator in:

`notebooks/experiments/02_real_fm_demodulation.ipynb`

Using the genuine 99.5 MHz broadcast-FM IQ recording, the production function successfully produced the same demodulated multiplex spectrum and recovered approximately 4 seconds of clean, clearly intelligible English audio after low-pass filtering and resampling.

### Result
PASS — the reusable FM discriminator is unit-tested, integration-tested and manually verified on genuine OTA IQ data.


## 2026-08-30 — Real OTA IQ smoke test passed

### What was tested
- Downloaded a genuine over-the-air FM IQ recording:
  `fm_rds_250k_1Msamples.iq`
- Known metadata:
  - sample rate: 250 kHz
  - center frequency: 99.5 MHz
  - format: complex64 / interleaved float32 I,Q
  - 1,000,000 complex samples
  - duration: 4 seconds
- Stored locally under:
  `data/external/fm_rds_250k_1Msamples.iq`
  and kept out of Git by `.gitignore`.

### IQWAV path exercised
- `load_raw_iq()`
- `magnitude_spectrum()`
- `welch_psd()`
- `spectrogram_data()`

### Verification
- IQWAV loader matched direct NumPy complex64 loading.
- Time-domain I/Q samples looked physically plausible.
- FFT showed a broad real FM spectrum across the expected ±125 kHz Nyquist span.
- Welch PSD showed consistent occupied spectral structure.
- Waterfall showed time-varying broadband FM energy with sensible frequency/time orientation.
- No obvious corruption, axis error, clipping, or file-format mismatch was observed.

### Result
PASS — current IQWAV raw-IQ ingestion and basic spectral-analysis foundation successfully processed a genuine OTA SDR capture.

### Notes
- Real data is visibly less ideal than synthetic data: asymmetry, spectral bumps, offsets, and time-varying structure are present.
- These effects should not be artificially cleaned up at this stage; future estimators must handle them.
- No FM demodulation or blind parameter estimation was performed in this milestone.



## 2026-08-30 — WAV and Raw IQ File Ingestion Implemented

### Added

Created:

- `src/iqwav/io/wav.py`
- `src/iqwav/io/raw_iq.py`

Implemented:

- `load_wav(path)`
- `load_wav_iq(path, i_channel=0, q_channel=1)`
- `load_raw_iq(path, dtype=np.float32, iq_order="IQ")`

### Capability

IQWAV can now load signal recordings from disk instead of operating only on arrays generated inside Python.

Supported input paths:

- standard WAV files,
- multi-channel WAV interpreted explicitly as I/Q,
- headerless interleaved raw IQ files.

### WAV Behavior

`load_wav`:

- returns WAV sampling rate and samples,
- preserves SciPy-loaded dtype and values,
- supports mono and multi-channel WAV,
- performs no amplitude normalization,
- does not automatically guess I/Q channel meaning.

`load_wav_iq`:

- requires at least two WAV channels,
- explicitly selects I and Q channels,
- combines them as `I + jQ`,
- returns a one-dimensional `complex128` IQ array.

### Raw IQ Behavior

`load_raw_iq`:

- reads headerless raw files with `np.fromfile`,
- supports explicit `"IQ"` or `"QI"` interleaving,
- supports real scalar dtypes such as float32 and int16,
- returns `complex128` IQ samples.

It deliberately does not infer:

- dtype,
- endianness,
- IQ ordering,
- sampling rate,
- center frequency.

These must currently be provided from metadata or operator knowledge.

### Tests

Added:

`tests/unit/test_io.py`

Current total:

- 268 tests passing.

Tests verify:

- mono WAV round-trip,
- stereo WAV round-trip,
- sampling-rate preservation,
- dtype/value preservation,
- exact WAV I/Q reconstruction,
- alternate channel selection,
- float32 raw IQ reconstruction,
- int16 raw IQ reconstruction,
- QI ordering,
- invalid path/input/channel/order/dtype handling.

### Manual Verification

Created:

`notebooks/learning/03_file_io_and_signal_analysis.ipynb`

Verified the complete path:

`known IQ signal`
→ save as WAV/raw IQ
→ reload from disk
→ reconstruct complex IQ
→ FFT / PSD / spectrogram.

A known 125 Hz complex IQ tone was recovered from both WAV and raw IQ files, and FFT analysis detected the expected 125 Hz spectral peak.

The WAV-loaded and raw-loaded IQ arrays matched each other and matched the original signal within expected floating-point precision.

### Current Capability

IQWAV now supports:

`file on disk`
→ WAV/raw IQ ingestion
→ complex NumPy IQ samples
→ FFT
→ PSD
→ spectrogram
→ existing DSP and demodulation utilities.

### Limitation

Raw IQ is headerless, so its representation cannot currently be determined automatically.

### Next

Begin analysis of externally sourced/real IQ recordings rather than only self-generated files.



## 2026-08-30 — Known-Timing BPSK/QPSK Demodulation Implemented

### Added

Created:

`src/iqwav/demod/digital.py`

with:

- `bpsk_demodulate(samples, samples_per_symbol)`
- `qpsk_demodulate(samples, samples_per_symbol)`

### Capability

IQWAV can now perform known-timing hard-decision demodulation for BPSK and QPSK.

Receiver assumptions:

- symbol boundaries are already known,
- no timing recovery,
- no carrier recovery,
- no CFO correction,
- no phase correction.

For each symbol interval, samples are block-averaged and then mapped back to bits using the corresponding decision regions.

### BPSK Decision Rule

- `real(symbol_average) >= 0` → bit `0`
- `real(symbol_average) < 0` → bit `1`

### QPSK Decision Rule

Using the existing Gray mapping:

- `I >= 0, Q >= 0` → `00`
- `I < 0, Q >= 0` → `01`
- `I < 0, Q < 0` → `11`
- `I >= 0, Q < 0` → `10`

### Tests

Added:

`tests/unit/test_digital_demodulation.py`

Current total:

- 251 tests passing.

Tests verify:

- clean BPSK round-trip,
- clean QPSK round-trip,
- all four QPSK Gray-mapped quadrants,
- `samples_per_symbol = 1`,
- real BPSK input,
- correct recovered bit counts,
- successful seeded recovery after moderate AWGN,
- invalid input handling.

### Manual Verification

Verified end-to-end synthetic communication chains:

`bits → BPSK waveform → AWGN → BPSK demodulation → recovered bits`

and:

`bits → QPSK waveform → AWGN → QPSK demodulation → recovered bits`

Recovered bits matched the transmitted bits in the controlled notebook test.

### Current Capability

IQWAV now supports:

`bits`
→ BPSK/QPSK symbol mapping
→ sampled rectangular waveform
→ AWGN / CFO / phase impairment injection
→ known-timing hard-decision demodulation
→ recovered bits.

### Limitation

The receiver currently assumes perfect symbol timing and does not estimate or correct timing, carrier frequency offset, or phase.

### Next

Add real `.wav` and raw IQ file ingestion before expanding receiver complexity.


## 2026-08-30 — Signal Power and AWGN Utilities Implemented

### Added

Created:

`src/iqwav/dsp/noise.py`

with:

- `signal_power(samples)`
- `add_awgn(samples, snr_db, rng=None)`

### Capability

IQWAV can now:

- compute average signal power using `mean(|x|^2)`,
- add controlled additive white Gaussian noise,
- generate real Gaussian noise for real signals,
- generate circular complex Gaussian noise for IQ signals,
- target a requested SNR in dB,
- reproduce noise deterministically using a seeded NumPy RNG.

### Tests

Added:

`tests/unit/test_noise.py`

Current total:

- 154 tests passing.

Tests verify:

- known signal powers,
- real and complex noise behavior,
- shape and dtype preservation,
- seeded reproducibility,
- measured SNR near requested values,
- invalid-input handling.

### Manual Verification

Compared clean and noisy IQ signals in the learning notebook.

Verified that:

- the waveform becomes visibly noisy,
- the desired tone remains present,
- the spectrum develops a noise floor around the tone.

### Current Capability

IQWAV now supports:

known signal generation
→ controlled noise injection
→ FFT / PSD / spectrogram analysis
→ FIR filtering.

### Next

Continue controlled channel-impairment utilities.


## 2026-08-30 — FIR Filtering Utilities Implemented

### Added

Created:

`src/iqwav/dsp/filters.py`

with:

- `design_lowpass_fir`
- `design_highpass_fir`
- `design_bandpass_fir`
- `apply_fir_filter`

### Capability

IQWAV can now design and apply basic FIR filters for real and complex signals.

Supported filter types:

- low-pass,
- high-pass,
- band-pass.

Filters are designed using `scipy.signal.firwin` and applied using `scipy.signal.lfilter`.

### Validation

Added validation for:

- sampling frequency,
- cutoff frequencies,
- band-pass edge ordering,
- FIR tap count,
- signal shape and finiteness,
- filter-tap shape and finiteness.

### Tests

Added:

`tests/unit/test_filters.py`

Current total:

- 141 tests passing.

Tests verify:

- valid FIR coefficient generation,
- low-pass behavior,
- high-pass behavior,
- band-pass behavior,
- real and complex signal support,
- output-length preservation,
- invalid-input handling.

### Manual Verification

Created a mixed signal containing low- and high-frequency tones and applied a low-pass FIR filter.

Verified in the spectrum that the low-frequency component remained while the high-frequency component was strongly attenuated.

### Current Capability

IQWAV now supports:

known signal generation
→ FFT/PSD/spectrogram analysis
→ basic FIR filtering.

### Next

Continue foundational DSP utilities.


## 2026-08-30 — Spectrogram / Waterfall Data Utility Implemented

### Added

Created:

`src/iqwav/dsp/spectrogram.py`

with:

`spectrogram_data(samples, fs, nperseg=256, noverlap=None)`

### Capability

IQWAV can now compute time-frequency power data for a signal.

The function returns:

- time axis in seconds,
- frequency axis in Hz,
- spectrogram power matrix in linear units.

The frequency axis is arranged as:

negative frequencies → 0 → positive frequencies.

The returned power matrix has shape:

`(number of frequency bins, number of time segments)`

This data will later support the GUI waterfall / spectrogram view required by the SIH problem statement.

### Validation

Added checks for:

- invalid/non-finite sampling frequency,
- non-1-D signals,
- empty signals,
- NaN/Inf samples,
- invalid `nperseg`,
- invalid `noverlap`.

### Tests

Added:

`tests/unit/test_spectrogram.py`

Current total:

- 108 tests passing.

Tests verify:

- output dimensions,
- increasing time axis,
- centered frequency ordering,
- correct signed frequency detection across time,
- invalid-input handling.

### Manual Verification

Used a stationary `-100 Hz` IQ tone and plotted the returned spectrogram data.

Observed a horizontal power ridge around `-100 Hz` across time, as expected for a constant-frequency signal.

### Current Capability

IQWAV now supports:

known tone generation
→ FFT magnitude
→ periodogram PSD
→ Welch PSD
→ spectrogram / waterfall data.

### Next

Continue foundational DSP utilities.


## 2026-08-30 — PSD Utilities Implemented

### Added

Created:

`src/iqwav/dsp/psd.py`

with:

- `periodogram_psd(samples, fs)`
- `welch_psd(samples, fs, nperseg=None)`

### Capability

IQWAV can now estimate power spectral density using:

- a standard periodogram,
- Welch averaged PSD.

Both functions return:

- frequency axis in Hz,
- PSD values in linear units.

Outputs are arranged as:

negative frequencies → 0 → positive frequencies.

### Validation

Added checks for:

- invalid/non-finite sampling frequency,
- non-1-D signals,
- empty signals,
- NaN/Inf samples,
- invalid `nperseg`.

### Tests

Added:

`tests/unit/test_psd.py`

Current total:

- 89 tests passing.

Tests verify:

- output size,
- centered frequency ordering,
- real-tone symmetric PSD peaks,
- signed complex-IQ peak location,
- default and explicit Welch segment lengths,
- invalid-input handling.

### Manual Verification

Used the learning notebook to compare periodogram and Welch PSD.

Observed:

- periodogram gives a sharper/taller peak for the clean synthetic tone,
- Welch gives a broader/smoother peak due to segment averaging.

### Current Capability

IQWAV now supports:

known tone generation
→ FFT magnitude analysis
→ periodogram PSD
→ Welch PSD.

### Next

Continue foundational DSP processing and visualization utilities.


## 2026-08-30 — FFT Magnitude Spectrum Utility Implemented

### Added

Created:

`src/iqwav/dsp/spectrum.py`

with:

`magnitude_spectrum(samples, fs, fftshift=True)`

### Capability

The function:

- accepts real or complex 1-D NumPy signal arrays,
- computes the FFT,
- computes raw FFT magnitude,
- generates the corresponding frequency axis in Hz,
- optionally applies FFT shift so frequency ordering becomes:

negative frequencies → 0 → positive frequencies.

### Validation

Added checks for:

- invalid or non-finite sampling frequency,
- non-1-D input,
- empty arrays,
- NaN or infinite samples.

### Tests

Added:

`tests/unit/test_spectrum.py`

Current total:

- 66 tests passing.

Spectrum tests verify:

- output length,
- correct frequency-axis construction,
- real-tone peaks at ±f,
- complex IQ tone peak at the correct signed frequency,
- shifted and unshifted FFT ordering,
- invalid input handling.

### Manual Verification

Used:

`notebooks/learning/01_tone_generation.ipynb`

to visually inspect the generated IQ spectrum and confirmed the expected spectral peak location.

### Current Capability

IQWAV now supports:

known synthetic tone generation
→ FFT spectrum analysis
→ frequency-domain verification.

### Next

Continue building foundational DSP analysis utilities.

## 2026-08-29 — Synthetic Tone Generator Implemented

### Added

Created:

`src/iqwav/modulation/tones.py`

with reusable generators for:

- real cosine tones,
- complex IQ tones.

Both support:

- sampling frequency,
- tone frequency,
- duration,
- amplitude,
- phase.

### Validation

Added checks for:

- invalid sampling frequency,
- invalid duration,
- negative amplitude,
- non-finite values,
- Nyquist violations,
- impossible sample counts,
- extreme `fs * duration` overflow.

For complex IQ tones, the exact Nyquist boundary is rejected because positive and negative frequency become indistinguishable there.

### Tests

Added:

`tests/unit/test_tones.py`

Current result:

- 52 tests passing.

Tests cover:

- sample count,
- time spacing,
- data types,
- amplitude,
- phase,
- known sample sequences,
- FFT frequency location,
- positive/negative IQ behavior,
- Nyquist policy,
- invalid inputs.

### Manual Verification

Created:

`notebooks/learning/01_tone_generation.ipynb`

and visually verified:

- real cosine waveform,
- I and Q components,
- circular IQ trajectory,
- opposite rotation for positive and negative IQ frequency.

### Current Capability

IQWAV can now generate deterministic real and complex synthetic tones with known ground-truth parameters for downstream DSP testing.

### Next

Implement reusable FFT/spectrum analysis utilities for known synthetic signals.



## 2026-08-29 — Modules 7–8 Completed and Python Environment Established

### Learning Progress

Completed:

- Module 7 — Digital Communication Fundamentals
- Module 8 — Digital Modulation

Current learning boundary is now Modules 0–8.

Next:

- Module 9 — Correlation & Statistical Signal Analysis

### Development Environment

Created project virtual environment using Python 3.11.

Installed initial dependencies:

- NumPy
- SciPy
- Matplotlib
- pytest

Configured the project using `pyproject.toml`.

Installed IQWAV in editable development mode using:

`pip install -e .`

Verified that:

`import iqwav`

resolves directly to:

`src/iqwav/`

### New Implementation Boundary

The project may now implement:

- synthetic digital-modulation generators,
- digital signal visualizations,
- constellation handling,
- pulse-shaping experiments,
- controlled demodulation where parameters are known,
- DSP infrastructure supported by Modules 0–8.

Blind-analysis functionality remains deferred until the relevant later modules are completed.

### Next

Create the first production milestone: controlled synthetic-signal and DSP foundation.



## 2026-08-29 — Repository Foundation

### Status

IQWAV project repository initialized.

### Completed

Created the initial directory architecture containing:

- configs
- data
- docs
- gnuradio
- models
- native
- notebooks
- outputs
- scripts
- src
- tests

Created the Python package structure under:

`src/iqwav/`

with initial subsystem directories for:

- io
- dsp
- modulation
- estimation
- synchronization
- amr
- demod
- interleaving
- fec
- correlation
- framing
- pipeline
- ui
- utils

Created:

- README.md
- LOGS.md
- .gitignore
- requirements.txt
- pyproject.toml

Added `.gitkeep` placeholders where required so important empty directories can exist in Git.

Configured `.gitignore` to prevent large/generated/local files from accidentally entering the repository.

Git repository initialized with:

- default branch: `main`

### Current Learning Position

Completed:

Modules 0 through 6.

Latest completed module:

Module 6 — Analog Modulation.

Next learning module:

Module 7 — Digital Communication Fundamentals.

### Current Product Capability

No production DSP processing components have yet been implemented.

The repository currently provides the software/project foundation only.

### Important Decision

IQWAV will be developed progressively alongside the learning curriculum rather than attempting to implement the complete SIH system immediately.

Development loop:

Theory → Experiment → Implementation → Test → Integrate.

### Next

Establish the Python development environment and choose the first production milestone supported by Modules 0–6.