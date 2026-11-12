# HBT photon timing: NIST research workflow

`hbt_nist_acquisition.py` is a separate, standalone extension of
`trng_ase_acquisition.py`. The original acquisition script stays unchanged.
It retains the dual-SPAD acquisition menu, short/full/continuous modes, saved-run
analysis, experiment metadata, detector balance, inter-arrival PDFs and offline
math checks. Laser current, temperature and ND attenuation are controlled manually.

## Start here

Requires Python 3.10+ and the same scientific dependencies as the original:

```text
python -m pip install numpy scipy matplotlib seaborn
python hbt_nist_acquisition.py --mode selftest --no-show
python hbt_nist_acquisition.py
```

The Swabian Time Tagger Python driver is required only for real acquisition;
install it using the instrument vendor's instructions. SciencePlots is optional.
In Spyder, open the new file and run it to use the menu. Review the settings near
the top and in the `PREDECLARED SAMPLING AND NIST ASSESSMENT` section before recording.
The existing serial, channels 2/3, 0.5 V trigger thresholds and 22 ns nominal dead
time are retained. Confirm these against the actual apparatus.

```text
python hbt_nist_acquisition.py --mode full --no-show
python hbt_nist_acquisition.py --mode analyze --folder "D:\measurements\RUN" --no-show
python -m unittest discover -p "test_hbt*.py" -v
```

Record the fixed ND filter, laser operating point, thermal settling, dark-count
conditions and any detector changes in the metadata/run notes. Configuration is
saved beside the new script in `.hbt_nist_config.json` (separate from the original
script's configuration). Outputs default to `Downloads/qrng_data` when available;
the menu lets you select another location.

## What is assessed

```text
405 nm laser -> fixed ND -> 50:50 splitter -> SPAD A / SPAD B
    -> Time Tagger timestamp + channel + integrity records
    -> first predefined consecutive per-channel inter-arrival intervals
         |-> fixed timing digitizer -> official SP 800-90B non-IID estimator
         |-> existing interval comparison -> selected SP 800-22 diagnostics
```

Before opening the device, the script saves `sampling_plan.json`. The defaults
select the first **1,000,000 consecutive intervals per detector**, independently,
without a common-window crop, pooling A/B, sorting, dead-time filtering or
post-hoc selection. Each detector therefore needs at least 1,000,001 timestamps.
Short runs are useful for acquisition checks but cannot meet the 90B input size.
The acquisition stop target remains a channel-A event count, as in the original;
channel B can be insufficient even when channel A reaches its target.

The declared digitizer is:

```text
symbol = min(interval_ps // bin_width_ps, 2**bits_per_symbol - 1)
```

Defaults are 100,000 ps (100 ns) bins and 8-bit symbols. The last bin contains
every interval at or above 25.5 microseconds. This intentionally preserves long
intervals in a saturation bin instead of discarding or wrapping them. The report
counts top-bin samples and occupied symbols. **The bin width is an illustrative
experimental choice, not a NIST requirement.** Select and justify it using the
instrument resolution and independent pilot data before collecting the assessment
run. A large saturation fraction can make this digitizer uninformative.

The 90B estimate applies to this defined digitized timing sequence. It does not
estimate every picosecond of entropy in a timestamp or establish quantum origin.
Dead time, afterpulsing, classical laser noise and drift require physical analysis.
The non-IID track avoids making an unsupported independence assumption.

Both branches use the same selected intervals. The existing extraction rule is
preserved: compare disjoint pairs, output 1 for first > second, 0 for first <
second, and discard ties. Comparison uses the **unquantized** intervals. The report
records ties, an unpaired final interval, bit bias and empirical marginal entropy.
That marginal statistic is not used to size a Toeplitz conditioner. The legacy
Toeplitz function remains available for its math self-test, but the analysis
pipeline does not generate conditioned output.

## Running the official SP 800-90B estimator

Build and self-test NIST's [EntropyAssessment software](https://github.com/usnistgov/SP800-90B_EntropyAssessment)
following its README. It is an external C++ dependency, not bundled with this script.
The Python workflow does not substitute a homegrown entropy estimator when it is
absent. Point it to a compatible compiled executable:

```text
python hbt_nist_acquisition.py --mode analyze --folder "D:\measurements\RUN" --ea-non-iid "C:\tools\ea_non_iid.exe" --no-show
```

The runner invokes `ea_non_iid -i -a -v raw_timing_A.u8 8` (then B), using the
saved symbol width. It requires at least 1,000,000 symbols and a complete planned
prefix. Each input byte holds exactly one symbol; this format is **not** the packed
extractor bitstream. Command arguments, input/executable SHA-256 hashes, stdout,
stderr, exit code and parsed final estimates are retained. Tool errors, timeouts,
unrecognized output and missing tools remain explicit statuses. `completed` means
the estimator ran successfully, not that the source passed validation.

NIST primarily documents a Linux/GCC build. A Linux executable cannot be launched
directly by Windows Python. Either run this Python analysis in the same Linux/WSL
environment as the estimator, or export on Windows and run the official tool from
the exported directory in Linux/WSL. The runner accepts a single executable path,
not a shell command such as `wsl ...`. For manual runs, retain the logs and command
alongside the report; the Python report will still say `not_run` until the runner
itself executes the assessment.

This supports initial sequential-data assessment only. Full
[SP 800-90B](https://doi.org/10.6028/NIST.SP.800-90B) evaluation also needs a justified
noise-source model and entropy claim, genuine restart data, startup/continuous
health tests and evidence over operating conditions. Splitting one continuous
recording into rows does not create restart data. These requirements are listed as
unassessed in every report.

## Supplementary SP 800-22 checks

The script implements three tests from [SP 800-22 Rev. 1a](https://doi.org/10.6028/NIST.SP.800-22r1a):
frequency (monobit), block frequency and runs. It is **not the full 15-test STS
suite**. Defaults use disjoint 100,000-bit sequences, 2,000-bit blocks and alpha
0.01. Reports include per-sequence p-values/decisions, runs-prerequisite failures,
pass counts and unused tail lengths. Short data is marked insufficient, never
padded or repeated to manufacture a test sequence.

Pass-fraction intervals and ten-bin p-value uniformity are reported only with at
least 55 sequences. Smaller runs still provide individual diagnostic results.
These aggregate formulas assume suitable independent sequences; splitting a
correlated source into disjoint chunks alone does not establish independence.
Statistical test passes do not establish unpredictability or cryptographic fitness.

With 1,000,000 raw intervals, extraction yields at most 500,000 bits before tie
rejection: at most five default sequences, often four. A meaningful uniformity
assessment at the default sequence length needs at least 5.5 million extracted
bits (over 11 million raw intervals when ties occur). If planning that experiment,
increase `RAW_SAMPLE_COUNT`, `FULL_A_EVENTS` and `FULL_TIMEOUT_S` together, or use
continuous acquisition with enough events on both channels. Continuous acquisition
still accumulates per-channel arrays in memory; budget RAM and disk accordingly.

For the full official STS, use `interval_bits_A.txt` or B in its ASCII-input mode,
with the exact sequence length and number of complete sequences recorded in the
report. The file contains only ASCII `0`/`1`, without padding. Configure the full
suite's test-specific parameters and data requirements separately; the three-test
configuration is not a guarantee of suitability for every STS test.

## Files and reproducibility

- `raw_event_log/block_*.npz`: stream-order blocks with timestamps, channels, event
  types and missed-event counts, saved before per-channel filtering.
- `raw_ch2.npy`, `raw_ch3.npy`, `acquisition.json`: original-compatible acquisition
  outputs. Any loss, overflow, invalid status or malformed timestamp data blocks
  NIST exports and extraction.
- `sampling_plan.json`: choices recorded before new acquisition. An old saved run
  without this file is explicitly labeled `retrospective_legacy_run`; the script
  does not pretend that its analysis choices were made in advance.
- `nist_analysis_<UTC>/nist_report.json`: a new report folder on every analysis,
  including the copied sampling plan, provenance and input hashes. Existing raw
  arrays and previous assessment folders are not overwritten.
- `raw_timing_A.u8`, `raw_timing_B.u8`: one byte per unconditioned timing symbol.
- `selected_intervals_*_ps.npy`: the exact selected, unquantized timing intervals.
- `interval_bits_*.txt`: all generated bits as ASCII, including any final partial
  byte. `interval_bits_*.bin`: complete MSB-first bytes only; omitted tail bits
  are counted explicitly in the report.
- `raw_timing_*.ea_stdout.txt` / `.ea_stderr.txt`: external estimator diagnostics
  when the official executable is invoked.

Stop continuous recording with Ctrl+C once so cleanup, final draining and saving
can complete. Terminating the kernel/process cannot guarantee those final steps;
the already written raw blocks provide partial records, not a valid completed run.
Raw block journaling does not replace the vendor's hardware/stream overflow checks.

The tests run without the instrument. They cover published NIST example values,
input formats, data-loss gates, timing validation, saved-plan reuse, legacy-run
provenance and mocked estimator outcomes. Real hardware and the compiled official
NIST estimator still require validation in the experimental environment.

## Two-detector HBT correlation

The same script now computes the A/B cross-correlation after acquisition and during
saved-run analysis. This is an independent diagnostic branch: it uses all usable
timestamps in their common observation span, not the prefix selected for NIST.
It never alters raw timestamps, the NIST timing digitizer or the comparison bits.
The only acquisition change for this feature is saving `g2_plan.json` before
recording; all pair counting happens afterward. It adds processing time and a new
output folder, not another hardware measurement or live correlation display.

The default settings are **1 ns bins**, **201 bins** with centers from -100 ns to
+100 ns, and zero instrument-delay correction. These are starting settings, not
measured instrument resolution. All bin edges are left-inclusive/right-exclusive;
the central bin covers [-0.5 ns, +0.5 ns). The reported central value is a bin
average, not an infinitely resolved measurement at exactly zero delay.

Every A/B pair inside the lag range contributes; this is not a nearest-neighbor
start/stop histogram. The explicit sign convention is:

```text
corrected_lag_ps = timestamp_B - timestamp_A - G2_DELAY_PS
```

A positive calibrated delay means B's instrument response arrives later than A's.
Calibrate with an appropriate common timing reference and account for detector,
cable and optical delays; do not automatically center the largest noisy bin in
the research data. Adjusting this analysis delay does not shift the stored data.

For stationary independent arrivals, expected accidental pairs in bin i are:

```text
E_i = (N_A / T) * (N_B / T) * integral_bin max(T - abs(tau + delay), 0) d_tau
g2_i = observed_pairs_i / E_i
```

The overlap integral corrects finite-window edges. For narrow lag ranges and zero
delay this reduces approximately to `R_A * R_B * T * bin_width`, consistent with
[Swabian's documented correlation normalization](https://www.swabianinstruments.com/static/documentation/TimeTagger/api/measurements/time_histograms.html).
The script does not force distant bins to average to one or subtract background.
Normalization assumes stationary rates; inspect the count-rate plot for drift.

The current acquisition does not measure independent start/end gates in hardware
timestamp units. Therefore, by default HBT uses the intersection of the observed
A and B event spans, with a half-open window, and explicitly labels the exposure
as inferred. That can introduce finite-sample endpoint bias. It does not invent
an exposure from Python wall-clock time. If independently measured gates are
available, the optional acquisition metadata field `capture_window_ps: [start,end]`
can supply them; its provenance must be established separately.

Each `hbt_g2_<UTC>` folder contains:

- `g2_correlation.pdf`: normalized curve, unity reference and approximate count
  uncertainty bands.
- `g2_histogram.csv` and `.npz`: exact lag edges, coincidence counts, expected
  accidental counts, normalized values and uncertainty bounds.
- `g2_report.json`: input hashes, settings, window provenance, count rates, central
  bin result and interpretation limitations.

Uncertainty bounds are 95% Garwood Poisson intervals on counts divided by the
estimated accidental denominator. A zero-count bin still has a positive upper
bound. These are approximate bands: they omit rate-estimation uncertainty,
dependencies between pairs, drift, detector response and other systematic effects.
They are not a complete uncertainty budget or a formal test of bunching.

The code flags fewer than 25 expected accidental pairs in the central bin. For
scale, at 100,000 counts/s on each detector, a 20 s recording and a 1 ns bin give
about 200 accidental pairs per central bin (roughly 7% Poisson relative counting
noise near g2=1). Smaller bins or lower rates require longer recordings. A shorter
bin cannot recover temporal structure already blurred by detector timing jitter.

Automatic analysis reuses the saved HBT plan. Menu option **8**, or the following
command, recalculates only HBT with the current settings and labels them
retrospective; it does not rerun the entropy estimator:

```text
python hbt_nist_acquisition.py --mode g2 --folder "D:\measurements\RUN" --g2-bin-ps 1000 --g2-half-bins 100 --g2-delay-ps 0 --no-show
```

Set `G2_ENABLED = False` or pass `--no-g2` to skip HBT. The histogram has an explicit
pair-processing limit (`G2_MAX_PAIRS`, default ten million). If exceeded, analysis
reports a blocked result; it never exports a silently truncated histogram. Narrow
the lag range or deliberately increase the limit when appropriate.

### How this helps the further study

Use HBT as a supporting measurement of photon correlations while comparing stable
operating points below, near and above threshold, and across selected temperature
settings. A [diode-emission study](https://arxiv.org/abs/1709.05798) demonstrates the
relevance of second-order coherence to changing diode operating conditions, but
does not establish what your particular 405 nm diode will show.

A resolved excess near zero delay is consistent with bunching, after instrumental
effects are excluded. A curve near one means no resolved excess at the measured
timescales and uncertainty. It does not prove coherent emission, independent timing
samples, quantum origin or secure random bits. Multimode emission, background,
timing jitter and bin averaging can suppress visible contrast. Crosstalk can
produce a spurious peak; detector effects are not automatically corrected.

Record dark/blocked-source controls and independently calibrate relative timing.
Keep the acquisition and analysis choices consistent across comparison runs;
check whether each run has adequate coincidence statistics. HBT supplements the
physical source characterization, while SP 800-90B and the extracted-bit tests
remain separate analyses.

HBT regression tests check exact all-pair counts and boundary conventions,
calibrated peak recovery, finite-window normalization, a simulated independent
baseline, large timestamps, loss blocking and unchanged raw/NIST files. Hardware
comparison against Swabian's correlation measurement remains a lab validation step.
