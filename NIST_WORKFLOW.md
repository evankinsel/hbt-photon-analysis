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
python -m unittest -v test_hbt_nist_acquisition.py
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
