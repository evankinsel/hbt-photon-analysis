# -*- coding: utf-8 -*-
"""
Dual-SPAD Photon-Counting / NIST Research Assessment

405 nm laser research | Swabian Time Tagger | Spyder

Features:
- Dual-SPAD photon timestamp acquisition
- Short, full, and continuous baseline recording
- Manual stopping with automatic data saving
- Count rates and A:B detector ratios
- Inter-arrival time distributions
- Exploratory QRNG bit extraction and statistics
- Publication-quality vector PDF plots
- Predeclared raw timing samples and official SP 800-90B non-IID runner
- Three supplementary SP 800-22 diagnostics (not the full 15-test suite)
- Independent two-detector HBT g2 histogram, normalization and count intervals

See NIST_WORKFLOW.md for the sampling model, setup, commands and limitations.

IMPORTANT:
Research outputs are not certified cryptographic keys.
The script does not control laser current or temperature.
"""

from __future__ import annotations

import json
import math
import secrets
import time
import socket
import os
import shutil
import argparse
import hashlib
import re
import subprocess

from scipy.special import gammaincc
from scipy.stats import chi2

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import fft

try:
    import scienceplots
except ImportError:
    scienceplots = None

try:
    from Swabian import TimeTagger
except ImportError:
    try:
        import TimeTagger
    except ImportError:
        TimeTagger = None


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MODE = "menu"

SERIAL = "1948000SQ6"

CH_A = 2
CH_B = 3

TRIGGER_A_V = 0.5
TRIGGER_B_V = 0.5

DEAD_TIME_NS = 22.0
WARNING_CPS = 100_000

SHORT_A_EVENTS = 100_000
FULL_A_EVENTS = 2_000_000

SHORT_TIMEOUT_S = 60
FULL_TIMEOUT_S = 180

STREAM_BUFFER_EVENTS = 500_000

# Default output folder
DEFAULT_DOWNLOADS = Path.home() / "Downloads"

if DEFAULT_DOWNLOADS.exists():
    SAVE_ROOT = DEFAULT_DOWNLOADS / "qrng_data"
else:
    SAVE_ROOT = Path("qrng_data")

ANALYZE_FOLDER = None

TOEPLITZ_DEMO = False  # Kept only for the legacy math self-test; no entropy sizing.
SHOW_FIGURES = True

TEMP_SETPOINT_C = None
ITH_MA = None
LASER_BIAS_MA = None

OPERATOR = None
SAMPLE_ID = None
RUN_NOTES = None

USER_CONFIG = Path(__file__).resolve().parent / ".hbt_nist_config.json"


# ============================================================
# MANUSCRIPT PLOTTING
# ============================================================

PLOT_LAYOUT = "single"

FIGURE_SIZES = {
    "single": (4.0, 3.0),
    "double": (7.0, 4.0)
}


def configure_publication_plots():
    """Configure consistent journal-style figures."""

    if scienceplots is not None:
        plt.style.use(["science", "ieee"])
    else:
        plt.style.use("default")
        print(
            "SciencePlots unavailable; using serif fallback."
        )

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.family": "serif",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        }
    )

    # Use Matplotlib mathtext if external LaTeX is unavailable.
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["mathtext.fontset"] = "stix"


def manuscript_figure(layout=None):
    """Create a figure sized for a manuscript column."""

    if layout is None:
        layout = PLOT_LAYOUT

    if layout not in FIGURE_SIZES:
        raise ValueError(
            "Layout must be 'single' or 'double'."
        )

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZES[layout],
        constrained_layout=True
    )

    return fig, ax


def save_manuscript_figure(fig, folder, filename):
    """Save a vector PDF of the figure."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        folder / filename,
        format="pdf",
        dpi=600
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def example_simulated_arrival_plot(folder=None):
    """Optional simulated example, never measured data."""

    configure_publication_plots()

    rng = np.random.default_rng(2026)

    samples_ns = rng.exponential(
        scale=250.0,
        size=20000
    )

    fig, ax = manuscript_figure()

    ax.hist(
        samples_ns,
        bins=65,
        density=True,
        histtype="step",
        label="Simulated arrivals"
    )

    x = np.linspace(0, 1800, 300)

    ax.plot(
        x,
        np.exp(-x / 250.0) / 250.0,
        linestyle="--",
        label="Exponential model"
    )

    ax.set(
        xlabel=r"Inter-arrival time $\Delta t$ (ns)",
        ylabel=r"Probability density (ns$^{-1}$)",
        title="Simulated inter-arrival distribution"
    )

    ax.legend(frameon=False)
    sns.despine(ax=ax)

    if folder is None:
        folder = SAVE_ROOT / "plot_style_example"

    save_manuscript_figure(
        fig,
        folder,
        "simulated_interarrival_example.pdf"
    )

    return Path(folder) / "simulated_interarrival_example.pdf"


# ============================================================
# GENERAL UTILITIES
# ============================================================

def stamp():
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            allow_nan=False
        )


def write_bits(path, bits):
    """Write complete MSB-first bytes only."""

    count = len(bits) // 8 * 8

    np.packbits(
        bits[:count],
        bitorder="big"
    ).tofile(path)

    return count


# ============================================================
# TIME TAGGER ACQUISITION
# ============================================================

def drain_stream(stream, buffers, counters, journal=None):
    """Read photon timestamps and check event integrity."""

    block = stream.getData()

    timestamps = np.asarray(
        block.getTimestamps(),
        dtype=np.int64
    )

    channels = np.asarray(block.getChannels())

    if timestamps.shape != channels.shape:
        raise RuntimeError(
            "Inconsistent timestamp and channel arrays."
        )

    get_types = getattr(block, "getEventTypes", None)
    get_missed = getattr(block, "getMissedEvents", None)

    if not callable(get_types) or not callable(get_missed):
        raise RuntimeError(
            "Time Tagger driver lacks event-integrity information."
        )

    event_types = np.asarray(get_types())
    missed_tags = np.asarray(get_missed())

    if event_types.shape != timestamps.shape:
        raise RuntimeError("Invalid event-type array.")

    if missed_tags.shape != timestamps.shape:
        raise RuntimeError("Invalid missed-event array.")

    if np.any(missed_tags < 0):
        raise RuntimeError("Negative missed-event count.")

    if journal is not None and timestamps.size:
        journal_dir, sequence = journal
        destination = journal_dir / f"block_{sequence:08d}.npz"
        with destination.open("xb") as handle:
            np.savez(handle, timestamps_ps=timestamps, channels=channels,
                     event_types=event_types, missed_events=missed_tags)
        journal[1] += 1

    valid = event_types == 0

    for channel in (CH_A, CH_B):

        selected = timestamps[
            valid & (channels == channel)
        ]

        if selected.size:
            buffers[channel].append(selected.copy())
            counters[channel] += int(selected.size)

    return (
        int(missed_tags.sum()),
        int(np.count_nonzero(~valid))
    )


def collect(target=None, timeout=None, folder=None):
    """
    Acquire both SPAD channels.

    target=None and timeout=None:
        Continuous acquisition until Ctrl+C.

    A normal manual stop saves all accumulated timestamps.
    """

    if CH_A == CH_B:
        raise ValueError("SPAD channels must differ.")

    if target is not None and target < 2:
        raise ValueError("Invalid event target.")

    if timeout is not None and timeout <= 0:
        raise ValueError("Invalid timeout.")

    if TimeTagger is None:
        raise RuntimeError(
            "Swabian Time Tagger driver is not available."
        )

    buffers = {
        CH_A: [],
        CH_B: []
    }

    counters = {
        CH_A: 0,
        CH_B: 0
    }

    journal = None
    if folder is not None:
        journal_dir = Path(folder) / "raw_event_log"
        journal_dir.mkdir(exist_ok=False)
        journal = [journal_dir, 0]

    missed = 0
    invalid_types = 0
    overflows = None

    status = "complete"
    error = None

    tagger = None
    stream = None

    start = time.monotonic()

    try:

        tagger = TimeTagger.createTimeTagger(SERIAL)

        tagger.setTriggerLevel(
            CH_A,
            TRIGGER_A_V
        )

        tagger.setTriggerLevel(
            CH_B,
            TRIGGER_B_V
        )

        tagger.clearOverflows()

        stream = TimeTagger.TimeTagStream(
            tagger,
            int(STREAM_BUFFER_EVENTS),
            [CH_A, CH_B]
        )

        stream.start()

        start = time.monotonic()
        next_print = start + 5

        while target is None or counters[CH_A] < target:

            elapsed = time.monotonic() - start

            if timeout is not None and elapsed >= timeout:
                status = "timeout"
                error = "Event target not reached before timeout."
                break

            time.sleep(0.05)

            m, problems = drain_stream(
                stream,
                buffers,
                counters,
                journal
            )

            missed += m
            invalid_types += problems

            if time.monotonic() >= next_print:

                elapsed = max(
                    time.monotonic() - start,
                    0.001
                )

                print(
                    f"Live: "
                    f"A {counters[CH_A]:,} "
                    f"({counters[CH_A] / elapsed:,.1f} cps) | "
                    f"B {counters[CH_B]:,} "
                    f"({counters[CH_B] / elapsed:,.1f} cps)"
                )

                next_print = time.monotonic() + 5

    except KeyboardInterrupt:

        if target is None:
            status = "stopped_by_user"
            error = "Continuous recording stopped normally."
        else:
            status = "interrupted"
            error = "Stopped before event target."

    except Exception as exc:

        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    finally:

        if stream is not None:

            try:

                stream.stop()

                m, problems = drain_stream(
                    stream,
                    buffers,
                    counters,
                    journal
                )

                missed += m
                invalid_types += problems

            except Exception as exc:

                status = "error"
                error = f"Final stream drain failed: {exc}"

        if tagger is not None:

            try:
                overflows = int(tagger.getOverflows())
            except Exception as exc:
                status = "error"
                error = f"Overflow query failed: {exc}"

            try:
                TimeTagger.freeTimeTagger(tagger)
            except Exception as exc:
                status = "error"
                error = f"Device release failed: {exc}"

    raw = {}

    for channel in (CH_A, CH_B):

        if buffers[channel]:
            raw[channel] = np.concatenate(
                buffers[channel]
            )
        else:
            raw[channel] = np.empty(
                0,
                dtype=np.int64
            )

    if (
        status == "complete"
        and target is not None
        and counters[CH_A] < target
    ):
        status = "incomplete"
        error = "Insufficient channel-A events."

    if (
        status in {"complete", "stopped_by_user"}
        and (missed or invalid_types or overflows)
    ):
        status = "data_loss"
        error = "Lost or invalid event tags detected."

    for channel, timestamps in raw.items():

        if (
            len(timestamps) > 1
            and np.any(np.diff(timestamps) <= 0)
        ):
            status = "timestamp_error"
            error = (
                f"Non-increasing timestamps "
                f"on channel {channel}."
            )

    info = {
        "status": status,
        "error": error,
        "valid": status in {
            "complete",
            "stopped_by_user"
        },
        "raw_events": {
            str(c): len(raw[c])
            for c in raw
        },
        "target_A": target,
        "wall_seconds": round(
            time.monotonic() - start,
            3
        ),
        "acquisition_mode": (
            "continuous"
            if target is None
            else "targeted"
        ),
        "raw_event_log_blocks": journal[1] if journal else 0,
        "missed_events": missed,
        "non_photon_tags": invalid_types,
        "hardware_overflows": overflows,
        "serial": SERIAL,
        "channels": [CH_A, CH_B],
        "trigger_volts": [
            TRIGGER_A_V,
            TRIGGER_B_V
        ],
        "recorded_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "temperature_setpoint_C": TEMP_SETPOINT_C,
        "threshold_current_mA": ITH_MA,
        "laser_bias_current_mA": LASER_BIAS_MA
    }

    return raw, info


# ============================================================
# COMMON EXPOSURE WINDOW
# ============================================================

def common_window(raw, target=None, exposure_seconds=None):
    """
    Compare both channels over a consistent exposure.

    Continuous baseline:
        Use full available recording and wall-clock duration.

    Targeted acquisition:
        Use channel A's first-to-last event window.
    """

    a = np.asarray(
        raw[CH_A],
        dtype=np.int64
    )

    b = np.asarray(
        raw[CH_B],
        dtype=np.int64
    )

    if target is not None:

        a = a[:target]

        if len(a) < 2:
            return (
                a,
                np.empty(0, dtype=np.int64),
                0.0,
                None
            )

        lo = int(a[0])
        hi = int(a[-1])

        b = b[
            (b >= lo) & (b <= hi)
        ]

        return (
            a,
            b,
            (hi - lo) * 1e-12,
            (lo, hi)
        )

    duration = max(
        0.0,
        float(exposure_seconds or 0)
    )

    if len(a) == 0 and len(b) == 0:
        return (
            a,
            b,
            duration,
            (0, int(duration * 1e12))
        )

    starts = [
        int(arr[0])
        for arr in (a, b)
        if len(arr)
    ]

    lo = min(starts)

    return (
        a,
        b,
        duration,
        (lo, lo + int(duration * 1e12))
    )


# ============================================================
# PHOTON MEASUREMENT AND MANUSCRIPT FIGURES
# ============================================================

def characterize(raw, info, folder):

    a, b, seconds, limits = common_window(
        raw,
        info.get("target_A"),
        info.get("wall_seconds")
    )

    na = len(a)
    nb = len(b)

    if seconds <= 0:

        report = {
            "problem": "No positive exposure window",
            "A_count": na,
            "B_count": nb
        }

        save_json(
            folder / "measurements.json",
            report
        )

        return report

    total = na + nb

    balance = {
        "exposure_seconds": seconds,
        "common_window_ps": list(limits),
        "A_count": na,
        "B_count": nb,
        "A_cps": na / seconds,
        "B_cps": nb / seconds,
        "A_over_B": na / nb if nb else None,
        "A_percent": (
            100 * na / total
            if total else None
        ),
        "B_percent": (
            100 * nb / total
            if total else None
        ),
        "normalized_imbalance": (
            (na - nb) / total
            if total else None
        ),
        "A_above_warning_cps": (
            na / seconds > WARNING_CPS
        ),
        "B_above_warning_cps": (
            nb / seconds > WARNING_CPS
        ),
        "dead_time_violations": {}
    }

    for label, timestamps in (
        ("A", a),
        ("B", b)
    ):

        intervals = np.diff(timestamps)

        balance["dead_time_violations"][label] = (
            int(
                np.count_nonzero(
                    intervals < round(DEAD_TIME_NS * 1000)
                )
            )
            if intervals.size else None
        )

    save_json(
        folder / "measurements.json",
        balance
    )

    # --------------------------------------------------------
    # FIGURE 1: PHOTON COUNT-RATE STABILITY
    # --------------------------------------------------------

    edges = np.linspace(
        0.0,
        seconds,
        max(
            2,
            min(
                2001,
                int(np.ceil(seconds)) + 1
            )
        )
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    bin_widths = np.diff(edges)

    times_a = (
        a - limits[0]
    ) * 1e-12

    times_b = (
        b - limits[0]
    ) * 1e-12

    hist_a = (
        np.histogram(times_a, bins=edges)[0]
        / bin_widths
    )

    hist_b = (
        np.histogram(times_b, bins=edges)[0]
        / bin_widths
    )

    fig, ax = manuscript_figure()

    ax.plot(
        centers,
        hist_a,
        label=f"SPAD A (ch{CH_A})"
    )

    ax.plot(
        centers,
        hist_b,
        label=f"SPAD B (ch{CH_B})"
    )

    ax.set(
        xlabel="Elapsed time (s)",
        ylabel=r"Count rate (s$^{-1}$)",
        title="Photon count-rate stability"
    )

    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    sns.despine(ax=ax)

    save_manuscript_figure(
        fig,
        folder,
        "count_rates.pdf"
    )

    # --------------------------------------------------------
    # FIGURE 2: INTER-ARRIVAL DISTRIBUTION
    # --------------------------------------------------------

    fig, ax = manuscript_figure()

    plotted = False

    for label, timestamps in (
        ("A", a),
        ("B", b)
    ):

        if len(timestamps) <= 2:
            continue

        dt = np.diff(timestamps) * 1e-3

        if not (
            np.max(dt) > np.min(dt) > 0
        ):
            continue

        edges2 = np.geomspace(
            float(np.min(dt)),
            float(np.max(dt)) * 1.001,
            85
        )

        density, _ = np.histogram(
            dt,
            bins=edges2,
            density=True
        )

        bin_centers = np.sqrt(
            edges2[:-1] * edges2[1:]
        )

        ax.plot(
            bin_centers,
            density,
            label=f"SPAD {label}"
        )

        plotted = True

    ax.axvline(
        DEAD_TIME_NS,
        linestyle="--",
        color="black",
        label="Nominal dead time"
    )

    ax.set(
        xscale="log",
        yscale="log" if plotted else "linear",
        xlabel=r"Inter-arrival time $\Delta t$ (ns)",
        ylabel=r"Probability density (ns$^{-1}$)",
        title="Inter-arrival distributions"
    )

    ax.legend(frameon=False)
    sns.despine(ax=ax)

    save_manuscript_figure(
        fig,
        folder,
        "interarrival.pdf"
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\n--- PHOTON MEASUREMENTS ---")

    print(
        f"Exposure: {seconds:.4f} seconds"
    )

    print(
        f"SPAD A: {na:,} counts | "
        f"{na / seconds:,.2f} cps"
    )

    print(
        f"SPAD B: {nb:,} counts | "
        f"{nb / seconds:,.2f} cps"
    )

    if total:
        print(
            f"A:B split: "
            f"{balance['A_percent']:.2f}% : "
            f"{balance['B_percent']:.2f}%"
        )
    else:
        print("A:B split: N/A (no events)")

    if nb:
        print(
            f"A/B ratio: {balance['A_over_B']:.4f}"
        )
    else:
        print("A/B ratio: N/A")

    print(
        "Dead-time violations:",
        balance["dead_time_violations"]
    )

    if nb < 1000:
        print(
            "Low-count notice: SPAD B has fewer "
            "than 1,000 events."
        )

    if (
        balance["A_above_warning_cps"]
        or balance["B_above_warning_cps"]
    ):
        print(
            "WARNING: Count rate exceeds "
            "configured research threshold."
        )

    return balance


# ============================================================
# EXPLORATORY QRNG EXTRACTION
# ============================================================

def von_neumann_intervals(timestamps):
    """Compare non-overlapping adjacent arrival intervals."""

    differences = np.diff(timestamps)

    pairs = differences[
        :(len(differences) // 2) * 2
    ].reshape(-1, 2)

    return (
        pairs[:, 0] > pairs[:, 1]
    )[
        pairs[:, 0] != pairs[:, 1]
    ].astype(np.uint8)


def toeplitz_hash(bits, output_length):
    """Toeplitz hashing over GF(2)."""

    n = len(bits)
    m = output_length

    if not 0 < m <= n:
        raise ValueError("Invalid Toeplitz dimensions.")

    seed = np.unpackbits(
        np.frombuffer(
            secrets.token_bytes(
                (n + m + 6) // 8
            ),
            dtype=np.uint8
        )
    )[:n + m - 1]

    fft_length = fft.next_fast_len(
        n + m - 1
    )

    convolution = fft.irfft(
        fft.rfft(
            seed.astype(float),
            n=fft_length
        )
        *
        fft.rfft(
            bits.astype(float),
            n=fft_length
        ),
        n=fft_length
    )[
        n - 1:n - 1 + m
    ]

    rounded = np.rint(convolution)

    if np.max(
        np.abs(convolution - rounded)
    ) > 0.25:
        raise ArithmeticError(
            "FFT precision insufficient."
        )

    return (
        (
            rounded.astype(np.int64) & 1
        ).astype(np.uint8),
        seed
    )


# ============================================================
# PREDECLARED SAMPLING AND NIST ASSESSMENT
# ============================================================

# Choose these from the measurement model / independent pilot data BEFORE a run.
# These are illustrative digitizer settings, not NIST-prescribed constants.
RAW_SAMPLE_COUNT = 1_000_000
RAW_BIN_WIDTH_PS = 100_000  # 100 ns; last bin includes every interval >=25.5 us.
RAW_BITS_PER_SYMBOL = 8
SP22_SEQUENCE_BITS = 100_000
SP22_BLOCK_BITS = 2_000
SP22_ALPHA = 0.01
EA_NON_IID = None  # Absolute path to a compiled official ea_non_iid executable.
EA_TIMEOUT_S = 3600
NIST_MIN_SAMPLES = 1_000_000

# Independent HBT diagnostic; these do not change the NIST/extractor inputs.
G2_ENABLED = True
G2_BIN_WIDTH_PS = 1000  # 1 ns, illustrative; choose for detector jitter/coherence time.
G2_HALF_BINS = 100  # 201 bins centered from -100 to +100 ns by default.
G2_DELAY_PS = 0  # Subtract independently calibrated B-minus-A instrument delay.
G2_MAX_PAIRS = 10_000_000  # Fail explicitly rather than truncate a large histogram.


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_plan(provenance):
    plan = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "channels": [CH_A, CH_B],
        "sample_count": RAW_SAMPLE_COUNT,
        "bin_width_ps": RAW_BIN_WIDTH_PS,
        "bits_per_symbol": RAW_BITS_PER_SYMBOL,
        "selection": "first consecutive per-channel intervals; no cross-channel pooling",
        "mapping": "min(interval_ps // bin_width_ps, 2**bits_per_symbol - 1)",
        "extractor": "compare disjoint pairs of the SAME selected unquantized intervals; discard ties",
        "sp22_sequence_bits": SP22_SEQUENCE_BITS,
        "sp22_block_bits": SP22_BLOCK_BITS,
        "sp22_alpha": SP22_ALPHA,
        "script_sha256": sha256_file(__file__),
        "physical_setup": {
            "laser_wavelength_nm": 405, "beam_splitter": "50:50",
            "attenuation": "fixed ND; document filter and optical configuration in run_notes",
            "temperature_setpoint_C": TEMP_SETPOINT_C,
            "laser_bias_current_mA": LASER_BIAS_MA,
            "run_notes": RUN_NOTES,
        },
    }
    validate_plan(plan)
    return plan


def validate_plan(plan):
    for key in ("sample_count", "bin_width_ps", "bits_per_symbol",
                "sp22_sequence_bits", "sp22_block_bits"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported sampling-plan version")
    if not 1 <= plan["bits_per_symbol"] <= 8:
        raise ValueError("The official estimator requires 1..8 bits per byte-sized sample")
    if plan["bin_width_ps"] > np.iinfo(np.int64).max:
        raise ValueError("Timing bin width exceeds int64")
    channels = plan["channels"]
    if (len(channels) != 2 or any(type(c) is not int for c in channels)
            or channels[0] == channels[1]):
        raise ValueError("Two distinct integer channels required")
    n, m = plan["sp22_sequence_bits"], plan["sp22_block_bits"]
    if n < 100 or not (20 <= m <= n and m > 0.01 * n and n // m < 100):
        raise ValueError("SP 800-22: n>=100, 20<=M<=n, M>0.01*n, floor(n/M)<100")
    if not 0 < plan["sp22_alpha"] < 1:
        raise ValueError("Significance alpha must be between zero and one")


def timing_intervals(timestamps):
    """Validate before subtraction; never silently sort, cast floats or filter gaps."""
    t = np.asarray(timestamps)
    if t.ndim != 1 or t.dtype.kind not in "iu":
        raise ValueError("Timestamps must be a one-dimensional integer array in ps")
    if t.size and (int(t.min()) < 0 or int(t.max()) > np.iinfo(np.int64).max):
        raise ValueError("Timestamps must fit nonnegative int64")
    t = t.astype(np.int64, copy=False)
    if np.any(t[1:] <= t[:-1]):
        raise ValueError("Non-increasing timestamps; no sorting or deduplication allowed")
    return np.diff(t)


def timing_symbols(intervals, bin_width_ps, bits_per_symbol):
    """Fixed saturating quantizer. No modulo, LSB selection or fitted bins."""
    return np.minimum(intervals // bin_width_ps,
                      (1 << bits_per_symbol) - 1).astype(np.uint8)


def interval_comparison_bits(intervals):
    pairs = intervals[:len(intervals) // 2 * 2].reshape(-1, 2)
    unequal = pairs[:, 0] != pairs[:, 1]
    return (pairs[unequal, 0] > pairs[unequal, 1]).astype(np.uint8), {
        "pairs": len(pairs), "ties_discarded": int(np.count_nonzero(~unequal)),
        "unpaired_intervals": len(intervals) % 2,
    }


def sp22_pvalues(bits, block_bits):
    """SP 800-22 Rev.1a sections 2.1--2.3 formulas (selected tests only).

    Low-level formulas also accept short reference vectors for self-tests.
    The production caller validates the recommended sequence/block sizes.
    """
    bits = np.asarray(bits)
    if bits.ndim != 1 or not len(bits) or np.any((bits != 0) & (bits != 1)):
        raise ValueError("Expected a nonempty binary vector")
    n = len(bits)
    if type(block_bits) is not int or not 1 <= block_bits <= n:
        raise ValueError("Invalid frequency-test block size")
    ones = int(bits.sum())
    p_frequency = math.erfc(abs(2 * ones - n) / math.sqrt(2 * n))
    blocks = n // block_bits
    fractions = bits[:blocks * block_bits].reshape(blocks, block_bits).mean(axis=1)
    chi_square = float(4 * block_bits * np.sum((fractions - 0.5) ** 2))
    p_block = float(gammaincc(blocks / 2, chi_square / 2))
    pi = ones / n
    eligible = abs(pi - 0.5) < 2 / math.sqrt(n) and 0 < pi < 1
    if eligible:
        runs = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
        p_runs = math.erfc(abs(runs - 2 * n * pi * (1 - pi)) /
                           (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    else:
        p_runs = 0.0  # Spec convention; separately record prerequisite failure.
    return {"frequency": p_frequency, "block_frequency": p_block,
            "runs": p_runs, "runs_prerequisite_met": eligible}


def sp22_assessment(bits, plan):
    validate_plan(plan)
    n, m, alpha = (plan[k] for k in
                   ("sp22_sequence_bits", "sp22_block_bits", "sp22_alpha"))
    count = len(bits) // n
    rows = []
    for i in range(count):
        p = sp22_pvalues(bits[i * n:(i + 1) * n], m)
        rows.append({"sequence": i, "bit_offset": i * n, **p,
                     "passes": {name: p[name] >= alpha for name in
                                ("frequency", "block_frequency", "runs")}})
    summaries = {}
    for name in ("frequency", "block_frequency", "runs"):
        values = np.array([r[name] for r in rows])
        passes = int(np.count_nonzero(values >= alpha))
        summary = {"sequences": count, "passes": passes,
                   "failures": count - passes,
                   "pass_fraction": passes / count if count else None,
                   "aggregate_status": "insufficient_sequences",
                   "uniformity_p": None, "pass_fraction_interval": None}
        # 55 is NIST's minimum for meaningful p-value uniformity analysis.
        if count >= 55:
            delta = 3 * math.sqrt(alpha * (1 - alpha) / count)
            lower, upper = max(0, 1 - alpha - delta), min(1, 1 - alpha + delta)
            histogram, _ = np.histogram(values, bins=10, range=(0, 1))
            chi_square = float(np.sum((histogram - count / 10) ** 2) / (count / 10))
            uniformity = float(gammaincc(4.5, chi_square / 2))
            summary.update({"aggregate_status": "evaluated",
                            "uniformity_p": uniformity,
                            "uniformity_pass": uniformity >= 0.0001,
                            "pass_fraction_interval": [lower, upper],
                            "pass_fraction_in_interval": lower <= passes / count <= upper})
        summaries[name] = summary
    return {"scope": "three supplementary tests; NOT the full NIST STS suite",
            "status": "evaluated" if count else "insufficient_bits",
            "alpha": alpha, "sequence_bits": n, "block_bits": m,
            "complete_sequences": count, "unused_tail_bits": len(bits) % n,
            "block_frequency_unused_bits_per_sequence": n % m,
            "runs_prerequisite_failures": sum(not r["runs_prerequisite_met"] for r in rows),
            "per_sequence": rows, "summary": summaries}


def parse_non_iid_output(output, bits_per_symbol, require_bitstring=False):
    """Parse the official verbose summary, not individual estimator lines."""
    number = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
    def read(label):
        values = re.findall(r"^" + re.escape(label) + r"\s*[:=]\s*" + number + r"\s*$",
                            output, flags=re.MULTILINE)
        return float(values[-1]) if values else None
    original = read("H_original")
    bitstring = read("H_bitstring")
    if original is None or not 0 <= original <= bits_per_symbol:
        raise ValueError("Missing or invalid H_original in official estimator output")
    if bitstring is not None and not 0 <= bitstring <= 1:
        raise ValueError("Invalid H_bitstring in official estimator output")
    if require_bitstring and bitstring is None:
        raise ValueError("Missing H_bitstring for a multi-symbol input")
    # A two-symbol alphabet may omit H_bitstring, per the official tool.
    estimate = min(original, bits_per_symbol * bitstring) if bitstring is not None else original
    return {"H_original_bits_per_symbol": original,
            "H_bitstring_bits_per_bit": bitstring,
            "initial_estimate_bits_per_symbol": estimate}


def run_non_iid(sample_file, bits_per_symbol, executable=None, timeout=EA_TIMEOUT_S):
    """Run official code with no shell; retain input/executable hashes and logs."""
    sample_file = Path(sample_file).resolve()
    argv = [str(executable or "ea_non_iid"), "-i", "-a", "-v",
            sample_file.name, str(bits_per_symbol)]
    report = {"status": "not_run", "command_argv": argv,
              "cwd": str(sample_file.parent), "samples": sample_file.stat().st_size,
              "input_sha256": sha256_file(sample_file), "track": "non-IID",
              "scope": "initial sequential-data estimate, not full SP 800-90B validation"}
    if report["samples"] < NIST_MIN_SAMPLES:
        return {**report, "status": "insufficient_samples",
                "minimum_samples": NIST_MIN_SAMPLES}
    if executable is None:
        return {**report, "reason": "Set EA_NON_IID or --ea-non-iid to run the official tool"}
    resolved = shutil.which(str(executable))
    if resolved is None:
        return {**report, "status": "tool_unavailable", "reason": str(executable)}
    argv[0] = str(Path(resolved).resolve())
    report["executable_sha256"] = sha256_file(argv[0])
    stdout_path = sample_file.with_suffix(".ea_stdout.txt")
    stderr_path = sample_file.with_suffix(".ea_stderr.txt")
    # File-backed output retains partial diagnostics on timeout and avoids pipe buffering.
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(argv, cwd=sample_file.parent, stdout=stdout,
                                       stderr=stderr, timeout=timeout, check=False, shell=False)
            report["returncode"] = completed.returncode
            report["status"] = "completed" if completed.returncode == 0 else "tool_error"
        except subprocess.TimeoutExpired:
            report["status"] = "timeout"
        except OSError as exc:
            report.update(status="tool_error", reason=str(exc))
    report.update(stdout_file=stdout_path.name, stderr_file=stderr_path.name)
    if report["status"] == "completed":
        try:
            alphabet_size = len(np.unique(np.fromfile(sample_file, dtype=np.uint8)))
            report.update(parse_non_iid_output(
                stdout_path.read_text(encoding="utf-8", errors="replace"),
                bits_per_symbol, require_bitstring=alphabet_size > 2))
            # "completed" is intentionally not a validation pass.
        except ValueError as exc:
            report.update(status="unparsed_output", reason=str(exc))
    return report


def acquisition_problems(info):
    problems = []
    if info.get("valid") is not True or info.get("status") not in {"complete", "stopped_by_user"}:
        problems.append("Acquisition is not marked complete or normally stopped")
    for key in ("missed_events", "non_photon_tags", "hardware_overflows"):
        if type(info.get(key)) is not int or info[key] != 0:
            problems.append(f"{key} must be explicitly recorded as zero")
    return problems


def analyze(folder):
    """Branch the same selected intervals into raw-symbol assessment and extraction.

    Every invocation creates a new analysis directory. Original arrays and previous
    assessments are never overwritten. Older acquisitions are explicitly retrospective.
    """
    folder = Path(folder)
    info_path = folder / "acquisition.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    output = folder / ("nist_analysis_" + stamp())
    output.mkdir()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "blocked", "acquisition_sha256": sha256_file(info_path),
              "analysis_script_sha256": sha256_file(__file__), "channels": {},
              "scope": "Research comparison; not certification, quantum-origin proof or key material",
              "unassessed_90b_requirements": ["physical noise-source model and H_submitter",
                  "1000 genuine restarts with 1000 samples each", "startup and continuous health tests",
                  "environmental/operating-condition validation", "conditioning validation"],
              "sources": ["https://doi.org/10.6028/NIST.SP.800-90B",
                  "https://github.com/usnistgov/SP800-90B_EntropyAssessment",
                  "https://doi.org/10.6028/NIST.SP.800-22r1a"]}
    problems = acquisition_problems(info)
    if info.get("channels") != [CH_A, CH_B]:
        problems.append("Saved channels differ from the current configuration")
    try:
        plan_path = folder / "sampling_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        else:
            plan = sampling_plan("retrospective_legacy_run")
        validate_plan(plan)
        if plan["channels"] != info.get("channels"):
            problems.append("Sampling-plan channels differ from acquired channels")
        report["sampling_plan"] = plan
        save_json(output / "sampling_plan.json", plan)
        raw = {}
        for channel in (CH_A, CH_B):
            raw_path = folder / f"raw_ch{channel}.npy"
            timestamps = np.load(raw_path, allow_pickle=False)
            intervals = timing_intervals(timestamps)
            if info.get("raw_events", {}).get(str(channel)) != len(timestamps):
                problems.append(f"Channel {channel} event count disagrees with metadata")
            raw[channel] = (intervals, sha256_file(raw_path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"Invalid saved input: {exc}")
    if problems:
        report["problems"] = problems
        save_json(output / "nist_report.json", report)
        print("NIST analysis blocked:", "; ".join(problems))
        return output
    # No common-window crop: take the predeclared sequential prefix of each channel.
    for label, channel in (("A", CH_A), ("B", CH_B)):
        all_intervals, raw_hash = raw[channel]
        intervals = all_intervals[:plan["sample_count"]]
        symbols = timing_symbols(intervals, plan["bin_width_ps"], plan["bits_per_symbol"])
        symbol_file = output / f"raw_timing_{label}.u8"
        symbols.tofile(symbol_file)  # Exactly ONE byte per symbol, never packed bits.
        np.save(output / f"selected_intervals_{label}_ps.npy", intervals)
        bits, pairing = interval_comparison_bits(intervals)
        packed_file = output / f"interval_bits_{label}.bin"
        exported = write_bits(packed_file, bits)
        ascii_file = output / f"interval_bits_{label}.txt"
        ascii_file.write_bytes((bits + np.uint8(48)).tobytes())  # Exact bit count, no padding.
        top_bin = (1 << plan["bits_per_symbol"]) - 1
        result = {"channel": channel, "raw_timestamps_sha256": raw_hash,
                  "available_intervals": len(all_intervals), "selected_intervals": len(intervals),
                  "planned_intervals": plan["sample_count"],
                  "selection_complete": len(intervals) == plan["sample_count"],
                  "occupied_symbols": len(np.unique(symbols)),
                  "top_bin_samples": int(np.count_nonzero(symbols == top_bin)),
                  "raw_symbol_file": symbol_file.name,
                  "raw_symbol_encoding": "one uint8 per sample; low bits hold symbol",
                  "bits_generated": len(bits), "packed_bits_exported": exported,
                  "packed_tail_bits_omitted": len(bits) - exported,
                  "ascii_bits_exported": len(bits),
                  "ascii_sha256": sha256_file(ascii_file),
                  "packed_sha256": sha256_file(packed_file), **pairing}
        if len(bits):
            p1 = float(bits.mean())
            result.update(p1=p1, empirical_marginal_hmin=-math.log2(max(p1, 1 - p1)))
        else:
            result.update(p1=None, empirical_marginal_hmin=None)
        result["entropy_note"] = "Marginal bias is descriptive only; it does not size a conditioner"
        executable = EA_NON_IID if result["selection_complete"] else None
        result["sp800_90b"] = run_non_iid(symbol_file, plan["bits_per_symbol"], executable, EA_TIMEOUT_S)
        if not result["selection_complete"] and len(symbols) >= NIST_MIN_SAMPLES:
            result["sp800_90b"].update(status="incomplete_sampling_plan",
                                     reason="Planned prefix was not fully acquired")
        result["sp800_22"] = sp22_assessment(bits, plan)
        report["channels"][label] = result
        # Persist each channel before potentially running the other channel's long estimator.
        report["status"] = "partial"
        save_json(output / "nist_report.json", report)
        print(f"SPAD {label}: {len(symbols):,} raw samples | {len(bits):,} comparison bits | "
              f"90B: {result['sp800_90b']['status']} | "
              f"22: {result['sp800_22']['complete_sequences']} complete sequences")
    report["status"] = "analysis_complete"  # Does NOT mean standards compliance.
    save_json(output / "nist_report.json", report)
    print("NIST research report:", (output / "nist_report.json").resolve())
    return output


def g2_settings(provenance):
    settings = {"schema_version": 1, "enabled": G2_ENABLED,
                "bin_width_ps": G2_BIN_WIDTH_PS, "half_bins": G2_HALF_BINS,
                "delay_ps": G2_DELAY_PS, "max_pairs": G2_MAX_PAIRS,
                "provenance": provenance,
                "lag_convention": "tau_ps = t_B - t_A - delay_ps"}
    validate_g2_settings(settings)
    return settings


def validate_g2_settings(settings):
    if settings.get("schema_version") != 1 or type(settings.get("enabled")) is not bool:
        raise ValueError("Unsupported HBT settings")
    for key in ("bin_width_ps", "half_bins", "max_pairs"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"HBT {key} must be a positive integer")
    if settings["bin_width_ps"] % 2:
        raise ValueError("Use an even integer ps bin width for bins centered at zero")
    if type(settings["delay_ps"]) is not int:
        raise ValueError("HBT delay_ps must be an integer")
    if settings["half_bins"] > 100_000:
        raise ValueError("HBT histogram limited to 200001 bins")
    extent = (settings["half_bins"] + 1) * settings["bin_width_ps"] + abs(settings["delay_ps"])
    if extent > np.iinfo(np.int64).max // 2:
        raise ValueError("HBT lag range exceeds integer arithmetic limits")


def g2_overlap_area(edges_ps, duration_ps, delay_ps=0):
    """Integral of max(T - abs(tau + delay), 0) over each bin, in ps**2.

    Integrate each linear segment separately to avoid subtracting two large
    squares when the bin width is tiny compared to the exposure.
    """
    t = float(duration_ps)
    edges = np.asarray(edges_ps, dtype=float) + delay_ps
    left, right = np.clip(edges[:-1], -t, t), np.clip(edges[1:], -t, t)
    neg_l, neg_r = np.minimum(left, 0), np.minimum(right, 0)
    pos_l, pos_r = np.maximum(left, 0), np.maximum(right, 0)
    return ((neg_r - neg_l) * (t + (neg_l + neg_r) / 2)
            + (pos_r - pos_l) * (t - (pos_l + pos_r) / 2))


def g2_pair_counts(a, b, edges, delay_ps, max_pairs):
    """Count ALL A/B pairs in [left, right) lag bins, not nearest neighbors.

    Sorted integer arrays are relative to the same window start. Search only
    nearby B events, then expand in bounded chunks; never form an NxM matrix.
    """
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    total = 0
    left_shift, right_shift = int(edges[0]) + delay_ps, int(edges[-1]) + delay_ps
    width = int(edges[1]) - int(edges[0])
    limit = np.iinfo(np.int64).max
    if (max(abs(left_shift), abs(right_shift)) + max(int(a[-1]), int(b[-1])) > limit):
        raise ValueError("HBT timestamp plus lag could overflow int64")
    for start in range(0, len(a), 50_000):
        anchors = a[start:start + 50_000]
        low = np.searchsorted(b, anchors + left_shift, side="left")
        high = np.searchsorted(b, anchors + right_shift, side="left")
        lengths = high - low
        chunk_pairs = int(lengths.sum())
        total += chunk_pairs
        if total > max_pairs:
            raise ValueError("HBT pair limit exceeded; increase G2_MAX_PAIRS or narrow the lag range; no partial histogram exported")
        # Bound temporary expansion even for very dense local bursts.
        cumulative = np.concatenate(([0], np.cumsum(lengths)))
        for offset in range(0, chunk_pairs, 250_000):
            flat = np.arange(offset, min(offset + 250_000, chunk_pairs))
            anchor_index = np.searchsorted(cumulative[1:], flat, side="right")
            b_index = low[anchor_index] + flat - cumulative[anchor_index]
            lag = b[b_index] - anchors[anchor_index] - delay_ps
            index = (lag - int(edges[0])) // width
            counts += np.bincount(index, minlength=len(counts))
    return counts


def calculate_g2(timestamps_a, timestamps_b, settings, window_ps=None):
    """Stationary-rate cross-correlation estimate with finite-window correction.

    If no independent capture window is available, use the intersection of the
    two observed timestamp spans and explicitly label this estimated exposure.
    """
    validate_g2_settings(settings)
    timing_intervals(timestamps_a)
    timing_intervals(timestamps_b)
    a = np.asarray(timestamps_a, dtype=np.int64)
    b = np.asarray(timestamps_b, dtype=np.int64)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("HBT needs at least two timestamps per detector")
    if window_ps is None:
        lo, hi = max(int(a[0]), int(b[0])), min(int(a[-1]), int(b[-1]))
        window_source = "inferred_intersection_of_observed_event_spans"
    else:
        if (len(window_ps) != 2 or any(type(x) is not int for x in window_ps)
                or not 0 <= window_ps[0] < window_ps[1] <= np.iinfo(np.int64).max):
            raise ValueError("Invalid independently measured capture window")
        lo, hi = window_ps
        window_source = "explicit_capture_window"
    if hi <= lo:
        raise ValueError("No positive overlapping observation span")
    a = a[(a >= lo) & (a < hi)] - lo
    b = b[(b >= lo) & (b < hi)] - lo
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Insufficient events inside the common HBT window")
    t = hi - lo
    width, half, delay = (settings[k] for k in ("bin_width_ps", "half_bins", "delay_ps"))
    centers = np.arange(-half, half + 1, dtype=np.int64) * width
    edges = np.arange(-half, half + 2, dtype=np.int64) * width - width // 2
    area = g2_overlap_area(edges, t, delay)
    if np.any(area <= 0):
        raise ValueError("HBT lag range lies outside the available overlap; reduce span/delay")
    counts = g2_pair_counts(a, b, edges, delay, settings["max_pairs"])
    expected = (len(a) / float(t)) * (len(b) / float(t)) * area
    normalized = counts / expected
    # Garwood intervals on counts; rates are treated as fixed. These are
    # approximate g2 uncertainty bands, not a full correlated-data inference.
    lower = np.zeros(len(counts), dtype=float)
    nonzero = counts > 0
    lower[nonzero] = 0.5 * chi2.ppf(0.025, 2 * counts[nonzero]) / expected[nonzero]
    upper = 0.5 * chi2.ppf(0.975, 2 * (counts + 1)) / expected
    warnings = ["Assumes stationary rates; no dark-count, afterpulse, crosstalk or jitter correction.",
                "Poisson count intervals omit rate-estimation uncertainty and pair dependence.",
                "A flat curve does not exclude correlations narrower than the instrument response."]
    if window_ps is None:
        warnings.append("Exposure is inferred from event endpoints, not measured acquisition gates; finite-sample boundary bias is possible.")
    if expected[half] < 25:
        warnings.append("Fewer than 25 expected accidental pairs in the central bin; collect longer before interpreting its value.")
    if max(abs(int(edges[0]) + delay), abs(int(edges[-1]) + delay)) > t / 10:
        warnings.append("Lag range is a substantial fraction of exposure; interpret edge estimates cautiously.")
    summary = {"status": "estimated", "settings": settings,
               "window_ps": [lo, hi], "window_source": window_source,
               "duration_seconds": t * 1e-12, "A_count": len(a), "B_count": len(b),
               "A_cps": len(a) / (t * 1e-12), "B_cps": len(b) / (t * 1e-12),
               "lag_convention": "tau_ps = t_B - t_A - delay_ps",
               "bin_convention": "left-inclusive, right-exclusive for every bin",
               "normalization": "C_i / [(N_A/T)(N_B/T) integral_bin max(T-|tau+delay|,0) d_tau]",
               "total_pairs_in_histogram": int(counts.sum()),
               "central_bin": {"edges_ps": [int(edges[half]), int(edges[half + 1])],
                   "counts": int(counts[half]), "expected_accidental_counts": float(expected[half]),
                   "g2_bin_average": float(normalized[half]),
                   "approximate_poisson_95_interval": [float(lower[half]), float(upper[half])]},
               "warnings": warnings}
    return summary, {"lag_ps": centers, "left_edge_ps": edges[:-1], "right_edge_ps": edges[1:],
                     "coincidences": counts, "expected_accidentals": expected,
                     "g2": normalized, "poisson95_lower": lower, "poisson95_upper": upper}


def analyze_g2(folder, use_saved_settings=True):
    """Independent HBT output folder. Never writes NIST, raw or extracted files."""
    folder = Path(folder)
    if not G2_ENABLED:
        return None
    output = folder / ("hbt_g2_" + stamp())
    output.mkdir()
    report = {"status": "blocked", "analysis_script_sha256": sha256_file(__file__),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    try:
        info_path = folder / "acquisition.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        report["acquisition_sha256"] = sha256_file(info_path)
        problems = acquisition_problems(info)
        if info.get("channels") != [CH_A, CH_B]:
            problems.append("Saved channels differ from the current configuration")
        if problems:
            raise ValueError("; ".join(problems))
        settings_path = folder / "g2_plan.json"
        if use_saved_settings and settings_path.exists():
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        else:
            settings = g2_settings("retrospective_current_settings")
        validate_g2_settings(settings)
        report["settings"] = settings
        if not settings["enabled"]:
            report["status"] = "disabled_in_saved_plan"
            save_json(output / "g2_report.json", report)
            return output
        arrays = []
        for channel in (CH_A, CH_B):
            path = folder / f"raw_ch{channel}.npy"
            values = np.load(path, allow_pickle=False)
            if info.get("raw_events", {}).get(str(channel)) != len(values):
                raise ValueError(f"Channel {channel} event count disagrees with metadata")
            report[f"raw_ch{channel}_sha256"] = sha256_file(path)
            arrays.append(values)
        summary, columns = calculate_g2(*arrays, settings,
                                       window_ps=info.get("capture_window_ps"))
        report.update(summary)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        report["problem"] = str(exc)
        save_json(output / "g2_report.json", report)
        print("HBT analysis blocked:", exc)
        return output
    save_json(output / "g2_report.json", report)
    # Keep integer ps delays and pair counts exact in the CSV.
    import csv
    with (output / "g2_histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(col.tolist() for col in columns.values())))
    np.savez(output / "g2_histogram.npz", **columns)
    try:
        fig, ax = manuscript_figure()
        lag_ns = columns["lag_ps"] * 1e-3
        ax.step(lag_ns, columns["g2"], where="mid", label="A/B correlation")
        ax.fill_between(lag_ns, columns["poisson95_lower"], columns["poisson95_upper"],
                        step="mid", alpha=0.2, label="Approx. 95% count interval")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set(xlabel=r"Corrected delay $\tau$ (ns)", ylabel=r"$g^{(2)}_{AB}(\tau)$",
               title="HBT cross-correlation")
        ax.legend(frameon=False)
        save_manuscript_figure(fig, output, "g2_correlation.pdf")
    except (OSError, RuntimeError, ValueError) as exc:
        report["plot_error"] = str(exc)
        save_json(output / "g2_report.json", report)
        plt.close("all")
    print("HBT correlation saved to:", output.resolve())
    return output


def nist_selftest():
    """Published short examples verify formulas; they are not production samples."""
    bits = np.array(list("1011010101"), dtype=np.uint8)
    assert abs(sp22_pvalues(bits, 3)["frequency"] - 0.527089) < 0.000001
    bits = np.array(list("1001101011"), dtype=np.uint8)
    assert abs(sp22_pvalues(bits, 3)["runs"] - 0.147232) < 0.000001
    bits = np.array(list("0110011010"), dtype=np.uint8)
    assert abs(sp22_pvalues(bits, 3)["block_frequency"] - 0.801252) < 0.000001
    np.testing.assert_array_equal(timing_symbols(np.array([1, 99, 100, 25500, 99999]), 100, 8),
                                  [0, 0, 1, 255, 255])
    estimate = parse_non_iid_output("H_original: 7.0\nH_bitstring: 0.5\n", 8, True)
    assert estimate["initial_estimate_bits_per_symbol"] == 4.0
    print("Offline NIST self-test PASSED: three published examples, quantizer and estimator parsing.")


def main(argv=None):
    """CLI overrides are optional; no arguments retains the Spyder-friendly menu."""
    global MODE, ANALYZE_FOLDER, SHOW_FIGURES, EA_NON_IID
    global RAW_SAMPLE_COUNT, RAW_BIN_WIDTH_PS, RAW_BITS_PER_SYMBOL
    global SP22_SEQUENCE_BITS, SP22_BLOCK_BITS
    global G2_ENABLED, G2_BIN_WIDTH_PS, G2_HALF_BINS, G2_DELAY_PS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["menu", "selftest", "short", "full", "continuous", "analyze", "g2"], default=MODE)
    parser.add_argument("--folder", type=Path, help="Saved acquisition folder for --mode analyze or g2")
    parser.add_argument("--no-show", action="store_true", help="Save PDFs without opening windows")
    parser.add_argument("--ea-non-iid", help="Official ea_non_iid executable; not run unless specified")
    parser.add_argument("--samples", type=int, default=RAW_SAMPLE_COUNT)
    parser.add_argument("--bin-width-ps", type=int, default=RAW_BIN_WIDTH_PS)
    parser.add_argument("--bits-per-symbol", type=int, default=RAW_BITS_PER_SYMBOL)
    parser.add_argument("--sequence-bits", type=int, default=SP22_SEQUENCE_BITS)
    parser.add_argument("--block-bits", type=int, default=SP22_BLOCK_BITS)
    parser.add_argument("--no-g2", action="store_true", help="Skip independent HBT processing")
    parser.add_argument("--g2-bin-ps", type=int, default=G2_BIN_WIDTH_PS)
    parser.add_argument("--g2-half-bins", type=int, default=G2_HALF_BINS)
    parser.add_argument("--g2-delay-ps", type=int, default=G2_DELAY_PS,
                        help="Subtract calibrated B-minus-A delay; use --mode g2 to reanalyze with current settings")
    args = parser.parse_args(argv)
    MODE = args.mode
    ANALYZE_FOLDER = args.folder if args.folder is not None else ANALYZE_FOLDER
    SHOW_FIGURES = SHOW_FIGURES and not args.no_show
    EA_NON_IID = args.ea_non_iid or EA_NON_IID
    RAW_SAMPLE_COUNT, RAW_BIN_WIDTH_PS = args.samples, args.bin_width_ps
    RAW_BITS_PER_SYMBOL = args.bits_per_symbol
    SP22_SEQUENCE_BITS, SP22_BLOCK_BITS = args.sequence_bits, args.block_bits
    G2_ENABLED = G2_ENABLED and not args.no_g2
    G2_BIN_WIDTH_PS, G2_HALF_BINS, G2_DELAY_PS = args.g2_bin_ps, args.g2_half_bins, args.g2_delay_ps
    g2_settings("configuration_check")
    sampling_plan("configuration_check")
    if MODE == "menu":
        experiment_menu()
    else:
        run_mode(MODE)


# ============================================================
# PERSISTENT CONFIGURATION
# ============================================================

def load_user_config():

    global SAVE_ROOT
    global OPERATOR
    global SAMPLE_ID
    global RUN_NOTES

    try:

        if USER_CONFIG.exists():

            data = json.loads(
                USER_CONFIG.read_text(
                    encoding="utf-8"
                )
            )

            if data.get("save_root"):
                SAVE_ROOT = Path(
                    data["save_root"]
                ).expanduser()

            OPERATOR = data.get(
                "operator",
                OPERATOR
            )

            SAMPLE_ID = data.get(
                "sample_id",
                SAMPLE_ID
            )

            RUN_NOTES = data.get(
                "run_notes",
                RUN_NOTES
            )

    except Exception:
        pass


def save_user_config():

    data = {
        "save_root": str(SAVE_ROOT),
        "operator": OPERATOR,
        "sample_id": SAMPLE_ID,
        "run_notes": RUN_NOTES
    }

    try:

        USER_CONFIG.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    except Exception:
        pass


def edit_metadata():

    global OPERATOR
    global SAMPLE_ID
    global RUN_NOTES
    global TEMP_SETPOINT_C
    global ITH_MA
    global LASER_BIAS_MA

    print("\n--- EXPERIMENT METADATA ---")

    def text_field(name, current):

        response = input(
            f"{name} [{current or ''}]: "
        ).strip()

        return response if response else current

    def float_field(name, current):

        response = input(
            f"{name} [{current if current is not None else ''}]: "
        ).strip()

        if not response:
            return current

        try:
            return float(response)
        except ValueError:
            print("Invalid number; unchanged.")
            return current

    OPERATOR = text_field(
        "Operator",
        OPERATOR
    )

    SAMPLE_ID = text_field(
        "Sample ID",
        SAMPLE_ID
    )

    RUN_NOTES = text_field(
        "Run notes",
        RUN_NOTES
    )

    TEMP_SETPOINT_C = float_field(
        "TEC setpoint (C)",
        TEMP_SETPOINT_C
    )

    ITH_MA = float_field(
        "Measured threshold current (mA)",
        ITH_MA
    )

    LASER_BIAS_MA = float_field(
        "Laser bias current (mA)",
        LASER_BIAS_MA
    )

    save_user_config()


def set_save_folder():

    global SAVE_ROOT

    print(
        "\nCurrent save folder:",
        SAVE_ROOT
    )

    response = input(
        "New path (Enter = keep, D = Downloads): "
    ).strip()

    if not response:
        return

    if response.upper() == "D":
        candidate = (
            Path.home() / "Downloads" / "qrng_data"
        )
    else:
        candidate = Path(
            response
        ).expanduser()

    candidate.mkdir(
        parents=True,
        exist_ok=True
    )

    SAVE_ROOT = candidate

    save_user_config()

    print(
        "Save folder updated:",
        SAVE_ROOT
    )


# ============================================================
# OFFLINE SELF-TEST
# ============================================================

def selftest():

    from scipy.linalg import toeplitz

    rng = np.random.default_rng(2026)

    for _ in range(8):

        x = rng.integers(
            0,
            2,
            47,
            dtype=np.uint8
        )

        y, seed = toeplitz_hash(
            x,
            23
        )

        dense = toeplitz(
            seed[46:46 + 23],
            seed[46::-1]
        )

        assert np.array_equal(
            y,
            (dense @ x) % 2
        )

    sample = np.array(
        [0, 5, 8, 10, 14]
    )

    expected = np.array(
        [1, 0],
        dtype=np.uint8
    )

    assert np.array_equal(
        von_neumann_intervals(sample),
        expected
    )

    print(
        "Offline self-test PASSED: "
        "Toeplitz and interval comparison."
    )


# ============================================================
# SAVED RUNS AND REANALYSIS
# ============================================================

def find_saved_runs():

    if not SAVE_ROOT.exists():
        return []

    return sorted(
        (
            p
            for p in SAVE_ROOT.iterdir()
            if (
                p.is_dir()
                and (p / "acquisition.json").is_file()
            )
        ),
        key=lambda p: p.name,
        reverse=True
    )


def choose_saved_run():

    if ANALYZE_FOLDER is not None:
        return Path(ANALYZE_FOLDER)

    runs = find_saved_runs()

    if not runs:
        print(
            "No saved runs found in",
            SAVE_ROOT.resolve()
        )
        return None

    print("\nSaved experiment folders:")

    for i, folder in enumerate(
        runs[:8],
        start=1
    ):

        print(
            f"{i}. {folder.name}"
        )

    response = input(
        "Choose 1-8 (Enter = latest), "
        "or enter folder path: "
    ).strip()

    if not response:
        return runs[0]

    if response.isdigit():

        index = int(response)

        if 1 <= index <= min(
            len(runs),
            8
        ):
            return runs[index - 1]

        return None

    folder = Path(
        response
    ).expanduser()

    if not (
        folder / "acquisition.json"
    ).is_file():

        print(
            "acquisition.json not found."
        )

        return None

    return folder


def analyze_saved_run(folder=None):

    if folder is None:
        folder = choose_saved_run()

    if folder is None:
        return

    folder = Path(folder)

    info = json.loads(
        (folder / "acquisition.json").read_text(
            encoding="utf-8"
        )
    )

    if info.get("channels") != [CH_A, CH_B]:
        print(
            "Saved channel settings differ "
            "from the current configuration."
        )
        return

    raw = {
        ch: np.load(
            folder / f"raw_ch{ch}.npy",
            allow_pickle=False
        )
        for ch in (CH_A, CH_B)
    }

    configure_publication_plots()

    analyze(folder)
    analyze_g2(folder)
    characterize(raw, info, folder)

    print(
        "Analysis saved to:",
        folder.resolve()
    )


# ============================================================
# RUN ACQUISITION
# ============================================================

def run_acquisition(target=None, timeout=None):

    global SAVE_ROOT

    load_user_config()
    configure_publication_plots()

    SAVE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    folder = SAVE_ROOT / stamp()

    folder.mkdir()
    save_json(folder / "sampling_plan.json", sampling_plan("before_acquisition"))
    save_json(folder / "g2_plan.json", g2_settings("before_acquisition"))

    print(
        "\nRecording BOTH SPAD channels."
    )

    print(
        "Experiment folder:",
        folder.resolve()
    )

    if target is None:

        print(
            "\nCONTINUOUS BASELINE RECORDING"
        )

        print(
            "No photon-count target or time limit."
        )

        print(
            "Press Ctrl+C ONCE in the Spyder console "
            "to stop and save."
        )

        print(
            "Do not use Stop/Restart Kernel."
        )

    acquisition_start = datetime.now(
        timezone.utc
    ).isoformat()

    raw, info = collect(
        target,
        timeout,
        folder
    )

    acquisition_end = datetime.now(
        timezone.utc
    ).isoformat()

    # Archive raw timestamp arrays before analysis.

    for channel, timestamps in raw.items():

        np.save(
            folder / f"raw_ch{channel}.npy",
            timestamps
        )

    info.update({
        "acquisition_start_utc": acquisition_start,
        "acquisition_end_utc": acquisition_end,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "operator": OPERATOR,
        "sample_id": SAMPLE_ID,
        "run_notes": RUN_NOTES,
        "save_root": str(SAVE_ROOT)
    })

    save_json(
        folder / "acquisition.json",
        info
    )

    print(
        "\nAcquisition status:",
        info["status"]
    )

    if info["error"]:
        print(
            "Details:",
            info["error"]
        )

    analyze(folder)  # Also writes a blocked report for invalid acquisitions.
    analyze_g2(folder)
    characterize(raw, info, folder)

    print(
        "\nEXPERIMENT SAVED:",
        folder.resolve()
    )


# ============================================================
# RUN MODES
# ============================================================

def run_mode(mode):

    if mode == "selftest":

        selftest()
        nist_selftest()

    elif mode == "short":

        run_acquisition(
            SHORT_A_EVENTS,
            SHORT_TIMEOUT_S
        )

    elif mode == "full":

        run_acquisition(
            FULL_A_EVENTS,
            FULL_TIMEOUT_S
        )

    elif mode == "analyze":

        analyze_saved_run()

    elif mode == "g2":

        folder = choose_saved_run()
        if folder is not None:
            configure_publication_plots()
            analyze_g2(folder, use_saved_settings=False)

    elif mode == "continuous":

        run_acquisition(
            target=None,
            timeout=None
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )


# ============================================================
# INTERACTIVE EXPERIMENT MENU
# ============================================================

def experiment_menu():

    load_user_config()

    while True:

        print(
            "\n" + "=" * 55
        )

        print(
            "405 nm LASER PHOTON-COUNTING EXPERIMENT"
        )

        print(
            "=" * 55
        )

        print(
            "1. Self-test          Check processing math"
        )

        print(
            f"2. Short run          "
            f"{SHORT_A_EVENTS:,} A events"
        )

        print(
            f"3. Full run           "
            f"{FULL_A_EVENTS:,} A events"
        )

        print(
            "4. Analyze data       Open saved experiment"
        )

        print(
            "5. Edit metadata      Operator and run settings"
        )

        print(
            "6. Set save folder    Change output location"
        )

        print(
            "7. Continuous baseline (manual stop)"
        )

        print(
            "8. HBT g2 only        Reanalyze with current HBT settings"
        )

        print(
            "0. Exit"
        )

        print(
            "-" * 55
        )

        print(
            f"SPAD A = channel {CH_A}; "
            f"SPAD B = channel {CH_B}"
        )

        print(
            "Laser settings are controlled manually."
        )

        print(
            "Save folder:",
            SAVE_ROOT
        )

        choice = input(
            "\nSelect 0-8: "
        ).strip()

        if choice in {
            "0",
            "q",
            "quit",
            "exit"
        }:

            print(
                "Experiment menu closed."
            )

            return

        if choice == "5":

            edit_metadata()
            continue

        if choice == "6":

            set_save_folder()
            continue

        choices = {
            "1": "selftest",
            "2": "short",
            "3": "full",
            "4": "analyze",
            "7": "continuous",
            "8": "g2"
        }

        mode = choices.get(
            choice
        )

        if mode is None:

            print(
                "Choose 0, 1, 2, 3, 4, 5, 6, 7, or 8."
            )

            continue

        try:

            run_mode(mode)

        except KeyboardInterrupt:

            print(
                "Interrupted. Returning to menu."
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
            ArithmeticError
        ) as exc:

            print(
                f"Experiment error: "
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()
