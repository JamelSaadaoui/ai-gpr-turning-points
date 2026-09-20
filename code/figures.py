"""Journal figures: all titles and explanatory notes belong outside the image.

Run from the notebook or import ``make_all_figures``. PNG, PDF, and SVG files
are generated from the same Matplotlib figure. The single unavailable October
2025 CPI observation used to deflate oil prices is log-linearly interpolated.
"""

from __future__ import annotations

import json
from io import BytesIO
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from PIL import Image

NAVY = "#183A56"
TEAL = "#218C8D"
RUST = "#AF5F3D"
GRAY = "#66737C"
SHOCK_SCALE = 100 * np.log(1.10)
LABELS = {
    "gpr_ai": "Aggregate AI-GPR",
    "military_conflict": "Military conflict",
    "diplomatic_tension": "Diplomatic tension",
    "terrorism": "Terrorism",
    "civil_war": "Civil war",
    "nuclear_threat": "Nuclear threat",
    "coup": "Coup / regime change",
    "sanctions": "Sanctions",
    "other": "Other events",
}
PRIMARY = ("military_conflict", "diplomatic_tension", "nuclear_threat")
OUTCOME_LABELS = {"lbrent": "Brent", "lwti": "WTI"}
MECHANISM_LABELS = {"lwip": "World industrial production",
                    "production_growth": "Oil-production growth"}


def style():
    """Set a restrained, reproducible scientific plotting style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "P052", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#92999D",
        "axes.linewidth": 0.65,
        "axes.labelcolor": "#202A31",
        "text.color": "#202A31",
        "xtick.color": "#434D55",
        "ytick.color": "#434D55",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def polish(ax, zero=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDE3E6", linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    if zero:
        ax.axhline(0, color="#7A858C", lw=0.8, zorder=2)
    ax.margins(x=0)


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Structural enforcement complements visual inspection.
    assert not fig._suptitle, "A figure cannot contain a title."
    assert all(not ax.get_title() for ax in fig.axes), "Subplot titles are forbidden."
    for ext in ("png", "pdf", "svg"):
        # Complete rendering in memory before touching the public filename.
        # A concurrent document builder must never encounter a half-written PNG.
        buffer = BytesIO()
        fig.savefig(buffer, format=ext, bbox_inches="tight", facecolor="white")
        payload = buffer.getvalue()
        if ext == "png":
            with Image.open(BytesIO(payload)) as preview:
                preview.verify()
            with Image.open(BytesIO(payload)) as preview:
                preview.load()
        elif ext == "pdf":
            assert payload.startswith(b"%PDF-") and payload.rstrip().endswith(b"%%EOF")
        else:
            assert b"</svg>" in payload
        with tempfile.NamedTemporaryFile(mode="wb", dir=out, prefix=f".{name}_",
                                         suffix=f".{ext}.tmp", delete=False) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        destination = out / f"{name}.{ext}"
        os.replace(temporary_path, destination)
        assert destination.stat().st_size == len(payload)
        if ext == "png":
            with Image.open(destination) as final_image:
                final_image.verify()
    plt.close(fig)


def panel_marker(ax, number):
    """Letters identify panels without adding narrative text or titles."""
    ax.text(0.025, 0.94, f"({chr(97 + number)})", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color="#4D5962")


def data_panels(data, out, labels=None):
    labels = labels or LABELS
    style()
    variables = [
        ("lwti", "Log real WTI", False),
        ("lbrent", "Log real Brent", False),
        ("lwip", "Log industrial production", False),
        ("lgop", "Log oil production", False),
        ("production_growth", "Oil-production growth", False),
    ] + [(k, v, True) for k, v in labels.items()]
    fig, axes = plt.subplots(5, 3, figsize=(10.8, 11.0))
    date = pd.to_datetime(data["date"])
    for i, (ax, (column, label, transform)) in enumerate(zip(axes.flat, variables)):
        y = np.log1p(data[column]) if transform else data[column]
        color = NAVY if i < 5 else TEAL
        ax.plot(date, y, color=color, lw=1.1)
        ax.set_ylabel(label, labelpad=7)
        panel_marker(ax, i)
        polish(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(date.min(), date.max())
    for i, ax in enumerate(axes.flat):
        if i >= len(variables):
            ax.set_visible(False)
        elif i + 3 >= len(variables):
            ax.set_xlabel("Year")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.045, top=0.995,
                        hspace=0.42, wspace=0.38)
    save(fig, out, "fig01_all_variables")

    fig, axes = plt.subplots(3, 3, figsize=(10.8, 7.4))
    for i, (ax, (column, label)) in enumerate(zip(axes.flat, labels.items())):
        # Inputs are a complete monthly calendar; NaNs intentionally break lines.
        instrument = data[column].diff().diff()
        ax.plot(date, instrument, color=NAVY, lw=0.85)
        ax.set_ylabel(label, labelpad=7)
        panel_marker(ax, i)
        polish(ax, zero=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(date.min(), date.max())
        if i >= 6:
            ax.set_xlabel("Year")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.065, top=0.995,
                        hspace=0.35, wspace=0.40)
    save(fig, out, "fig02_turning_points")


def normalize_result_columns(frame):
    """Allow transparent compatibility with explicitly named estimator exports."""
    frame = frame.copy()
    alternatives = {
        "event": ["variable", "raw", "category"],
        "h": ["horizon"],
        "beta": ["beta_log_gpr", "coefficient", "coef"],
        "se": ["hc3_se"],
    }
    for target, candidates in alternatives.items():
        if target not in frame:
            for candidate in candidates:
                if candidate in frame:
                    frame = frame.rename(columns={candidate: target})
                    break
    if "event" not in frame and "series" in frame:
        reverse = {v: k for k, v in LABELS.items()}
        frame["event"] = frame["series"].map(reverse).fillna(frame["series"])
    return frame


def irf_panel(ax, result, color=NAVY, bands=True, label=None):
    q = result.sort_values("h")
    h = q["h"].to_numpy(float)
    coefficient = q["beta"].to_numpy(float) * SHOCK_SCALE
    se = q["se"].to_numpy(float) * SHOCK_SCALE
    if bands:
        ax.fill_between(h, coefficient - 1.959963984540054 * se,
                        coefficient + 1.959963984540054 * se,
                        color=color, alpha=0.12, lw=0, zorder=1)
        ax.fill_between(h, coefficient - 1.6448536269514722 * se,
                        coefficient + 1.6448536269514722 * se,
                        color=color, alpha=0.20, lw=0, zorder=2)
    ax.plot(h, coefficient, color=color, lw=1.9, zorder=4, label=label)
    polish(ax, zero=True)
    ax.set_xlim(0, h.max())
    ax.set_xticks(np.arange(0, h.max() + 1, 12))


def band_legend(fig):
    handles = [Line2D([], [], color=NAVY, lw=1.9, label="Estimate"),
               Patch(facecolor=NAVY, alpha=0.32, label="90% interval"),
               Patch(facecolor=NAVY, alpha=0.12, label="95% interval")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.012),
               ncol=3, handlelength=2.1, columnspacing=2.0)


def separate_irfs(results, out, labels=None):
    labels = labels or LABELS
    results = normalize_result_columns(results)
    for outcome in ("lbrent", "lwti"):
        fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.1))
        for i, (ax, (event, label)) in enumerate(zip(axes.flat, labels.items())):
            q = results.loc[(results["outcome"] == outcome) &
                            (results["event"] == event)]
            assert len(q), f"Missing baseline results for {outcome} / {event}."
            irf_panel(ax, q)
            ax.set_ylabel(label, labelpad=7)
            panel_marker(ax, i)
            if i >= 6:
                ax.set_xlabel("Horizon (months)")
        band_legend(fig)
        fig.supylabel("Oil price response (log points × 100)", x=0.006, fontsize=11)
        fig.subplots_adjust(left=0.105, right=0.995, bottom=0.07, top=0.95,
                            hspace=0.37, wspace=0.43)
        save(fig, out, f"fig03_{OUTCOME_LABELS[outcome].lower()}_separate_irfs")


def joint_irfs(results, out, labels=None):
    labels = labels or LABELS
    results = normalize_result_columns(results)
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.9), sharex=True)
    for i, event in enumerate(PRIMARY):
        row_axes = []
        for j, outcome in enumerate(("lbrent", "lwti")):
            ax = axes[i, j]
            q = results.loc[(results["outcome"] == outcome) &
                            (results["event"] == event)]
            assert len(q), f"Missing joint results for {outcome} / {event}."
            irf_panel(ax, q)
            ax.set_ylabel(f"{labels[event]}\n{OUTCOME_LABELS[outcome]}", labelpad=8)
            panel_marker(ax, i * 2 + j)
            row_axes.append(ax)
            if i == 2:
                ax.set_xlabel("Horizon (months)")
        lo = min(ax.get_ylim()[0] for ax in row_axes)
        hi = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(lo, hi)
    band_legend(fig)
    fig.supylabel("Oil price response (log points × 100)", x=0.005, fontsize=11)
    fig.subplots_adjust(left=0.125, right=0.995, bottom=0.065, top=0.95,
                        hspace=0.27, wspace=0.28)
    save(fig, out, "fig04_joint_irfs")


def mechanism_irfs(results, out, labels=None):
    """Supplementary joint responses of activity and physical oil production."""
    labels = labels or LABELS
    results = normalize_result_columns(results)
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.9), sharex=True)
    colors = {"lwip": NAVY, "production_growth": NAVY}
    for i, event in enumerate(PRIMARY):
        for j, outcome in enumerate(("lwip", "production_growth")):
            ax = axes[i, j]
            q = results.loc[(results["outcome"] == outcome) &
                            (results["event"] == event)]
            assert len(q), f"Missing mechanism results for {outcome} / {event}."
            irf_panel(ax, q, color=colors[outcome])
            ax.set_ylabel(f"{labels[event]}\n{MECHANISM_LABELS[outcome]}", labelpad=8)
            panel_marker(ax, i * 2 + j)
            if i == 2:
                ax.set_xlabel("Horizon (months)")
    band_legend(fig)
    fig.supylabel("Response to a 10% index increase (log points × 100)", x=0.005, fontsize=11)
    fig.subplots_adjust(left=0.145, right=0.995, bottom=0.065, top=0.95,
                        hspace=0.27, wspace=0.30)
    save(fig, out, "figS01_mechanism_irfs")


def make_all_figures(data, results, out, labels=None):
    """Create paper figures from in-memory estimates computed in the same run.

    ``results`` must contain the DataFrame ``baseline_iv``. The common
    interface uses event, outcome, h, beta and se. An alias adapter makes column
    names explicit and does not change values or select significance.
    """
    labels = labels or LABELS
    style()
    data_panels(data, out, labels)
    baseline = results["baseline_iv"]
    separate_irfs(baseline.loc[baseline["model"] == "single"], out, labels)
    joint_irfs(baseline.loc[baseline["model"] == "joint"], out, labels)
    if "mechanism_iv" in results and len(results["mechanism_iv"]):
        mechanism_irfs(results["mechanism_iv"], out, labels)
    captions = {
        "fig01_all_variables": {
            "title": "Variables used in the analysis",
            "notes": "Monthly observations on a complete calendar. The unavailable October 2025 CPI deflator is filled by log-linear interpolation; no oil-price observation is interpolated. Panels (a)–(e) show log real WTI, log real Brent, log world industrial production, log world oil production, and oil-production growth (the first difference of log production). Panels (f)–(n) show log(1 + index) for aggregate AI-GPR, military conflict, diplomatic tension, terrorism, civil war, nuclear threats, coups or regime changes, sanctions, and other events. Each panel has its own vertical scale."
        },
        "fig02_turning_points": {
            "title": "Geopolitical turning-point instruments",
            "notes": "Second differences of the raw aggregate and category-specific AI-GPR indices, Z_t = G_t − 2G_{t−1} + G_{t−2}. The filter is applied before estimation and is not a threshold-based event selection. Panel order follows the nine geopolitical categories in Figure 1. Vertical scales differ across panels."
        },
        "fig03_brent_separate_irfs": {
            "title": "Real Brent responses to geopolitical turning-point variation",
            "notes": "Separate lag-augmented IV local projections for the aggregate AI-GPR index and each event category. Responses equal 100 × log(1.10) × the estimated coefficient and therefore express log-price changes multiplied by 100 for a 10% rise in 1 + index. Dark and light shading indicate pointwise 90% and 95% heteroskedasticity-robust HC3 confidence intervals without a HAC correction. Horizons are months after the instrument date. The specification includes three lags of the log oil price, two lags of log industrial production, two lags of oil-production growth, and two lags of the relevant log(1 + index). Panel order matches Figure 2; vertical scales differ."
        },
        "fig03_wti_separate_irfs": {
            "title": "Real WTI responses to geopolitical turning-point variation",
            "notes": "The specification, panel order, normalization, and pointwise confidence intervals match the Brent figure, with real WTI as the outcome."
        },
        "fig04_joint_irfs": {
            "title": "Joint responses to military conflict diplomatic tension and nuclear threats",
            "notes": "Joint lag-augmented IV local projections instrument the three log(1 + category index) treatments with their corresponding raw second differences. Rows show military conflict, diplomatic tension, and nuclear threats; the left column shows real Brent and the right column real WTI. All six lagged geopolitical regressors are included. Responses and heteroskedasticity-robust HC3 confidence intervals use the same normalization as the separate-model figures. Vertical scales are common within each row."
        },
        "figS01_mechanism_irfs": {
            "title": "World activity and oil-production responses to principal geopolitical turning points",
            "notes": "Joint lag-augmented IV local projections. Rows show military conflict, diplomatic tension, and nuclear threats. The left column reports log world industrial production; the right column reports the first difference of log global oil production. The dependent variable receives three lags, while the other oil-market fundamentals, real Brent, and each geopolitical treatment receive two lags. Responses equal 100 × log(1.10) × the coefficient. Shading gives pointwise 90% and 95% HC3 intervals. These estimates provide channel evidence but are not a formal mediation decomposition."
        },
    }
    Path(out, "figure_captions.json").write_text(json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")
    return captions
