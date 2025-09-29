# FILE: pretty_plots.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

def human_int(x, _pos=None):
    return f"{int(x):,}"

def pick_colors(methods):
    cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {m: cyc[i % len(cyc)] for i, m in enumerate(sorted(methods))}

def smooth_series(y: pd.Series, window: int):
    """Centered rolling median; falls back gracefully with few points."""
    if window <= 1 or len(y) <= 2:
        return y.values
    # force odd window, clamp to length
    w = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    w = max(w, 3)
    return y.rolling(window=w, center=True, min_periods=1).median().values

def style_axes(ax, ylabel, label_size: int, tick_size: int):
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def make_plots(df, outdir, window, tick_size, legend_size, label_size, fig_width, fig_height, dpi):  

    os.makedirs(outdir, exist_ok=True)

    needed = {"method", "dataset_size", "avg_lead_time", "fp_rate_per_day"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset_size"] = pd.to_numeric(df["dataset_size"])
    df.sort_values(["method", "dataset_size"], inplace=True)

    methods = df["method"].unique()
    cmap = pick_colors(methods)

    # Optional subtitle from shared meta
    #meta_cols = ["epochs", "batch", "ema_window", "seed", "label_mode", "horizon_min"]
    #meta_parts = []
    # for c in meta_cols:
    #     if c in df.columns:
    #         vals = sorted(df[c].dropna().unique().tolist())
    #         if len(vals) == 1:
    #             meta_parts.append(f"{c}={vals[0]}")
    #subtitle = " | ".join(meta_parts)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height), sharex=True)

    # Lead time (top)
    for m in methods:
        if(m == "Baseline 1 (SRI EMA)" or m== "Baseline 2 (SRI+SNET EMA)" ):
            continue
        sub = df[df["method"] == m]
        x = sub["dataset_size"].values
        y = sub["avg_lead_time"].values
        y_s = smooth_series(pd.Series(y), window)
        color = cmap[m]

        ax1.scatter(x, y, s=28, alpha=0.45, color=color, edgecolor="none")
        ax1.plot(x, y_s, linewidth=2.5, color=color, label=m)

    style_axes(ax1, "Average lead time (minutes)", label_size, tick_size)

    # FP/day (bottom) — auto symlog if wide dynamic range
    fp_all = df["fp_rate_per_day"].replace([np.inf, -np.inf], np.nan).dropna()
    use_symlog = (fp_all.max() / max(fp_all.min(), 1e-6)) > 30 if len(fp_all) else False

    for m in methods:
        if(m == "Baseline 1 (SRI EMA)" or m== "Baseline 2 (SRI+SNET EMA)" ):
            continue
        sub = df[df["method"] == m]
        x = sub["dataset_size"].values
        y = sub["fp_rate_per_day"].values
        y_s = smooth_series(pd.Series(y), window)
        color = cmap[m]

        ax2.scatter(x, y, s=28, alpha=0.45, color=color, edgecolor="none")
        ax2.plot(x, y_s, linewidth=2.5, color=color, label=m)

    style_axes(ax2, "FP rate per day", label_size, tick_size)
    if use_symlog:
        ax2.set_yscale("symlog", linthresh=0.05)

    # X axis
    ax2.set_xlabel("Dataset size (minutes)", fontsize=label_size)
    ax2.xaxis.set_major_formatter(FuncFormatter(human_int))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, prune=None))
    ax2.tick_params(axis='x', labelsize=tick_size)

    # Legend (single, on top)
    handles, labels = ax1.get_legend_handles_labels()
    ncols = 2
    leg = ax1.legend(
        handles, labels,
        loc="lower right",
        ncol=ncols,
        frameon=False,
        fontsize=legend_size
    )
    if leg and leg.get_title():
        leg.get_title().set_fontsize(legend_size)

    # Titles
    #fig.suptitle("Sweep Results: Smoothed Comparison by Dataset Size", fontsize=label_size + 2, y=0.98)
    #if subtitle:
        #fig.text(0.5, 0.955, subtitle, ha="center", va="center", fontsize=max(legend_size, tick_size), color="#444")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(outdir, "sweep_beauty.png")
    out_svg = os.path.join(outdir, "sweep_beauty.svg")

    plt.savefig(out_png, dpi=dpi)
    plt.savefig(out_svg, dpi=dpi)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sweep_results.csv")
    ap.add_argument("--outdir", default="plots_pretty", help="Output directory for figures")
    ap.add_argument("--window", type=int, default=5, help="Smoothing window (odd, centered rolling median)")
    # in main()
    ap.add_argument("--fig_width", type=float, default=12, help="Figure width in inches")
    ap.add_argument("--fig_height", type=float, default=8, help="Figure height in inches")
    ap.add_argument("--dpi", type=int, default=180, help="Save DPI for PNG/SVG")


    # NEW: sizing controls
    ap.add_argument("--tick_size", type=int, default=10, help="Axis tick value font size")
    ap.add_argument("--legend_size", type=int, default=10, help="Legend label font size")
    ap.add_argument("--label_size", type=int, default=12, help="Axis label font size")

    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    # call to make_plots(...)
    make_plots(
    df=df,
    outdir=args.outdir,
    window=args.window,
    tick_size=args.tick_size,
    legend_size=args.legend_size,
    label_size=args.label_size,
    fig_width=args.fig_width,
    fig_height=args.fig_height,
    dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
