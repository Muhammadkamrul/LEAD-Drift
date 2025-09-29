# FILE: plot_tradeoff.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

custom_labels = {
    "LEAD-Drift (MLP)": "LEAD-Drift",
    "Baseline 3 (Weighted KPIs)": "Weighted KPIs",
    "Baseline 4 (Paper's Method)": "Target Distance"
}


def plot_performance_tradeoff_with_legend(
    df,
    out_path,
    methods_to_plot=None,
    title_size=18,
    label_size=14,
    tick_size=12,
    legend_size=12,
    # NEW: ideal region controls
    ideal_fp=None,          # if None, use median fp_rate_per_day
    ideal_lead=None,        # if None, use median avg_lead_time
    marker_size=400,        # big points on plot
    legend_markersize=9,    # small markers in legend (no overlap)
):
    """
    Generates a 2D performance trade-off plot (Lead Time vs. FP Rate) with
    a clean legend and a correctly shaded ideal quadrant (top-left).
    X-axis: FP/day (lower is better). Y-axis: Lead time (higher is better).
    """

    if df.empty:
        print("Input DataFrame is empty. Cannot generate plot.")
        return

    if methods_to_plot:
        df = df[df['method'].isin(methods_to_plot)]
        if df.empty:
            print("Warning: None of the specified methods found. Plot will be empty.")
            return

    # Aggregate per method
    df_agg = df.groupby('method', as_index=False).agg(
        avg_lead_time=('avg_lead_time', 'mean'),
        fp_rate_per_day=('fp_rate_per_day', 'mean')
    )
    if df_agg.empty:
        print("No aggregated rows to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors and markers
    methods = df_agg['method'].tolist()  # original names from CSV
    display_labels = [custom_labels.get(m, m) for m in methods]  # <-- mapped names
    cmap = plt.get_cmap('tab10', len(methods))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'h', '<', '>']

    # Scatter points (legend ignores these labels because we use proxies)
    for i, row in df_agg.iterrows():
        ax.scatter(
            row['fp_rate_per_day'],
            row['avg_lead_time'],
            s=marker_size,
            color=cmap(i),
            marker=markers[i % len(markers)],
            edgecolor='black',
            linewidth=0.7,
            zorder=4
        )

    # Titles and labels
    #ax.set_title('Performance Trade-off: Proactiveness vs. Efficiency', fontsize=title_size, pad=14)
    ax.set_xlabel('False Positive Rate (per day)  (lower is better)', fontsize=label_size)
    ax.set_ylabel('Average Lead Time (minutes)  (higher is better)', fontsize=label_size)

    # Improve aesthetics
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4, zorder=0)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.25, zorder=0)
    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=tick_size)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Small padding on axes limits for breathing room
    x_vals = df_agg['fp_rate_per_day'].values
    y_vals = df_agg['avg_lead_time'].values
    xpad = 0.08 * (np.nanmax(x_vals) - np.nanmin(x_vals) + 1e-9)
    ypad = 0.08 * (np.nanmax(y_vals) - np.nanmin(y_vals) + 1e-9)
    ax.set_xlim(np.nanmin(x_vals) - xpad, np.nanmax(x_vals) + xpad)
    ax.set_ylim(np.nanmin(y_vals) - ypad, np.nanmax(y_vals) + ypad)

    # Decide ideal thresholds (top-left quadrant)
    fp_thr = float(ideal_fp) if ideal_fp is not None else float(df_agg['fp_rate_per_day'].median())
    lt_thr = float(ideal_lead) if ideal_lead is not None else float(df_agg['avg_lead_time'].median())

    # After limits are set, compute rectangle for top-left (x <= fp_thr, y >= lt_thr)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Clip thresholds to current limits
    fp_thr_clamped = max(xmin, min(fp_thr, xmax))
    lt_thr_clamped = max(ymin, min(lt_thr, ymax))

    # Draw dashed reference lines
    ax.axvline(fp_thr_clamped, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)
    ax.axhline(lt_thr_clamped, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)

    # Shade the top-left quadrant as the ideal region
    ideal_rect = Rectangle(
        (xmin, lt_thr_clamped),
        width=(fp_thr_clamped - xmin),
        height=(ymax - lt_thr_clamped),
        facecolor='tab:green',
        alpha=0.10,
        edgecolor=None,
        zorder=0,
        label='_nolegend_'  # do not show in legend
    )
    ax.add_patch(ideal_rect)

    # Label the shaded region once
    ax.text(
        xmin + 0.02 * (xmax - xmin),
        lt_thr_clamped + 0.92 * (ymax - lt_thr_clamped),
        'Ideal\nregion',
        color='tab:green',
        fontsize=label_size,
        va='top',
        ha='left'
    )

    # Build a clean legend with proxy markers (small, uniform size)
    proxy_handles = []
    for i, disp in enumerate(display_labels):
        proxy_handles.append(
            Line2D([0], [0],
                marker=markers[i % len(markers)],
                color='none',
                markerfacecolor=cmap(i),
                markeredgecolor='black',
                markersize=legend_markersize,
                linewidth=0,
                label=disp)  # <-- use mapped name here
        )

    leg = ax.legend(
        handles=proxy_handles,
        loc='lower right',
        frameon=False,
        fontsize=legend_size,
        title_fontsize=legend_size + 1,
        ncol=1,
        handletextpad=0.6,
        columnspacing=0.8
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave room for legend on the right
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a 2D performance trade-off plot from sweep results.")
    parser.add_argument('--csv', required=True, help="Path to the sweep_results.csv file.")
    parser.add_argument('--outdir', required=True, help="Directory to save the output plot.")
    parser.add_argument('--methods', nargs='*', default=None, help="Optional: method names to include.")
    parser.add_argument('--title_size', type=int, default=18)
    parser.add_argument('--label_size', type=int, default=14)
    parser.add_argument('--tick_size', type=int, default=12)
    parser.add_argument('--legend_size', type=int, default=12)
    parser.add_argument('--marker_size', type=float, default=400)
    parser.add_argument('--legend_markersize', type=float, default=9)
    # Optional manual thresholds for the ideal quadrant
    parser.add_argument('--ideal_fp', type=float, default=None, help="Manual FP/day threshold for ideal region.")
    parser.add_argument('--ideal_lead', type=float, default=None, help="Manual lead-time threshold for ideal region.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, 'performance_tradeoff_plot.png')

    try:
        df_results = pd.read_csv(args.csv)
        plot_performance_tradeoff_with_legend(
            df=df_results,
            out_path=out_path,
            methods_to_plot=args.methods,
            title_size=args.title_size,
            label_size=args.label_size,
            tick_size=args.tick_size,
            legend_size=args.legend_size,
            marker_size=args.marker_size,
            legend_markersize=args.legend_markersize,
            ideal_fp=args.ideal_fp,
            ideal_lead=args.ideal_lead
        )
    except FileNotFoundError:
        print(f"Error: The file '{args.csv}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
