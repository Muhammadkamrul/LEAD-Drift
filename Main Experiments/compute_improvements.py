# FILE: compute_improvements.py
import argparse, os
import pandas as pd
import numpy as np

DEFAULT_BASELINES = [
    "Baseline 1 (SRI EMA)",
    "Baseline 2 (SRI+SNET EMA)",
    "Baseline 3 (Weighted KPIs)",
    "Baseline 4 (Paper's Method)",
]

def safe_pct(numer, denom):
    """Percent = numer/denom*100; returns NaN when denom==0 (avoids inf)."""
    denom = np.where(np.asarray(denom) == 0, np.nan, denom)
    return (np.asarray(numer) / denom) * 100.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to sweep_results.csv")
    ap.add_argument("--outdir", required=True, help="Directory to save outputs")
    ap.add_argument("--lead_method", default="LEAD-Drift (MLP)", help="Name of the LEAD-Drift method in CSV")
    ap.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES, help="Baseline method names to compare")
    ap.add_argument("--min_size", type=float, default=None, help="Minimum dataset_size (inclusive)")
    ap.add_argument("--max_size", type=float, default=None, help="Maximum dataset_size (inclusive)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Ensure types + optional range filtering
    if "dataset_size" not in df.columns or "method" not in df.columns:
        raise ValueError("CSV must contain columns: 'dataset_size', 'method'")
    for col in ["avg_lead_time", "fp_rate_per_day"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: '{col}'")

    df = df.copy()
    df["dataset_size"] = pd.to_numeric(df["dataset_size"])
    if (args.min_size is not None) or (args.max_size is not None):
        m = pd.Series(True, index=df.index)
        if args.min_size is not None: m &= df["dataset_size"] >= args.min_size
        if args.max_size is not None: m &= df["dataset_size"] <= args.max_size
        df = df[m]
        if df.empty:
            raise ValueError("No rows left after dataset_size filtering.")

    # If you swept multiple runs/seeds per size, average within (size, method) first
    grouped = (df.groupby(["dataset_size", "method"], as_index=False)
                 .agg(avg_lead_time=("avg_lead_time","mean"),
                      fp_rate_per_day=("fp_rate_per_day","mean")))

    # Pivot wide for easier arithmetic
    lead_w = grouped.pivot(index="dataset_size", columns="method", values="avg_lead_time")
    fp_w   = grouped.pivot(index="dataset_size", columns="method", values="fp_rate_per_day")

    if args.lead_method not in lead_w.columns:
        raise ValueError(f"Lead method '{args.lead_method}' not found in CSV methods: {list(lead_w.columns)}")

    rows = []
    summary_rows = []

    for base in args.baselines:
        if base not in lead_w.columns:
            print(f"[WARN] Baseline '{base}' not found. Skipping.")
            continue

        # Align on sizes where both methods exist
        common_sizes = lead_w.index.intersection(lead_w[[args.lead_method, base]].dropna().index)
        if len(common_sizes) == 0:
            print(f"[WARN] No overlapping dataset_size rows for baseline '{base}'. Skipping.")
            continue

        # Lead-time improvement (minutes & %): higher is better
        lt_lead = lead_w.loc[common_sizes, args.lead_method]
        lt_base = lead_w.loc[common_sizes, base]
        lt_impr_abs = lt_lead - lt_base
        lt_impr_pct = safe_pct(lt_impr_abs, lt_base)

        # FP/day reduction (absolute & %): lower is better
        fp_lead = fp_w.loc[common_sizes, args.lead_method]
        fp_base = fp_w.loc[common_sizes, base]
        fp_impr_abs = fp_base - fp_lead
        fp_impr_pct = safe_pct(fp_impr_abs, fp_base)

        # Store per-size rows
        tmp = pd.DataFrame({
            "baseline": base,
            "dataset_size": common_sizes,
            "lead_time_improvement_min": lt_impr_abs.values,
            "lead_time_improvement_pct": lt_impr_pct,
            "fp_reduction_per_day": fp_impr_abs.values,
            "fp_reduction_pct": fp_impr_pct,
        })
        rows.append(tmp)

        # Summary (nanmean skips NaNs created when baseline==0)
        summary_rows.append({
            "baseline": base,
            "mean_lead_time_improvement_min": float(np.nanmean(lt_impr_abs)),
            "mean_lead_time_improvement_pct": float(np.nanmean(lt_impr_pct)),
            "mean_fp_reduction_per_day": float(np.nanmean(fp_impr_abs)),
            "mean_fp_reduction_pct": float(np.nanmean(fp_impr_pct)),
            "num_sizes_compared": int(len(common_sizes)),
        })

    if not rows:
        raise SystemExit("No baselines compared. Nothing to write.")

    per_size = pd.concat(rows, ignore_index=True)
    per_size_path = os.path.join(args.outdir, "improvement_by_size.csv")
    per_size.to_csv(per_size_path, index=False)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "improvement_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Pretty print
    pd.set_option("display.precision", 3)
    print("\n=== Per-baseline averages (LEAD-Drift vs baseline) ===")
    print(summary.sort_values("baseline").to_string(index=False))
    print(f"\nWrote:\n  {per_size_path}\n  {summary_path}")

if __name__ == "__main__":
    main()
