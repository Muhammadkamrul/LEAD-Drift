# FILE: sweep_dataset_size.py
import os
import argparse
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt

def run_cmd(cmd, cwd=None):
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_dataset_final", help="Where to write generated CSVs.")
    ap.add_argument("--out_dir", default="out_sweep", help="Top-level output dir for all runs and final artifacts.")
    ap.add_argument("--min_minutes", type=int, default=5000)
    ap.add_argument("--max_minutes", type=int, default=100000)
    ap.add_argument("--step_minutes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    # Compare args (passed through)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ema_window", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=15)
    ap.add_argument("--label_mode", type=str, default="horizon_binary", choices=["horizon_binary", "early_ramp", "original_ramp"])
    ap.add_argument("--horizon_min", type=int, default=120)

    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    all_csvs = []

    run_id_counter = 1
    for minutes in range(args.min_minutes, args.max_minutes + 1, args.step_minutes):
        run_id = f"run_{run_id_counter:03d}_m{minutes}"
        print(f"\n===== {run_id} =====")

        # 1) Generate dataset
        data_csv = os.path.join(args.data_dir, f"collector_health_data_{minutes}.csv")
        gen_cmd = [
            sys.executable, "-m", "generate_final_dataset",
            "--out", data_csv,
            "--minutes", str(minutes),
            "--seed", str(args.seed),
        ]
        run_cmd(gen_cmd)

        # 2) Compare on that dataset
        run_outdir = os.path.join(args.out_dir, f"compare_m{minutes}")
        comp_cmd = [
            sys.executable, "-m", "compare_different_dataset_size",
            "--data", data_csv,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--ema_window", str(args.ema_window),
            "--timesteps", str(args.timesteps),
            "--seed", str(args.seed),
            "--dataset_minutes", str(minutes),
            "--run_id", run_id,
            "--label_mode", args.label_mode,
            "--horizon_min", str(args.horizon_min),
            "--outdir", run_outdir
        ]
        run_cmd(comp_cmd)

        # 3) Collect per-run CSV
        per_method_csv = os.path.join(run_outdir, "metrics_per_method.csv")
        if not os.path.exists(per_method_csv):
            raise FileNotFoundError(f"Expected metrics CSV not found at {per_method_csv}")
        all_csvs.append(per_method_csv)

        run_id_counter += 1

    # 4) Aggregate into one CSV
    frames = [pd.read_csv(p) for p in all_csvs]
    agg = pd.concat(frames, ignore_index=True)
    agg_csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    agg.to_csv(agg_csv_path, index=False)
    print(f"\nSaved aggregated results CSV: {agg_csv_path}")

    # 5) Make a single plot: two y-axes (lead time vs FP/day) by method over dataset size
    # Consistent colors per method for both metrics
    methods = sorted(agg["method"].unique())
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(methods)}

    fig, ax1 = plt.subplots(figsize=(11, 7))
    ax2 = ax1.twinx()

    # Sort x for nice lines
    # We'll plot points connected by lines for each method
    for m in methods:
        dfm = agg[agg["method"] == m].sort_values("dataset_size")
        x = dfm["dataset_size"].values
        y_lead = dfm["avg_lead_time"].values
        y_fp = dfm["fp_rate_per_day"].values

        ax1.plot(x, y_lead, label=f"{m} – lead time", linestyle='-', linewidth=2.0, color=color_map[m])
        ax2.plot(x, y_fp, label=f"{m} – FP/day", linestyle='--', linewidth=2.0, color=color_map[m])

    ax1.set_xlabel("Dataset size (minutes)")
    ax1.set_ylabel("Average lead time (minutes)")
    ax2.set_ylabel("FP rate per day")

    ax1.grid(True, alpha=0.3)

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plt.title("Sweep: Avg Lead Time (solid, left axis) & FP/day (dashed, right axis) vs Dataset Size")
    plot_path = os.path.join(args.out_dir, "sweep_methods.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved combined plot: {plot_path}")

if __name__ == "__main__":
    main()
