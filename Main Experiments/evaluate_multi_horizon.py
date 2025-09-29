# FILE: evaluate_multi_horizon.py
import argparse
import os
import re
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import from your existing project files
from common import read_json
from compare_different_dataset_size import make_forward_looking_dataset

def estimate_time_to_failure(scores, horizons, thresholds):
    """
    Infers a time-to-failure range based on which models have crossed 
    their individually tuned thresholds.
    """
    active_horizons = [h for h, s in zip(horizons, scores) if s >= thresholds[h]]
    
    if not active_horizons:
        return f">{max(horizons)} min"
    
    min_active_h = min(active_horizons)
    larger_horizons = sorted([h for h in horizons if h > min_active_h])
    upper_bound_str = str(larger_horizons[0]) if larger_horizons else str(max(horizons))

    return f"{min_active_h}-{upper_bound_str} min"

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-horizon models using their tuned thresholds.")
    parser.add_argument('--data', required=True, help="Path to the dataset CSV.")
    parser.add_argument('--models_dir', required=True, help="Directory containing the trained models and _meta.json files.")
    parser.add_argument('--event_index', type=int, default=0, help="The index of the drift event to analyze.")
    parser.add_argument('--outdir', required=True, help="Directory to save the evaluation plot.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. Load Data, Models, and Tuned Thresholds ---
    print("Loading data, models, and tuned thresholds...")
    df = pd.read_csv(args.data)
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events = read_json(event_path)["events"]
    features = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']
    
    models = {}
    thresholds = {}
    horizons = []
    ema_window = 5 # Default, will be read from first meta file

    for f in sorted(os.listdir(args.models_dir)):
        if f.endswith('.h5') and f.startswith('risk_model_h'):
            match = re.search(r'h(\d+)', f)
            if match:
                h = int(match.group(1))
                model_path = os.path.join(args.models_dir, f)
                meta_path = os.path.join(args.models_dir, f.replace('.h5', '_meta.json'))
                
                if not os.path.exists(meta_path):
                    print(f"Warning: Metadata file not found for model {f}. Skipping.")
                    continue
                
                horizons.append(h)
                models[h] = tf.keras.models.load_model(model_path)
                with open(meta_path, 'r') as meta_f:
                    meta = json.load(meta_f)
                    thresholds[h] = meta['optimal_threshold']
                    ema_window = meta.get('ema_window', ema_window)

    if not models:
        raise FileNotFoundError(f"No valid model/metadata pairs found in {args.models_dir}")
    print(f"Loaded models for horizons: {horizons}")
    print(f"Using tuned thresholds: {thresholds}")

    # --- 2. Select Event and Prepare Data ---
    event = events[args.event_index]
    print(f"\nAnalyzing event starting at t={event['start']} with failure at t={event['failure_time']}")
    df_slice = df[(df['t'] >= event['start'] - max(horizons) * 1.5) & (df['t'] <= event['failure_time'] + 10)].copy()
    X_eval, _, t_eval = make_forward_looking_dataset(df_slice, [], features)

    # --- 3. Make Predictions and Display Results ---
    results = {'Time': t_eval}
    for h in horizons:
        raw_scores = models[h].predict(X_eval).flatten()
        ema_scores = pd.Series(raw_scores, index=t_eval).ewm(span=ema_window, adjust=False).mean()
        results[f'Risk (H={h})'] = ema_scores
        results[f'Thresh (H={h})'] = thresholds[h]
    
    results_df = pd.DataFrame(results)
    
    score_cols = [f'Risk (H={h})' for h in horizons]
    results_df['Est. Time to Failure'] = results_df[score_cols].apply(
        lambda row: estimate_time_to_failure(row.values, horizons, thresholds), axis=1
    )
    
    print("\n--- Numerical Results: Smoothed Risk Scores Over Time ---")
    display_df = results_df[results_df['Time'] >= event['start'] - 5]
    print(display_df.to_string(float_format="%.4f"))

    # --- 4. Plot the Results ---
    fig, ax = plt.subplots(figsize=(15, 6))
    for h in horizons:
        ax.plot(results_df['Time'], results_df[f'Risk (H={h})'], label=f'Risk (Horizon {h} min)', lw=2)
        ax.axhline(y=thresholds[h], color=ax.get_lines()[-1].get_color(), linestyle=':', label=f'Threshold (H={h})')
    
    ax.axvline(x=event['start'], color='grey', linestyle='--', label='Drift Start')
    ax.axvline(x=event['failure_time'], color='red', linestyle='-', label='Failure Time')
    
    #ax.set_title(f'Multi-Horizon Risk Scores for Event at t={event["start"]}', fontsize=16)
    ax.set_xlabel('Time (minutes)', fontsize=18)
    ax.set_ylabel('Predicted Risk Score (Smoothed)', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.outdir, f'multi_horizon_plot_event_{args.event_index}.png')
    plt.savefig(plot_path, dpi=120)
    print(f"\nSUCCESS: Saved multi-horizon evaluation plot to:\n{plot_path}")

if __name__ == '__main__':
    main()