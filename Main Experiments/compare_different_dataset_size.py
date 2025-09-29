# FILE: compare_baseline_lstm.py
import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

# Import from your existing project files
from common import read_json
from risk_model import create_risk_model
from plotting import plot_comparison_barchart, plot_full_comparison_timeline
from rigorous_evaluation_paper import calculate_feature_importance

# Set default seed for reproducibility (can be overridden by --seed)
DEFAULT_SEED = 42

def set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def make_forward_looking_dataset(
    df, events, features, is_lstm=False, timesteps=15,
    horizon_min=120,            # how early you want to detect (in minutes)
    label_mode="horizon_binary" # "horizon_binary" | "early_ramp" | "original_ramp"
):
    """
    Create model inputs X and a forward-looking target Y.
    - horizon_binary: y=1 if failure occurs within next `horizon_min` minutes, else 0.
    - early_ramp: ramp from (failure_time - horizon_min) to failure_time.
    - original_ramp: ramp from event['start'] to failure_time.
    """
    df_features = df.copy()
    df_features['cpu_delta'] = df_features['cpu_pct'].diff().fillna(0)
    df_features['sri_delta'] = df_features['sri'].diff().fillna(0)

    y = pd.Series(0.0, index=df.index, dtype='float32')

    for e in events:
        if label_mode == "horizon_binary":
            mask = (df['t'] >= (e['failure_time'] - horizon_min)) & (df['t'] < e['failure_time'])
            y.loc[df.index[mask]] = 1.0
        elif label_mode == "early_ramp":
            mask = (df['t'] >= (e['failure_time'] - horizon_min)) & (df['t'] < e['failure_time'])
            idx = df.index[mask]
            if len(idx) > 0:
                y.loc[idx] = np.linspace(0.2, 1.0, len(idx))
        elif label_mode == "original_ramp":
            mask = (df['t'] >= e['start']) & (df['t'] < e['failure_time'])
            idx = df.index[mask]
            if len(idx) > 0:
                y.loc[idx] = np.linspace(0.0, 1.0, len(idx))

    X, Y, t = [], [], []
    start_index = timesteps if is_lstm else 1

    for i in range(start_index, len(df_features)):
        if is_lstm:
            X.append(df_features[features].iloc[i - timesteps:i].values)
        else:
            X.append(df_features[features].iloc[i - 1].values)
        Y.append(y.iloc[i - 1])
        t.append(df_features['t'].iloc[i - 1])

    return np.array(X, dtype='float32'), np.array(Y, dtype='float32').reshape(-1, 1), np.array(t)

def evaluate_method(alert_times, events, test_duration_days):
    """Generic function to calculate per-event metrics for any method."""
    tp, fn, lead_times = 0, 0, []
    for event in events:
        alarms_in_window = alert_times[(alert_times >= event['start']) & (alert_times <= event['failure_time'])]
        if len(alarms_in_window) > 0:
            tp += 1
            lead_times.append(event['failure_time'] - alarms_in_window[0])
        else:
            fn += 1

    fp = 0
    for time in alert_times:
        if not any(event['start'] <= time <= event['end'] for event in events):
            fp += 1

    return {
        "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "avg_lead_time": float(np.mean(lead_times)) if lead_times else 0.0,
        "fp_rate": fp / test_duration_days if test_duration_days > 0 else 0.0
    }

def tune_baseline_threshold(score_series, events, lower_is_worse=True):
    """Finds the best alert threshold for a baseline method on the training data."""
    best_f1 = -1.0
    best_thresh = float(score_series.iloc[0]) if len(score_series) else 0.0

    y_true = np.zeros(len(score_series), dtype=int)
    for event in events:
        start_idx = np.searchsorted(score_series.index.values, event['start'])
        end_idx = np.searchsorted(score_series.index.values, event['end'])
        y_true[start_idx:end_idx] = 1

    # Guard for flat series
    smin, smax = float(score_series.min()), float(score_series.max())
    if smax == smin:
        return smin

    for thresh in np.linspace(smin, smax, 50):
        y_pred = (score_series < thresh) if lower_is_worse else (score_series > thresh)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh

def dump_metrics(outdir, comparison_results, run_params):
    os.makedirs(outdir, exist_ok=True)
    # JSON (nested) for debugging/inspection
    metrics_json_path = os.path.join(outdir, "metrics.json")
    payload = {
        "run_params": run_params,
        "results": {
            name: {
                "detection_rate": float(data["metrics"]["detection_rate"]),
                "avg_lead_time": float(data["metrics"]["avg_lead_time"]),
                "fp_rate_per_day": float(data["metrics"]["fp_rate"]),
                "threshold": float(data["threshold"]),
                "lower_is_worse": bool(data["lower_is_worse"])
            }
            for name, data in comparison_results.items()
        }
    }
    with open(metrics_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Flat CSV (one row per method)
    rows = []
    for name, data in comparison_results.items():
        m = data["metrics"]
        rows.append({
            "run_id": run_params.get("run_id"),
            "epochs": run_params.get("epochs"),
            "batch": run_params.get("batch"),
            "ema_window": run_params.get("ema_window"),
            "dataset_size": run_params.get("dataset_minutes"),
            "seed": run_params.get("seed"),
            "method": name,
            "detection_rate": float(m["detection_rate"]),
            "avg_lead_time": float(m["avg_lead_time"]),
            "fp_rate_per_day": float(m["fp_rate"]),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "metrics_per_method.csv")
    df.to_csv(csv_path, index=False)
    return metrics_json_path, csv_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--ema_window', type=int, default=15)
    ap.add_argument('--timesteps', type=int, default=15, help="Lookback window for MLP/LSTM feature alignment.")
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED, help="Random seed for numpy/tensorflow.")
    ap.add_argument('--dataset_minutes', type=int, default=None, help="Dataset size in minutes (for logging).")
    ap.add_argument('--run_id', type=str, default=None, help="Run identifier (for logging).")
    # Early-detection target options (you can keep defaults)
    ap.add_argument('--label_mode', type=str, default="horizon_binary", choices=["horizon_binary", "early_ramp", "original_ramp"])
    ap.add_argument('--horizon_min', type=int, default=120)

    args = ap.parse_args()
    set_seeds(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events = read_json(event_path)["events"]

    split_idx = int(0.8 * len(df))
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
    train_events = [e for e in events if e['end'] <= df_train['t'].max()]
    test_events = [e for e in events if e['start'] >= df_test['t'].min()]
    test_duration_days = (df_test['t'].max() - df_test['t'].min()) / 1440.0

    comparison_results = {}
    features = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']

    # --- Evaluate LEAD-Drift (MLP with New Target) ---
    print("Training and evaluating LEAD-Drift (MLP)...")
    X_train_mlp, Y_train_mlp, t_train_mlp = make_forward_looking_dataset(
        df_train, train_events, features,
        is_lstm=False, timesteps=args.timesteps,
        horizon_min=args.horizon_min, label_mode=args.label_mode
    )
    X_test_mlp, _, t_test_mlp = make_forward_looking_dataset(
        df_test, test_events, features,
        is_lstm=False, timesteps=args.timesteps,
        horizon_min=args.horizon_min, label_mode=args.label_mode
    )

    tf.keras.backend.clear_session()
    set_seeds(args.seed)
    mlp_model = create_risk_model(input_dim=len(features))
    mlp_model.fit(X_train_mlp, Y_train_mlp, epochs=args.epochs, batch_size=args.batch, verbose=0)

    raw_scores_mlp = mlp_model.predict(X_test_mlp, verbose=0).flatten()
    ema_scores_mlp = pd.Series(raw_scores_mlp, index=t_test_mlp).ewm(span=args.ema_window, adjust=False).mean()

    train_raw_scores_mlp = mlp_model.predict(X_train_mlp, verbose=0).flatten()
    train_ema_scores_mlp = pd.Series(train_raw_scores_mlp, index=t_train_mlp).ewm(span=args.ema_window, adjust=False).mean()
    best_mlp_thresh = tune_baseline_threshold(train_ema_scores_mlp, train_events, lower_is_worse=False)

    mlp_alert_times = t_test_mlp[ema_scores_mlp.values >= best_mlp_thresh]
    comparison_results['LEAD-Drift (MLP)'] = {
        "metrics": evaluate_method(mlp_alert_times, test_events, test_duration_days),
        "alerts": mlp_alert_times,
        "score_series": ema_scores_mlp,
        "threshold": best_mlp_thresh,
        "lower_is_worse": False
    }

    # --- Baselines ---
    feature_importances = calculate_feature_importance(mlp_model, X_train_mlp)

    print("Evaluating Baseline 1: EMA on SRI...")
    sri_ema_train = df_train['sri'].ewm(span=args.ema_window, adjust=False).mean()
    sri_ema_test = df_test['sri'].ewm(span=args.ema_window, adjust=False).mean()
    b1_thresh = tune_baseline_threshold(sri_ema_train, train_events, lower_is_worse=True)
    b1_alert_times = df_test['t'][sri_ema_test < b1_thresh].values
    comparison_results['Baseline 1 (SRI EMA)'] = {
        "metrics": evaluate_method(b1_alert_times, test_events, test_duration_days),
        "alerts": b1_alert_times,
        "score_series": sri_ema_test,
        "threshold": b1_thresh,
        "lower_is_worse": True
    }

    print("Evaluating Baseline 2: EMA on SRI+SNET...")
    snet_ema_train = df_train['snet'].ewm(span=args.ema_window, adjust=False).mean()
    snet_ema_test = df_test['snet'].ewm(span=args.ema_window, adjust=False).mean()
    health_score_train = 0.5 * sri_ema_train + 0.5 * snet_ema_train
    health_score_test = 0.5 * sri_ema_test + 0.5 * snet_ema_test
    b2_thresh = tune_baseline_threshold(health_score_train, train_events, lower_is_worse=True)
    b2_alert_times = df_test['t'][health_score_test < b2_thresh].values
    comparison_results['Baseline 2 (SRI+SNET EMA)'] = {
        "metrics": evaluate_method(b2_alert_times, test_events, test_duration_days),
        "alerts": b2_alert_times,
        "score_series": health_score_test,
        "threshold": b2_thresh,
        "lower_is_worse": True
    }

    print("Evaluating Baseline 3: Weighted KPI EMA...")
    kpis = ['snet', 'sri', 'ram_pct', 'storage_pct', 'cpu_pct']
    weights = {k: float(feature_importances.get(k, 0)) for k in kpis}
    total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    health_score_train_b3 = pd.Series(0.0, index=df_train.index)
    health_score_test_b3 = pd.Series(0.0, index=df_test.index)
    for kpi in kpis:
        ema_train = df_train[kpi].ewm(span=args.ema_window, adjust=False).mean()
        ema_test = df_test[kpi].ewm(span=args.ema_window, adjust=False).mean()
        term_train, term_test = ((100 - ema_train), (100 - ema_test)) if kpi == 'cpu_pct' else (ema_train, ema_test)
        health_score_train_b3 += weights[kpi] * term_train
        health_score_test_b3 += weights[kpi] * term_test
    health_score_train_b3 /= total_weight
    health_score_test_b3 /= total_weight
    b3_thresh = tune_baseline_threshold(health_score_train_b3, train_events, lower_is_worse=True)
    b3_alert_times = df_test['t'][health_score_test_b3 < b3_thresh].values
    comparison_results['Baseline 3 (Weighted KPIs)'] = {
        "metrics": evaluate_method(b3_alert_times, test_events, test_duration_days),
        "alerts": b3_alert_times,
        "score_series": health_score_test_b3,
        "threshold": b3_thresh,
        "lower_is_worse": True
    }

    print("Evaluating Baseline 4 (Paper's Method)...")
    kpis_to_check = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri']
    target_vector = pd.Series({'cpu_pct': 55, 'ram_pct': 62, 'storage_pct': 56, 'snet': 100, 'sri': 100})

    def calculate_distance(df_slice):
        diff = df_slice[kpis_to_check] - target_vector
        return np.sqrt(np.sum(diff**2, axis=1))

    drift_score_train = calculate_distance(df_train)
    drift_score_test = calculate_distance(df_test)
    drift_score_train.index = df_train['t']
    drift_score_test.index = df_test['t']
    b4_thresh = tune_baseline_threshold(drift_score_train, train_events, lower_is_worse=False)
    b4_alert_times = df_test['t'][drift_score_test > b4_thresh].values
    comparison_results["Baseline 4 (Paper's Method)"] = {
        "metrics": evaluate_method(b4_alert_times, test_events, test_duration_days),
        "alerts": b4_alert_times,
        "score_series": drift_score_test,
        "threshold": b4_thresh,
        "lower_is_worse": False
    }

    # --- Report to console ---
    print("\n--- Method Comparison Results ---")
    print(f"{'Method':<28} | {'Detection Rate':<15} | {'Avg Lead Time':<15} | {'FP Rate (per day)':<20}")
    print("-" * 85)
    for name, data in comparison_results.items():
        m = data["metrics"]
        dr = f"{m['detection_rate']:.2%}"
        lt = f"{m['avg_lead_time']:.2f}"
        fp = f"{m['fp_rate']:.2f}"
        print(f"{name:<28} | {dr:<15} | {lt:<15} | {fp:<20}")

    # --- Plots (per single run) ---
    results_for_barchart = {name: data['metrics'] for name, data in comparison_results.items()}
    plot_comparison_barchart(results_for_barchart, os.path.join(args.outdir, "comparison_barchart.png"))

    timeline_data = {name: (data['score_series'].iloc[:-1], data['threshold'], data['lower_is_worse']) for name, data in comparison_results.items()}
    all_alerts = {name: data['alerts'] for name, data in comparison_results.items()}
    plot_full_comparison_timeline(
        timeline_data,
        all_alerts,
        test_events,
        os.path.join(args.outdir, "full_comparison_timeline.png")
    )

    # --- Persist metrics for sweeping ---
    run_params = {
        "run_id": args.run_id,
        "epochs": args.epochs,
        "batch": args.batch,
        "ema_window": args.ema_window,
        "dataset_minutes": args.dataset_minutes,
        "seed": args.seed,
        "label_mode": args.label_mode,
        "horizon_min": args.horizon_min,
    }
    metrics_json_path, per_method_csv_path = dump_metrics(args.outdir, comparison_results, run_params)

    print(f"\nSaved metrics to:\n  {metrics_json_path}\n  {per_method_csv_path}")
    print("Generated comparison plots in the output directory.")

if __name__ == '__main__':
    main()
