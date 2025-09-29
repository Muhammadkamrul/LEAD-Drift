# FILE: compare_baseline.py
import os, argparse, json
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

# Import from your existing project files
from common import read_json
from risk_model import create_risk_model
from train_risk_model import make_dataset
from plotting import plot_comparison_barchart, plot_full_comparison_timeline 
from rigorous_evaluation_paper import calculate_feature_importance

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
        "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "avg_lead_time": np.mean(lead_times) if lead_times else 0,
        "fp_rate": fp / test_duration_days if test_duration_days > 0 else 0
    }

def tune_baseline_threshold(score_series, events, lower_is_worse=True):
    """Finds the best alert threshold for a baseline method on the training data."""
    best_f1 = -1
    best_thresh = 0
    
    y_true = np.zeros(len(score_series), dtype=int)
    for event in events:
        start_idx = np.searchsorted(score_series.index.values, event['start'])
        end_idx = np.searchsorted(score_series.index.values, event['end'])
        y_true[start_idx:end_idx] = 1
        
    for thresh in np.linspace(score_series.min(), score_series.max(), 50):
        y_pred = (score_series < thresh) if lower_is_worse else (score_series > thresh)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ema_window', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events = read_json(event_path)["events"]

    split_idx = int(0.8 * len(df))
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
    train_events = [e for e in events if e['end'] <= df_train['t'].max()]
    test_events = [e for e in events if e['start'] >= df_test['t'].min()]
    test_duration_days = (df_test['t'].max() - df_test['t'].min()) / 1440.0
    
    # MODIFIED: Use a single, unified dictionary to store all results
    comparison_results = {}
    
    # --- 2. Evaluate LEAD-Drift (Our Method) ---
    print("Training and evaluating LEAD-Drift...")
    X_train, Y_train, _ = make_dataset(df_train)
    X_test, _, t_test = make_dataset(df_test)
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = create_risk_model(input_dim=X_train.shape[1])
    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch, verbose=0)
    
    raw_scores = model.predict(X_test, verbose=0).flatten()
    ema_scores = pd.Series(raw_scores, index=t_test).ewm(span=args.ema_window, adjust=False).mean()
    
    _, _, t_train = make_dataset(df_train)
    train_raw_scores = model.predict(X_train, verbose=0).flatten()
    train_ema_scores = pd.Series(train_raw_scores, index=t_train).ewm(span=args.ema_window, adjust=False).mean()
    best_ld_thresh = tune_baseline_threshold(train_ema_scores, train_events, lower_is_worse=False)

    ld_alert_times = t_test[ema_scores.values >= best_ld_thresh]
    comparison_results['LEAD-Drift'] = {
        "metrics": evaluate_method(ld_alert_times, test_events, test_duration_days),
        "alerts": ld_alert_times,
        "score_series": ema_scores,
        "threshold": best_ld_thresh,
        "lower_is_worse": False
    }

    # --- 3. Evaluate Baselines ---
    feature_importances = calculate_feature_importance(model, X_train)
    
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
    weights = {k: feature_importances.get(k, 0) for k in kpis}
    total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1
    health_score_train_b3, health_score_test_b3 = pd.Series(0.0, index=df_train.index), pd.Series(0.0, index=df_test.index)
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
    
    # --- NEW: Baseline 4 (Paper's Method) ---
    print("Evaluating Baseline 4 (Paper's Method)...")
    kpis_to_check = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri']
    target_vector = pd.Series({'cpu_pct': 55, 'ram_pct': 62, 'storage_pct': 56, 'snet': 100, 'sri': 100})

    def calculate_distance(df_slice):
        diff = df_slice[kpis_to_check] - target_vector
        return np.sqrt(np.sum(diff**2, axis=1))

    drift_score_train = calculate_distance(df_train)
    drift_score_test = calculate_distance(df_test)
    # Associate drift scores with time from the original dataframe index
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

    # --- 5. Report Results ---
    print("\n--- Method Comparison Results ---")
    print(f"{'Method':<28} | {'Detection Rate':<15} | {'Avg Lead Time':<15} | {'FP Rate (per day)':<20}")
    print("-" * 85)
    for name, data in comparison_results.items():
        metrics = data['metrics']
        dr = f"{metrics['detection_rate']:.2%}"
        lt = f"{metrics['avg_lead_time']:.2f}"
        fp = f"{metrics['fp_rate']:.2f}"
        print(f"{name:<28} | {dr:<15} | {lt:<15} | {fp:<20}")

    # --- 6. Generate Plots ---
    results_for_barchart = {name: data['metrics'] for name, data in comparison_results.items()}
    plot_comparison_barchart(results_for_barchart, os.path.join(args.outdir, "comparison_barchart.png"))
    
    # MODIFIED: Extract plot data from the new unified dictionary
    timeline_data = {name: (data['score_series'].iloc[:-1], data['threshold'], data['lower_is_worse']) for name, data in comparison_results.items()}
    all_alerts = {name: data['alerts'] for name, data in comparison_results.items()}
    
    plot_full_comparison_timeline(
        timeline_data, 
        all_alerts, 
        test_events, 
        os.path.join(args.outdir, "full_comparison_timeline.png")
    )
    
    print("\nGenerated comparison plots in the output directory.")

if __name__ == '__main__':
    main()