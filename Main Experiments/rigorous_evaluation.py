import os, argparse, json
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from collections import defaultdict
from common import read_json
from risk_model import create_risk_model
from train_risk_model import make_dataset
# MODIFIED: Import the bar_attribution plotting function
from plotting import plot_lead_time_distribution, plot_performance_tradeoff, plot_example_detection_timeline, bar_attribution
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- Place the new function here ---
def calculate_feature_importance(model, X_sample):
    """
    Calculates the feature importance by measuring the gradient of the
    output with respect to the input features.
    """
    X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        y_pred = model(X_tensor, training=False)
    grads = tape.gradient(y_pred, X_tensor)
    abs_grads = tf.math.abs(grads)
    mean_grads = tf.reduce_mean(abs_grads, axis=0).numpy()
    names = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']
    raw_importances = {name: float(grad) for name, grad in zip(names, mean_grads)}
    total_importance = sum(raw_importances.values())
    if total_importance == 0: return raw_importances
    normalized_importances = {k: v / total_importance for k, v in raw_importances.items()}
    return normalized_importances

def run_evaluation_on_fold(df_train, df_test, events, args, threshold):
    X_train, Y_train, _ = make_dataset(df_train)
    X_test, _, t_test = make_dataset(df_test)
    tf.keras.backend.clear_session()
    model = create_risk_model(input_dim=X_train.shape[1])
    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch, verbose=0)
    
    # MODIFIED: Calculate feature importance for the model in this fold
    importances = calculate_feature_importance(model, X_train)

    raw_risk_scores = model.predict(X_test, verbose=0).flatten()
    # ... (rest of the function is the same) ...
    risk_series = pd.Series(raw_risk_scores, index=t_test)
    ema_risk_scores = risk_series.ewm(span=args.ema_window, adjust=False).mean()
    alert_times = t_test[ema_risk_scores >= threshold]
    test_events = [e for e in events if df_test['t'].min() <= e['start'] <= df_test['t'].max()]
    if not test_events: return None
    tp, fn, lead_times = 0, 0, []
    for event in test_events:
        alarms_in_window = alert_times[(alert_times >= event['start']) & (alert_times <= event['failure_time'])]
        if len(alarms_in_window) > 0:
            tp += 1
            lead_times.append(event['failure_time'] - alarms_in_window[0])
        else: fn += 1
    fp = 0
    for time in alert_times:
        if not any(event['start'] <= time <= event['end'] for event in test_events): fp += 1
    
    result = {
        "tp": tp, "fn": fn, "fp": fp, "lead_times": lead_times,
        "test_duration_days": (t_test[-1] - t_test[0]) / 1440.0,
        "plot_data": {
            "t": t_test, "raw_scores": raw_risk_scores, "ema_scores": ema_risk_scores.to_numpy(),
            "alert_times": alert_times, "events": test_events
        },
        "importances": importances # MODIFIED: Return the importances
    }
    return result

def main():
    # ... (Argument parsing and initial setup is unchanged) ...
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ema_window', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events = read_json(event_path)["events"]

    # ... (Trade-off curve generation is unchanged) ...
    print("--- Generating Performance Trade-off Curve Data ---")
    thresholds_to_test = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    tradeoff_results_by_thresh = {}
    for threshold in thresholds_to_test:
        print(f"Testing with threshold τ = {threshold:.2f}...")
        kf = KFold(n_splits=args.folds, shuffle=False)
        fold_results = []
        for train_index, test_index in kf.split(df):
            result = run_evaluation_on_fold(df.iloc[train_index], df.iloc[test_index], events, args, threshold)
            if result: fold_results.append(result)
        tradeoff_results_by_thresh[threshold] = fold_results

    # ... (Final metrics generation is unchanged) ...
    best_threshold = 0.10
    final_fold_results = tradeoff_results_by_thresh[best_threshold]
    
    # ... (Printing final metrics is unchanged) ...
    all_lead_times = [lt for r in final_fold_results for lt in r['lead_times']]
    detection_rates = [(r['tp'] / (r['tp'] + r['fn'])) for r in final_fold_results if (r['tp'] + r['fn']) > 0]
    total_fp = sum(r['fp'] for r in final_fold_results)
    total_days = sum(r['test_duration_days'] for r in final_fold_results)
    fp_rate_per_day = total_fp / total_days
    print(f"\n--- Rigorous Evaluation Final Results (at τ={best_threshold}) ---")
    print(f"Detection Rate:          {np.mean(detection_rates):.2%} ± {np.std(detection_rates):.2%}")
    print(f"Average Lead Time (mins): {np.mean(all_lead_times):.2f} ± {np.std(all_lead_times):.2f}")
    print(f"False Positive Rate (per day): {fp_rate_per_day:.2f}")

    # --- MODIFIED: Aggregate, display, and plot feature importances ---
    print("\n--- Average Feature Importance (across all folds) ---")
    summed_importances = defaultdict(float)
    for r in final_fold_results:
        for feature, value in r['importances'].items():
            summed_importances[feature] += value
    
    avg_importances = {f: v / len(final_fold_results) for f, v in summed_importances.items()}
    # Sort for display
    sorted_importances = sorted(avg_importances.items(), key=lambda item: item[1], reverse=True)
    
    for feature, value in sorted_importances:
        print(f"{feature:<15} {value:.3f}")

    bar_attribution(
        vals=[avg_importances[k] for k in ['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta', 'sri_delta']],
        labels=['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta', 'sri_delta'],
        path=os.path.join(args.outdir, 'average_feature_importance.png'),
        title='Average Feature Importance Across Folds'
    )

    # --- Plot generation (unchanged) ---
    # ...
    print("\nGenerated 4 plots in the output directory.")

if __name__ == '__main__':
    main()