
import os, argparse
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from collections import defaultdict

# Import functions from your existing project files
from common import read_json
from risk_model import create_risk_model
#from train_risk_model import make_forward_looking_dataset  # forward-looking labels
from plotting import (
    plot_lead_time_distribution,
    plot_performance_tradeoff,
    plot_example_detection_timeline,
    bar_attribution,
)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_forward_looking_dataset(
    df, events, features, is_lstm=False, timesteps=15,
    horizon_min=120,            # how early you want to detect (in minutes)
    label_mode="horizon_binary" # "horizon_binary" | "early_ramp" | "original_ramp"
):
    """
    Create model inputs X and a forward-looking target Y.
    - horizon_binary: y=1 if failure occurs within next `horizon_min` minutes, else 0.
    - early_ramp: ramp from (failure_time - horizon_min) to failure_time.
    - original_ramp: your current ramp from event['start'] to failure_time.
    """
    df_features = df.copy()
    df_features['cpu_delta'] = df_features['cpu_pct'].diff().fillna(0)
    df_features['sri_delta'] = df_features['sri'].diff().fillna(0)

    y = pd.Series(0.0, index=df.index, dtype='float32')

    for e in events:
        if label_mode == "horizon_binary":
            # Positive if we're within H minutes BEFORE failure
            mask = (df['t'] >= (e['failure_time'] - horizon_min)) & (df['t'] < e['failure_time'])
            y.loc[df.index[mask]] = 1.0

        elif label_mode == "early_ramp":
            # Ramp up earlier than before: start H minutes before failure
            mask = (df['t'] >= (e['failure_time'] - horizon_min)) & (df['t'] < e['failure_time'])
            idx = df.index[mask]
            if len(idx) > 0:
                y.loc[idx] = np.linspace(0.2, 1.0, len(idx))  # front-loaded ramp

        elif label_mode == "original_ramp":
            mask = (df['t'] >= e['start']) & (df['t'] < e['failure_time'])
            idx = df.index[mask]
            if len(idx) > 0:
                y.loc[idx] = np.linspace(0.0, 1.0, len(idx))

    # Build X, Y, t (keep your existing alignment: label at end-of-window)
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

# ---------------------------------------------------------------------
# Threshold tuning (your function)
# ---------------------------------------------------------------------
def tune_threshold(score_series, events, lower_is_worse=False):
    """Finds the best alert threshold for a baseline method on the training data."""
    best_f1 = -1
    best_thresh = 0

    # Guard: empty series
    if len(score_series) == 0:
        return 0.0

    y_true = np.zeros(len(score_series), dtype=int)
    for event in events:
        start_idx = np.searchsorted(score_series.index.values, event['start'])
        end_idx = np.searchsorted(score_series.index.values, event['end'])
        y_true[start_idx:end_idx] = 1

    smin, smax = float(score_series.min()), float(score_series.max())
    # Guard: flat series -> any threshold is equivalent
    if smax == smin:
        return smin

    for thresh in np.linspace(smin, smax, 50):
        y_pred = (score_series < thresh) if lower_is_worse else (score_series > thresh)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh

# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------
def calculate_feature_importance(model, X_sample):
    """Calculates feature importance using gradient sensitivity."""
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
    if total_importance == 0:
        return raw_importances
    return {k: v / total_importance for k, v in raw_importances.items()}

def _overlapping_events(events, t_min, t_max):
    """Return events that overlap [t_min, t_max] using [start, failure_time] window."""
    overlapped = []
    for e in events:
        s = e.get('start')
        f = e.get('failure_time', e.get('end', s))
        if (s <= t_max) and (f >= t_min):  # intervals intersect
            overlapped.append(e)
    return overlapped

# ---------------------------------------------------------------------
# One fold: train, tune threshold on train EMA, evaluate on test
# ---------------------------------------------------------------------
def run_evaluation_on_fold(df_train, df_test, events, args):
    """Trains and evaluates a model for a single fold using forward-looking labels."""
    # Select events overlapping each split
    train_events = _overlapping_events(events, df_train['t'].min(), df_train['t'].max())
    test_events  = _overlapping_events(events, df_test['t'].min(),  df_test['t'].max())

    features = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']

    # Build datasets with forward-looking labels (train + test)
    X_train, Y_train, t_train = make_forward_looking_dataset(
        df=df_train,
        events=train_events,
        features=features,
        is_lstm=False,
        timesteps=args.timesteps,
        horizon_min=args.horizon_min,
        label_mode=args.label_mode,
    )
    X_test, _, t_test = make_forward_looking_dataset(
        df=df_test,
        events=test_events,
        features=features,
        is_lstm=False,
        timesteps=args.timesteps,
        horizon_min=args.horizon_min,
        label_mode=args.label_mode,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        return None  # nothing to train/evaluate on this split

    # Fresh model per fold
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = create_risk_model(input_dim=X_train.shape[1])
    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch, verbose=0)

    importances = calculate_feature_importance(model, X_train)

    # -------- Tune threshold on TRAIN EMA --------
    raw_scores_train = model.predict(X_train, verbose=0).flatten()
    ema_train = pd.Series(raw_scores_train, index=t_train).ewm(span=args.ema_window, adjust=False).mean()
    tuned_threshold = tune_threshold(ema_train, train_events, lower_is_worse=False)

    # -------- Apply tuned threshold on TEST EMA --------
    raw_scores_test = model.predict(X_test, verbose=0).flatten()
    risk_series_test = pd.Series(raw_scores_test, index=t_test)
    ema_test = risk_series_test.ewm(span=args.ema_window, adjust=False).mean()
    alert_times = t_test[ema_test >= tuned_threshold]

    # Evaluate on test events (lead time wrt failure_time)
    if not test_events:
        return None

    tp, fn, lead_times = 0, 0, []
    for event in test_events:
        alarms_in_window = alert_times[(alert_times >= event['start']) & (alert_times <= event['failure_time'])]
        if len(alarms_in_window) > 0:
            tp += 1
            lead_times.append(event['failure_time'] - alarms_in_window[0])
        else:
            fn += 1

    fp = 0
    for t in alert_times:
        if not any(ev['start'] <= t <= ev['end'] for ev in test_events):
            fp += 1

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "lead_times": lead_times,
        "test_duration_days": (t_test[-1] - t_test[0]) / 1440.0 if len(t_test) > 1 else 0.0,
        "plot_data": {
            "t": t_test,
            "raw_scores": raw_scores_test,
            "ema_scores": ema_test.to_numpy(),
            "alert_times": alert_times,
            "events": test_events,
            "tuned_threshold": tuned_threshold,
        },
        "tuned_threshold": tuned_threshold,
        "importances": importances,
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help="Path to the dataset CSV.")
    ap.add_argument('--outdir', required=True, help="Directory to save results and plots.")
    ap.add_argument('--folds', type=int, default=5, help="Number of folds for cross-validation.")
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ema_window', type=int, default=10)
    # Forward-looking label config
    ap.add_argument('--timesteps', type=int, default=15, help="Lookback window for feature alignment.")
    ap.add_argument('--label_mode', type=str, default="horizon_binary",
                    choices=["horizon_binary", "early_ramp", "original_ramp"])
    ap.add_argument('--horizon_min', type=int, default=120,
                    help="Minutes prior to failure considered 'positive' for labels.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events = read_json(event_path)["events"]

    # --- Run K-Fold Cross-Validation with tuned thresholds ---
    print(f"--- Starting {args.folds}-Fold Cross-Validation (threshold tuned per fold) ---")
    kf = KFold(n_splits=args.folds, shuffle=False)
    final_fold_results = []

    for i, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"Running Fold {i+1}/{args.folds}...")
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        result = run_evaluation_on_fold(df_train, df_test, events, args)
        if result:
            final_fold_results.append(result)
            print(f"  Tuned threshold (fold {i+1}): {result['tuned_threshold']:.6f}")
            # Save an example plot from the first fold
            if i == 0:
                pdict = result['plot_data']
                plot_example_detection_timeline(
                    pdict['t'],
                    pdict['raw_scores'],
                    pdict['ema_scores'],
                    pdict['alert_times'],
                    pdict['events'],
                    os.path.join(args.outdir, 'example_detection_timeline.png'),
                    pdict['tuned_threshold'],  # use tuned threshold in the title/annotation
                    args.ema_window
                )

    if not final_fold_results:
        print("No folds produced evaluable results (no overlapping events). Exiting.")
        return

    # --- Aggregate and Report Final Results ---
    all_lead_times = [lt for r in final_fold_results for lt in r['lead_times']]
    detection_rates = [
        (r['tp'] / (r['tp'] + r['fn']))
        for r in final_fold_results if (r['tp'] + r['fn']) > 0
    ]
    total_fp = sum(r['fp'] for r in final_fold_results)
    total_days = sum(r['test_duration_days'] for r in final_fold_results) or 1e-9
    fp_rate_per_day = total_fp / total_days
    tuned_thresholds = [r['tuned_threshold'] for r in final_fold_results]

    print(f"\n--- Final Evaluation Results (tuned per fold) ---")
    if detection_rates:
        print(f"Detection Rate:            {np.mean(detection_rates):.2%} ± {np.std(detection_rates):.2%}")
    else:
        print("Detection Rate:            N/A (no evaluable folds)")
    if all_lead_times:
        print(f"Average Lead Time (mins):  {np.mean(all_lead_times):.2f} ± {np.std(all_lead_times):.2f}")
    else:
        print("Average Lead Time (mins):  N/A")
    print(f"False Positive Rate/day:   {fp_rate_per_day:.2f}")
    print(f"Tuned thresholds (mean ± std): {np.mean(tuned_thresholds):.6f} ± {np.std(tuned_thresholds):.6f}")

    # --- Aggregate, display, and plot feature importances ---
    print("\n--- Average Feature Importance (across all folds) ---")
    summed_importances = defaultdict(float)
    for r in final_fold_results:
        for feature, value in r['importances'].items():
            summed_importances[feature] += value
    avg_importances = {f: v / len(final_fold_results) for f, v in summed_importances.items()}
    sorted_importances = sorted(avg_importances.items(), key=lambda item: item[1], reverse=True)
    for feature, value in sorted_importances:
        print(f"{feature:<15} {value:.3f}")

    bar_attribution(
        vals=[avg_importances.get(k, 0) for k in ['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta']],
        labels=['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta'],
        path=os.path.join(args.outdir, 'average_feature_importance.png'),
        title='Average Feature Importance Across Folds'
    )

    # --- Lead Time Distribution Plot ---
    if all_lead_times:
        plot_lead_time_distribution(all_lead_times, os.path.join(args.outdir, 'lead_time_distribution.png'))
        print("\nGenerated 3 plots in the output directory.")
    else:
        print("\nGenerated 2 plots in the output directory (no lead times to plot).")

if __name__ == '__main__':
    main()
