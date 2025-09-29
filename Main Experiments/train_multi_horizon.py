# FILE: train_multi_horizon.py
import argparse
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# Import from your existing project files
from common import read_json
from risk_model import create_risk_model
from compare_different_dataset_size import make_forward_looking_dataset

def tune_alert_threshold(score_series, events, lower_is_worse=False):
    """
    Finds the best alert threshold on a score series by maximizing the F1-score.
    
    Args:
        score_series (pd.Series): A time-indexed series of risk scores.
        events (list): A list of ground truth event dictionaries.
        lower_is_worse (bool): Set to True if lower scores indicate higher risk.
    
    Returns:
        float: The threshold that yields the best F1-score.
    """
    best_f1 = -1.0
    best_thresh = 0.5

    # Create the ground truth binary array
    y_true = np.zeros(len(score_series), dtype=int)
    for event in events:
        # For proactive labels, the "event" is the pre-failure window
        mask = (score_series.index >= event['start']) & (score_series.index < event['failure_time'])
        y_true[mask] = 1

    smin, smax = score_series.min(), score_series.max()
    if smax == smin:
        return float(smin)

    # Iterate through potential thresholds to find the best one
    for thresh in np.linspace(smin, smax, 100):
        y_pred = (score_series < thresh) if lower_is_worse else (score_series > thresh)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            
    return best_thresh

def main():
    parser = argparse.ArgumentParser(description="Train multiple risk models for different time horizons and tune their thresholds.")
    parser.add_argument('--data', required=True, help="Path to the dataset CSV.")
    parser.add_argument('--outdir', required=True, help="Directory to save the trained models and metadata.")
    parser.add_argument('--horizons', required=True, nargs='+', type=int, help="A list of horizons in minutes (e.g., 30 60 120).")
    parser.add_argument('--ema_window', type=int, default=5, help="EMA window for smoothing scores during threshold tuning.")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events = read_json(event_path)["events"]
    features = ['cpu_pct', 'ram_pct', 'storage_pct', 'snet', 'sri', 'cpu_delta', 'sri_delta']
    
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx]
    train_events = [e for e in events if e['end'] <= df_train['t'].max()]

    print(f"\n--- Starting training for horizons: {args.horizons} ---")
    for h in args.horizons:
        print(f"\n--- Processing model for horizon H={h} minutes ---")
        
        # 1. Create dataset with the specific horizon
        X_train, Y_train, t_train = make_forward_looking_dataset(
            df_train, train_events, features, horizon_min=h
        )
        
        # 2. Create and train a fresh model
        print("Training model...")
        tf.keras.backend.clear_session()
        tf.random.set_seed(args.seed)
        model = create_risk_model(input_dim=X_train.shape[1])
        model.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch, verbose=0)
        
        # 3. Tune the alert threshold
        print("Tuning alert threshold...")
        raw_scores_train = model.predict(X_train).flatten()
        ema_scores_train = pd.Series(raw_scores_train, index=t_train).ewm(span=args.ema_window, adjust=False).mean()
        optimal_threshold = tune_alert_threshold(ema_scores_train, train_events)
        print(f"Found optimal threshold: {optimal_threshold:.4f}")

        # 4. Save the model and its metadata (including the threshold)
        model_path = os.path.join(args.outdir, f'risk_model_h{h}.h5')
        meta_path = os.path.join(args.outdir, f'risk_model_h{h}_meta.json')
        
        model.save(model_path)
        
        metadata = {
            "horizon": h,
            "optimal_threshold": optimal_threshold,
            "ema_window": args.ema_window
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"SUCCESS: Saved model to {model_path}")
        print(f"SUCCESS: Saved metadata to {meta_path}")

if __name__ == '__main__':
    main()