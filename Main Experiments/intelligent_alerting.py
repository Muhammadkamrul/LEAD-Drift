import os, argparse
import numpy as np, pandas as pd
import tensorflow as tf

# Import functions from your existing scripts
from common import read_json
from train_risk_model import make_dataset 
# MODIFIED: Import the new, advanced plotting function
from plotting import plot_proactive_detection_results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to the full dataset CSV (e.g., synthetic_data.csv)')
    ap.add_argument('--model', required=True, help='Path to the trained risk_model.h5 file')
    ap.add_argument('--outdir', required=True, help='Directory to save plots and results')
    ap.add_argument('--ema_window', type=int, default=5, help='The window size in minutes for the EMA calculation.')
    ap.add_argument('--ema_threshold', type=float, default=0.15, help='The risk score threshold for the EMA to trigger an alert.')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Load Model, Data, and Predict ---
    print("Loading model, data, and predicting scores...")
    model = tf.keras.models.load_model(args.model)
    df = pd.read_csv(args.data)
    
    split_idx = int(0.8 * len(df))
    df_val = df.iloc[split_idx:].copy()

    X_val, _, t_val = make_dataset(df_val)
    raw_risk_scores = model.predict(X_val).flatten()

    # --- 2. Apply Intelligent Alerting Logic ---
    print("Applying EMA trend detection...")
    risk_series = pd.Series(raw_risk_scores, index=t_val)
    EMA_WINDOW = args.ema_window
    ema_risk_scores = risk_series.ewm(span=EMA_WINDOW, adjust=False).mean().to_numpy()
    
    EMA_THRESHOLD = args.ema_threshold
    alerts_active = ema_risk_scores >= EMA_THRESHOLD
    alert_times = t_val[alerts_active]
    
    # --- 3. Per-Event Performance Evaluation ---
    print("\n--- Per-Event Performance Evaluation ---")
    
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events_data = read_json(event_path)
    
    val_events = [e for e in events_data['events'] if e['start'] >= t_val[0]]
    
    true_positives = 0
    false_negatives = 0
    detection_latencies = []
    
    for event in val_events:
        alarms_in_window = alert_times[(alert_times >= event['start']) & (alert_times <= event['end'])]
        
        if len(alarms_in_window) > 0:
            true_positives += 1
            first_alarm_time = alarms_in_window[0]
            ttd = first_alarm_time - event['start']
            detection_latencies.append(ttd)
        else:
            false_negatives += 1
            
    false_positives = 0
    for time in alert_times:
        is_in_any_event = any(event['start'] <= time <= event['end'] for event in val_events)
        if not is_in_any_event:
            false_positives += 1
            
    avg_ttd = np.mean(detection_latencies) if detection_latencies else float('nan')
    
    print(f"Total Drift Events in Validation Set: {len(val_events)}")
    print("-" * 20)
    print(f"True Positives (Events Detected):  {true_positives}")
    print(f"False Negatives (Events Missed):   {false_negatives}")
    print(f"False Positives (False Alarms):    {false_positives}")
    print("-" * 20)
    print(f"Average Time-to-Detect (TTD): {avg_ttd:.2f} minutes")

    # --- 4. MODIFIED: Visualize the Results with New Function ---
    plot_path_main = os.path.join(args.outdir, "proactive_detection_timeline.png")
    plot_path_zoom = os.path.join(args.outdir, "proactive_detection_zoom_plot.png")

    # Call the new plotting function, passing only the validation set events for clarity
    plot_proactive_detection_results(
        t=t_val,
        raw_scores=raw_risk_scores,
        ema_scores=ema_risk_scores,
        alert_times=alert_times, # Pass the actual times of the alerts
        events=val_events, # Pass only the events relevant to the plot
        out_path_main=plot_path_main,
        out_path_zoom=plot_path_zoom,
        ema_threshold=EMA_THRESHOLD,
        ema_window=EMA_WINDOW
    )
    print(f"\nSaved main visualization -> {plot_path_main}")
    print(f"Saved zoomed-in visualization -> {plot_path_zoom}")

if __name__ == '__main__':
    main()