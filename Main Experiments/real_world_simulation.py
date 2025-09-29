import os, argparse
import numpy as np, pandas as pd
import tensorflow as tf

from common import read_json
from train_risk_model import make_dataset 
from plotting import plot_incident_lifecycle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to the full dataset CSV')
    ap.add_argument('--model', required=True, help='Path to the trained risk_model.h5 file')
    ap.add_argument('--outdir', required=True, help='Directory to save plots and results')
    ap.add_argument('--ema_window', type=int, default=5, help='EMA window size.')
    ap.add_argument('--ema_threshold', type=float, default=0.15, help='EMA risk score threshold.')
    ap.add_argument('--recovery_period', type=int, default=15, help='Minutes EMA must be below threshold to close incident.')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Load, Predict, and Calculate EMA (Unchanged) ---
    print("Loading model, data, and predicting scores...")
    model = tf.keras.models.load_model(args.model)
    df = pd.read_csv(args.data)
    X, _, t = make_dataset(df)
    raw_risk_scores = model.predict(X).flatten()
    risk_series = pd.Series(raw_risk_scores, index=t)
    ema_risk_scores = risk_series.ewm(span=args.ema_window, adjust=False).mean()
    
    # --- 2. Simulate Real-World Incident Management (Unchanged) ---
    print("\n--- Starting Real-World Simulation ---")
    system_state = "NORMAL"
    incident_log = []
    current_incident = None
    recovery_counter = 0

    for current_time, ema_score in ema_risk_scores.items():
        if system_state == "NORMAL":
            if ema_score >= args.ema_threshold:
                system_state = "ALERTING"
                current_incident = {"start_time": current_time, "end_time": None, "status": "OPEN"}
                print(f"\033[91m[Time: {current_time}] ALERT: Risk threshold exceeded. New incident opened.\033[0m")
        
        elif system_state == "ALERTING":
            if ema_score < args.ema_threshold:
                recovery_counter += 1
                if recovery_counter >= args.recovery_period:
                    system_state = "NORMAL"
                    current_incident["end_time"] = current_time
                    current_incident["status"] = "CLOSED"
                    incident_log.append(current_incident)
                    print(f"\032[92m[Time: {current_time}] RECOVERY: System stable. Incident closed.\033[0m")
                    current_incident = None
                    recovery_counter = 0
            else:
                recovery_counter = 0

    if current_incident and current_incident['status'] == 'OPEN':
        current_incident['end_time'] = t[-1]
        current_incident['status'] = 'CLOSED_AT_SIM_END'
        incident_log.append(current_incident)

    # --- 3. Visualize the Simulation (Unchanged) ---
    print("\n--- Simulation Complete: Incident Summary ---")
    # ... (Summary printing is now part of the evaluation section) ...

    plot_path = os.path.join(args.outdir, "incident_lifecycle_plot.png")
    plot_incident_lifecycle(t, raw_risk_scores, ema_risk_scores, args.ema_threshold, incident_log, plot_path, args.ema_window)
    print(f"\nSaved incident lifecycle visualization -> {plot_path}")

    # --- 4. NEW: Evaluation Summary ---
    print("\n--- Proactive Detection Evaluation Summary ---")
    event_path = os.path.join(os.path.dirname(args.data), "drift_events.json")
    events_data = read_json(event_path)
    
    lead_times = []
    
    print(f"{'Event Start':<12} {'Failure Time':<13} {'Detection Time':<15} {'Proactive Lead Time (mins)':<28}")
    print("-" * 70)

    for event in events_data["events"]:
        detection_time = None
        # Find the incident that corresponds to this ground truth event
        for incident in incident_log:
            if event['start'] <= incident['start_time'] <= event['failure_time']:
                detection_time = incident['start_time']
                break
        
        if detection_time:
            lead_time = event['failure_time'] - detection_time
            lead_times.append(lead_time)
            print(f"{event['start']:<12} {event['failure_time']:<13} {detection_time:<15} {lead_time:<28}")
        else:
            print(f"{event['start']:<12} {event['failure_time']:<13} {'MISSED':<15} {'N/A':<28}")

    if lead_times:
        avg_lead_time = np.mean(lead_times)
        print("-" * 70)
        print(f"\nAverage Proactive Detection Lead Time: {avg_lead_time:.2f} minutes")
    else:
        print("\nNo events were successfully detected.")

if __name__ == '__main__':
    main()