import numpy as np
import pandas as pd
import argparse, os
import json
from plotting import plot_generated_data

def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

def degradation_event(start_time, kpis, span_degrade=45, span_recover=15):
    """
    MODIFIED: Simulates a more severe degradation and logs the exact failure_time.
    """
    cpu, snet, sri = kpis['cpu'], kpis['snet'], kpis['sri']
    
    # --- 1. Degradation Phase (More Severe) ---
    end_degrade = start_time + span_degrade
    # Drop from the current value down to a near-zero (5%) failure state
    snet[start_time:end_degrade] = np.linspace(snet[start_time-1], 5, span_degrade)
    sri[start_time:end_degrade]  = np.linspace(sri[start_time-1], 5, span_degrade)
    # More severe CPU spike
    cpu[start_time:end_degrade] += np.linspace(0, 40, span_degrade)

    # --- 2. Recovery Phase (Restart) ---
    start_recover = end_degrade
    end_recover = start_recover + span_recover
    snet[start_recover:end_recover] = np.linspace(snet[start_recover-1], 100, span_recover)
    sri[start_recover:end_recover] = np.linspace(sri[start_recover-1], 100, span_recover)
    cpu[start_recover:end_recover] = np.linspace(cpu[start_recover-1], 55, span_recover)
    
    # NEW: Define the exact point of failure
    failure_time = end_degrade - 1
    
    return {
        "start": start_time, 
        "end": end_degrade, 
        "failure_time": failure_time, # Add failure time to ground truth
        "type": "collector_degradation"
    }


def gen(minutes=2880, seed=7):
    # This function's logic remains the same, but it now calls the improved degradation_event
    rng = np.random.default_rng(seed)
    t = np.arange(minutes, dtype=int)
    events = []
    day = 1440.0
    cpu = 55 + 8*np.sin(2*np.pi*(t%day)/day) + rng.normal(0, 2.0, size=minutes)
    ram = 62 + 7*np.sin(2*np.pi*((t+300)%day)/day) + rng.normal(0, 2.2, size=minutes)
    sto = 56 + 6*np.sin(2*np.pi*((t+600)%day)/day) + rng.normal(0, 2.5, size=minutes)
    snet = 100 - np.abs(rng.normal(0, 0.4, size=minutes))
    sri  = 100 - np.abs(rng.normal(0, 0.5, size=minutes))
    kpis = {'cpu': cpu, 'snet': snet, 'sri': sri}

    interval = 720 
    for start_time in range(interval, minutes, interval): # Start a bit later
        if start_time + 100 < minutes: 
            event = degradation_event(start_time, kpis)
            events.append(event)
            
    for _ in range(20):
        start = rng.integers(0, minutes - 30)
        span = rng.integers(5, 15)
        is_in_brownout = any(evt['start'] <= start <= evt['end'] for evt in events)
        if not is_in_brownout:
            cpu[start:start+span] += rng.normal(0, 5, span)
            sri[start:start+span] -= np.abs(rng.normal(0, 2, span))
            
    def bound(x): return np.clip(x, 0, 100)
    cpu, ram, sto, snet, sri = bound(cpu), bound(ram), bound(sto), bound(snet), bound(sri)

    df = pd.DataFrame({ "t": t, "cpu_pct": cpu, "ram_pct": ram, "storage_pct": sto, "snet": snet, "sri": sri })
    return df, events

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=int, default=10000) # Increased default
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df, events = gen(args.minutes, args.seed)
    df.to_csv(args.out, index=False)

    event_path = os.path.join(os.path.dirname(args.out), "drift_events.json")
    write_json(event_path, {"events": events})
    
    print(f"Saved synthetic dataset -> {args.out}")
    print(f"Saved drift event ground truth -> {event_path}")

    plot_path = os.path.join(os.path.dirname(args.out), "synthetic_data_plot.png")
    plot_generated_data(df, events, plot_path)
    print(f"Saved visualization of generated data -> {plot_path}")

if __name__ == "__main__":
    main()