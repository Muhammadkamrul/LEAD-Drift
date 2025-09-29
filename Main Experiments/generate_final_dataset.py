# FILE: generate_final_dataset.py
import numpy as np
import pandas as pd
import argparse, os, json, random
from plotting import plot_dataset_snapshot

def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

def benign_high_load_period(start_time, duration, kpis):
    """Simulates a period of intense but NORMAL activity."""
    end_time = start_time + duration
    # Ramp up CPU and RAM together to a high but stable state
    kpis['cpu'][start_time:end_time] = np.linspace(kpis['cpu'][start_time-1], 90, duration)
    kpis['ram'][start_time:end_time] = np.linspace(kpis['ram'][start_time-1], 85, duration)
    # SRI/SNET remain healthy
    
def imbalance_failure_event(start_time, kpis, rng):
    """A failure characterized by an imbalance between KPIs."""
    buildup_duration = rng.integers(40, 60)
    crash_duration = 10
    buildup_end = start_time + buildup_duration
    crash_end = buildup_end + crash_duration
    failure_time = crash_end - 1

    # During buildup, CPU goes very high, but RAM is abnormally low
    kpis['cpu'][start_time:crash_end] = np.linspace(kpis['cpu'][start_time-1], 98, buildup_duration + crash_duration)
    kpis['ram'][start_time:buildup_end] = np.linspace(kpis['ram'][start_time-1], 30, buildup_duration) # RAM drops!
    
    # SRI becomes noisy and unstable during buildup
    kpis['sri'][start_time:buildup_end] -= np.abs(rng.normal(0, 3, size=buildup_duration)).cumsum()
    
    # Final crash
    kpis['sri'][buildup_end:crash_end] = np.linspace(kpis['sri'][buildup_end-1], 5, crash_duration)
    kpis['snet'][buildup_end:crash_end] = np.linspace(kpis['snet'][buildup_end-1], 5, crash_duration)

    # Recovery
    # ... (recovery logic can be added here if needed) ...

    return {
        "start": int(start_time), "end": int(crash_end), "failure_time": int(failure_time),
        "type": "imbalance_failure"
    }

def gen(minutes=43200, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t = np.arange(minutes, dtype=int)
    events = []
    
    # Baseline KPIs
    day = 1440.0
    cpu = 45 + 10*np.sin(2*np.pi*(t%day)/day) + rng.normal(0, 2.0, size=minutes)
    ram = 55 + 10*np.sin(2*np.pi*((t+300)%day)/day) + rng.normal(0, 2.2, size=minutes)
    snet = 100 - np.abs(rng.normal(0, 0.5, size=minutes))
    sri  = 100 - np.abs(rng.normal(0, 0.5, size=minutes))
    kpis = {'cpu': cpu, 'ram': ram, 'snet': snet, 'sri': sri}

    # 1. Add Benign High-Load Periods (these are NOT failure events)
    num_benign_events = minutes // 2880 # ~ twice a day
    for _ in range(num_benign_events):
        start_time = rng.integers(300, minutes - 300)
        # Ensure it doesn't overlap with a real failure scheduled later
        if not any(e['start'] - 300 <= start_time <= e['end'] + 300 for e in events):
             benign_high_load_period(start_time, rng.integers(60, 120), kpis)

    # 2. Add the real, challenging failure events
    num_failures = minutes // 1440
    for _ in range(num_failures):
        start_time = rng.integers(300, minutes - 300)
        if not any(e['start'] - 300 <= start_time <= e['end'] + 300 for e in events):
            event = imbalance_failure_event(start_time, kpis, rng)
            events.append(event)
            
    def bound(x): return np.clip(x, 0, 100)
    cpu, ram, snet, sri = bound(cpu), bound(ram), bound(snet), bound(sri)

    df = pd.DataFrame({ "t": t, "cpu_pct": cpu, "ram_pct": ram, "snet": snet, "sri": sri })
    df['storage_pct'] = 50 + 5*np.sin(2*np.pi*((t+900)%day)/day)
    
    return df, sorted(events, key=lambda x: x['start'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=int, default=43200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    output_dir = os.path.dirname(args.out)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    df, events = gen(args.minutes, args.seed)
    df.to_csv(args.out, index=False)
    event_path = os.path.join(output_dir, "drift_events.json")
    write_json(event_path, {"events": events})
    
    print(f"Saved FINAL dataset -> {args.out}")
    print(f"Saved drift event ground truth -> {event_path}")

    snapshot_path = os.path.join(output_dir, "final_dataset_snapshot.png")
    plot_dataset_snapshot(df, events, snapshot_path)
    print(f"Saved visualization of the new dataset -> {snapshot_path}")

if __name__ == "__main__":
    main()