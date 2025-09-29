# FILE: generate_expert_data.py
import numpy as np
import pandas as pd
import argparse, os, json, random
from plotting import plot_dataset_snapshot

def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

# --- THE NEW, CHALLENGING FAILURE TYPE ---
def systemic_pressure_event(start_time, kpis, rng):
    """
    Simulates a hidden failure where SRI/SNET look healthy while CPU/RAM
    show severe stress, followed by a sudden crash.
    """
    # Phase 1: Long buildup period
    buildup_duration = rng.integers(60, 90)
    crash_duration = 5
    
    buildup_end = start_time + buildup_duration
    crash_end = buildup_end + crash_duration
    failure_time = crash_end - 1

    # During buildup, CPU and RAM show clear signs of stress
    kpis['cpu'][start_time:buildup_end] = np.linspace(kpis['cpu'][start_time-1], 98, buildup_duration)
    kpis['ram'][start_time:buildup_end] = np.linspace(kpis['ram'][start_time-1], 95, buildup_duration)
    
    # CRITICAL: During this time, SRI and SNET are forced to look healthy
    kpis['sri'][start_time:buildup_end] = 100 - np.abs(rng.normal(0, 0.2, size=buildup_duration))
    kpis['snet'][start_time:buildup_end] = 100 - np.abs(rng.normal(0, 0.2, size=buildup_duration))
    
    # Phase 2: The system finally crashes
    kpis['sri'][buildup_end:crash_end] = np.linspace(kpis['sri'][buildup_end-1], 0, crash_duration)
    kpis['snet'][buildup_end:crash_end] = np.linspace(kpis['snet'][buildup_end-1], 0, crash_duration)
    
    # Recovery Phase
    span_recover = 20
    start_recover = crash_end
    end_recover = start_recover + span_recover
    kpis['cpu'][start_recover:end_recover] = np.linspace(kpis['cpu'][start_recover-1], 55, span_recover)
    kpis['ram'][start_recover:end_recover] = np.linspace(kpis['ram'][start_recover-1], 62, span_recover)
    kpis['sri'][start_recover:end_recover] = 100
    kpis['snet'][start_recover:end_recover] = 100

    return {
        "start": int(start_time), "end": int(crash_end), "failure_time": int(failure_time),
        "type": "systemic_pressure"
    }

def gen(minutes=43200, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t = np.arange(minutes, dtype=int)
    events = []
    
    # Baseline KPIs
    day = 1440.0
    cpu = 55 + 8*np.sin(2*np.pi*(t%day)/day) + rng.normal(0, 2.0, size=minutes)
    ram = 62 + 7*np.sin(2*np.pi*((t+300)%day)/day) + rng.normal(0, 2.2, size=minutes)
    snet = 100 - np.abs(rng.normal(0, 0.5, size=minutes))
    sri  = 100 - np.abs(rng.normal(0, 0.5, size=minutes))
    kpis = {'cpu': cpu, 'ram': ram, 'snet': snet, 'sri': sri}

    # Generate only the new, challenging failure type
    num_failures = minutes // 1440
    for _ in range(num_failures):
        start_time = rng.integers(300, minutes - 300)
        if not any(e['start'] - 300 <= start_time <= e['end'] + 300 for e in events):
            event = systemic_pressure_event(start_time, kpis, rng)
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
    
    print(f"Saved EXPERT dataset -> {args.out}")
    print(f"Saved drift event ground truth -> {event_path}")

    snapshot_path = os.path.join(output_dir, "expert_dataset_snapshot.png")
    from plotting import plot_dataset_snapshot
    plot_dataset_snapshot(df, events, snapshot_path)
    print(f"Saved visualization of the new dataset -> {snapshot_path}")

if __name__ == "__main__":
    main()