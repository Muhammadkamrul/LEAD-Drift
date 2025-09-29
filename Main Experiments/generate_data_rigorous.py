import numpy as np
import pandas as pd
import argparse, os, json, random
from plotting import plot_generated_data

def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

# --- NEW: Define multiple, distinct failure event templates ---

def slow_degradation_event(start_time, kpis):
    """A gradual 45-minute brownout with a moderate CPU spike."""
    span_degrade = 45
    span_recover = 15
    end_degrade = start_time + span_degrade
    
    kpis['snet'][start_time:end_degrade] = np.linspace(kpis['snet'][start_time-1], 10, span_degrade)
    kpis['sri'][start_time:end_degrade]  = np.linspace(kpis['sri'][start_time-1], 10, span_degrade)
    kpis['cpu'][start_time:end_degrade] += np.linspace(0, 35, span_degrade)
    
    # Recovery
    start_recover = end_degrade
    end_recover = start_recover + span_recover
    kpis['snet'][start_recover:end_recover] = np.linspace(kpis['snet'][start_recover-1], 100, span_recover)
    kpis['sri'][start_recover:end_recover] = np.linspace(kpis['sri'][start_recover-1], 100, span_recover)
    kpis['cpu'][start_recover:end_recover] = np.linspace(kpis['cpu'][start_recover-1], 55, span_recover)
    
    return {"start": start_time, "end": end_degrade, "failure_time": end_degrade - 1, "type": "slow_degradation"}

def sudden_crash_event(start_time, kpis):
    """A rapid 10-minute failure simulating a software crash."""
    span_degrade = 10
    span_recover = 20
    end_degrade = start_time + span_degrade

    kpis['snet'][start_time:end_degrade] = np.linspace(kpis['snet'][start_time-1], 0, span_degrade)
    kpis['sri'][start_time:end_degrade]  = np.linspace(kpis['sri'][start_time-1], 0, span_degrade)
    kpis['cpu'][start_time:end_degrade] += np.geomspace(1, 50, span_degrade) # Sharp CPU spike

    # Recovery
    start_recover = end_degrade
    end_recover = start_recover + span_recover
    kpis['snet'][start_recover:end_recover] = 100
    kpis['sri'][start_recover:end_recover]  = 100
    kpis['cpu'][start_recover:end_recover] = 55
    
    return {"start": start_time, "end": end_degrade, "failure_time": end_degrade - 1, "type": "sudden_crash"}

def resource_leak_event(start_time, kpis):
    """A very slow leak affecting RAM over 3 hours, culminating in a failure."""
    span_degrade = 180
    span_recover = 20
    end_degrade = start_time + span_degrade

    # RAM and CPU slowly increase over a long period
    kpis['ram'][start_time:end_degrade] += np.linspace(0, 30, span_degrade)
    kpis['cpu'][start_time:end_degrade] += np.linspace(0, 25, span_degrade)
    # snet/sri only start failing in the last 30 minutes
    sri_fail_start = end_degrade - 30
    kpis['sri'][sri_fail_start:end_degrade] = np.linspace(kpis['sri'][sri_fail_start-1], 15, 30)
    
    # Recovery
    start_recover = end_degrade
    end_recover = start_recover + span_recover
    kpis['sri'][start_recover:end_recover] = 100
    kpis['ram'][start_recover:end_recover] = np.linspace(kpis['ram'][start_recover-1], 62, span_recover)
    kpis['cpu'][start_recover:end_recover] = np.linspace(kpis['cpu'][start_recover-1], 55, span_recover)
    
    return {"start": start_time, "end": end_degrade, "failure_time": end_degrade - 1, "type": "resource_leak"}


def gen(minutes=43200, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t = np.arange(minutes, dtype=int)
    
    # Define the available failure types
    event_generators = [slow_degradation_event, sudden_crash_event, resource_leak_event]
    events = []
    
    # Baseline KPIs
    day = 1440.0
    cpu = 55 + 8*np.sin(2*np.pi*(t%day)/day) + rng.normal(0, 2.0, size=minutes)
    ram = 62 + 7*np.sin(2*np.pi*((t+300)%day)/day) + rng.normal(0, 2.2, size=minutes)
    sto = 56 + 6*np.sin(2*np.pi*((t+600)%day)/day) + rng.normal(0, 2.5, size=minutes)
    snet = 100 - np.abs(rng.normal(0, 0.4, size=minutes))
    sri  = 100 - np.abs(rng.normal(0, 0.5, size=minutes))
    kpis = {'cpu': cpu, 'ram': ram, 'snet': snet, 'sri': sri}

    # MODIFIED: Randomly select a failure type at each interval
    interval = 1440 # ~ every 24 hours
    for start_time in range(interval, minutes, interval):
        if start_time + 200 < minutes: # Ensure enough room for the event
            generator = random.choice(event_generators)
            event = generator(start_time, kpis)
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
    # ... main function remains the same ...
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=int, default=43200) # 1 month default
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # ... rest of main ...
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