# FILE: generate_snapshot.py
import pandas as pd
from common import read_json
from plotting import plot_dataset_snapshot
import os, argparse

def main():
    # ... (Argument parsing is unchanged) ...
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    DATA_PATH =args.data
    EVENT_PATH = args.events

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    events = read_json(EVENT_PATH)['events']

    print("Generating dataset snapshot plot...")
    plot_dataset_snapshot(df, events, os.path.join(args.outdir, 'dataset_snapshot.png'))
    print(f"Saved snapshot to {args.outdir}")

if __name__ == '__main__':
    main()