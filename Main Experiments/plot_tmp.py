# FILE: generate_snapshot.py
import pandas as pd
from common import read_json
from plot_tmp_2 import plot_dataset_snapshot_multi
import os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs=2, required=True, help="Two CSV paths")
    ap.add_argument('--events', nargs=2, required=True, help="Two JSON paths")
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--labels', nargs=2, default=None,
                    help="Optional labels for the two datasets (e.g., 'simA simB')")
    ap.add_argument('--label_fontsize', type=int, default=15)
    ap.add_argument('--legend_fontsize', type=int, default=15)
    ap.add_argument('--tick_fontsize', type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data_paths = args.data
    event_paths = args.events
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in data_paths]

    print(f"Loading data from:\n  1) {data_paths[0]}\n  2) {data_paths[1]}")
    df1 = pd.read_csv(data_paths[0])
    df2 = pd.read_csv(data_paths[1])

    print(f"Loading events from:\n  1) {event_paths[0]}\n  2) {event_paths[1]}")
    events1 = read_json(event_paths[0])['events']
    events2 = read_json(event_paths[1])['events']

    print("Generating combined dataset snapshot plot...")
    out_path = os.path.join(args.outdir, 'dataset_snapshot_combined.png')
    plot_dataset_snapshot_multi(
        [df1, df2], [events1, events2], labels, out_path,
        label_fontsize=args.label_fontsize,
        legend_fontsize=args.legend_fontsize,
        tick_fontsize=args.tick_fontsize
    )
    print(f"Saved snapshot to {out_path}")

if __name__ == '__main__':
    main()
