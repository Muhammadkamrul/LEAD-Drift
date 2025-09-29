import matplotlib.pyplot as plt
import numpy as np

# keep your existing save_tight(fig, out_path) helper as-is

def save_tight(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def _first_event_of_type(events, event_type):
    for e in events:
        if e.get('type') == event_type:
            return e
    return None

def plot_dataset_snapshot_multi(
    df_list, events_list, labels, out_path,
    label_fontsize=15, legend_fontsize=15, tick_fontsize=15
):
    """
    Create ONE figure from TWO (df, events) pairs.
    Panels are stacked vertically. For each dataset i and each event type present
    in that dataset, we plot KPI time series around the FIRST occurrence of that type.
    """
    assert len(df_list) == 2 and len(events_list) == 2 and len(labels) == 2, \
        "Expect two datasets, two event lists, and two labels."

    # Set tick sizes globally for this plot
    plt.rcParams["xtick.labelsize"] = tick_fontsize
    plt.rcParams["ytick.labelsize"] = tick_fontsize

    # Collect event types per dataset (keep only types that actually exist within each list)
    types_per_ds = []
    for evs in events_list:
        types_per_ds.append(sorted({e.get('type') for e in evs if 'type' in e}))

    # Build a panel list of (ds_index, event_type) in order: all types of ds0, then all types of ds1
    panel_specs = []
    for ds_idx, types in enumerate(types_per_ds):
        for et in types:
            panel_specs.append((ds_idx, et))

    n_panels = len(panel_specs)
    if n_panels == 0:
        raise ValueError("No event types found in the provided events files.")

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    # We’ll gather legend handles just once (KPI lines are identical across panels)
    shared_handles, shared_labels = None, None

    for ax, (ds_idx, event_type) in zip(axes, panel_specs):
        df = df_list[ds_idx]
        events = events_list[ds_idx]
        label_prefix = labels[ds_idx]

        # Find first matching event of this type in this dataset
        event = _first_event_of_type(events, event_type)
        if event is None:
            ax.set_visible(False)
            continue

        # Slice with padding around the event window
        padding = (event['end'] - event['start']) * 0.5
        start_t = int(event['start'] - padding)
        end_t   = int(event['end'] + padding)
        df_slice = df[(df['t'] >= start_t) & (df['t'] <= end_t)]

        # Plot KPIs
        l1 = ax.plot(df_slice['t'], df_slice['snet'], label='net_conn', color='green')[0]
        l2 = ax.plot(df_slice['t'], df_slice['sri'], label='serv_resp', color='red')[0]
        l3 = ax.plot(df_slice['t'], df_slice['cpu_pct'], label='cpu_pct', color='purple', linestyle='--')[0]
        ax.axvspan(event['start'], event['end'], color='orange', alpha=0.1, label='Drift Window')

        # Y label per panel
        ax.set_ylabel("KPI Value (%)", fontsize=label_fontsize)

        # Panel title indicates dataset and event type
        #ax.set_title(f"{label_prefix} — {event_type}", fontsize=label_fontsize)

        # Capture legend handles once (from the first real panel)
        if shared_handles is None:
            handles, labels_txt = ax.get_legend_handles_labels()
            # Remove duplicates while preserving order
            seen = set()
            filtered = []
            filtered_labels = []
            for h, lab in zip(handles, labels_txt):
                if lab not in seen:
                    filtered.append(h)
                    filtered_labels.append(lab)
                    seen.add(lab)
            shared_handles, shared_labels = filtered, filtered_labels

    # Shared X label
    axes[-1].set_xlabel("Time (minutes)", fontsize=label_fontsize)

    # One legend for the whole figure (bottom-right of last panel)
    if shared_handles:
        axes[-1].legend(shared_handles, shared_labels, fontsize=legend_fontsize,
                        loc="lower right", frameon=True)

    # Save with your existing helper
    try:
        save_tight(fig, out_path)
    except NameError:
        # Fallback if save_tight isn't in scope
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
