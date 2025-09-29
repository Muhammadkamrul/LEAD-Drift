import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import ConnectionPatch
from adjustText import adjust_text

# Add these two new functions to your plotting.py file

def save_tight(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

# --- NEW, DETAILED COMPARISON PLOT ---
def plot_full_comparison_timeline(timeline_data, all_alerts, events, out_path):
    """
    Generates a multi-panel plot to compare the detailed alert behavior of all methods.
    """
    methods = list(timeline_data.keys())
    fig, axes = plt.subplots(len(methods), 1, figsize=(15, 3 * len(methods)), sharex=True)
    fig.suptitle('Detailed Alert Timeline Comparison', fontsize=16, y=0.99)

    event_to_plot = events[0] if events else None
    if not event_to_plot: return

    for i, (name, ax) in enumerate(zip(methods, axes)):
        score_series, threshold, lower_is_worse = timeline_data[name]
        alert_times = all_alerts[name]

        # Plot the score and threshold
        ax.plot(score_series.index, score_series, label='Method Score', color='blue')
        ax.axhline(y=threshold, color='red', linestyle='--', lw=1.5, label='Alert Threshold')

        # Shade the ground truth window
        ax.axvspan(event_to_plot['start'], event_to_plot['end'], color='orange', alpha=0.2, label='Drift Window')
        ax.axvline(x=event_to_plot['failure_time'], color='red', linestyle=':', label='Failure Time')

        # Find and plot the earliest alert (TP)
        alarms_in_window = alert_times[(alert_times >= event_to_plot['start']) & (alert_times <= event_to_plot['failure_time'])]
        if len(alarms_in_window) > 0:
            ax.axvline(x=alarms_in_window[0], color='green', linestyle='-', lw=2, label='Earliest Alert (TP)')

        # Find and plot all False Positives
        fp_found = False
        for alert_time in alert_times:
            if not any(event['start'] <= alert_time <= event['end'] for event in events):
                label = 'False Positive' if not fp_found else ""
                ax.axvline(x=alert_time, color='magenta', linestyle=':', lw=2, label=label)
                fp_found = True
        
        ax.set_title(name)
        ax.set_ylabel('Score')
        ax.legend(loc='upper left')

    # Set shared x-axis properties
    padding = (event_to_plot['end'] - event_to_plot['start']) * 0.5
    axes[-1].set_xlim(event_to_plot['start'] - padding, event_to_plot['end'] + padding + 50) # Add extra space for FPs
    axes[-1].set_xlabel('Time (minutes)')
    
    fig.subplots_adjust(top=0.95, hspace=0.3)
    save_tight(fig, out_path)

def plot_comparison_barchart(results, out_path):
    """Generates a grouped bar chart comparing metrics across methods."""
    labels = list(results.keys())
    detection_rates = [r['detection_rate'] * 100 for r in results.values()] # as percentage
    lead_times = [r['avg_lead_time'] for r in results.values()]
    fp_rates = [r['fp_rate'] for r in results.values()]

    x = np.arange(len(labels))
    width = 0.25
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 5)) # Increased figure width
    fig.suptitle('Comparison of LEAD-Drift with Baseline Methods', fontsize=16)

    # Detection Rate Plot
    axes[0].bar(x, detection_rates, width, label='Detection Rate', color='tab:blue')
    axes[0].set_ylabel('Detection Rate (%)')
    axes[0].set_title('Effectiveness')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 110)

    # Lead Time Plot
    axes[1].bar(x, lead_times, width, label='Lead Time', color='tab:green')
    axes[1].set_ylabel('Average Lead Time (mins)')
    axes[1].set_title('Proactiveness')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    # False Positive Rate Plot
    axes[2].bar(x, fp_rates, width, label='FP Rate', color='tab:red')
    axes[2].set_ylabel('False Positives (per day)')
    axes[2].set_title('Operational Cost')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    
    save_tight(fig, out_path)

def plot_comparison_timeline(timeline_data, events, out_path):
    """MODIFIED: Plots each score series against its own index for robustness."""
    event_to_plot = events[0] if events else None
    if not event_to_plot: return
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.axvspan(event_to_plot['start'], event_to_plot['end'], color='orange', alpha=0.2, label='Drift Window')
    ax.axvline(x=event_to_plot['failure_time'], color='red', linestyle=':', label='Failure Time')
    
    colors = ['blue', 'green', 'purple', 'brown', 'cyan'] 
    
    for i, (name, (score_series, threshold, lower_is_worse)) in enumerate(timeline_data.items()):
        normalized_score = (score_series - score_series.min()) / (score_series.max() - score_series.min())
        # MODIFIED: Plot against the series' own index, which contains the correct time values.
        ax.plot(score_series.index, normalized_score, label=name, color=colors[i], lw=2)
        
        alert_times = score_series.index[score_series < threshold if lower_is_worse else score_series > threshold]
        alarms_in_window = alert_times[(alert_times >= event_to_plot['start']) & (alert_times <= event_to_plot['failure_time'])]
        if len(alarms_in_window) > 0:
            # The series index is already a numpy array, so simple indexing is fine.
            ax.axvline(x=alarms_in_window[0], color=colors[i], linestyle='--', lw=2, label=f'{name} Alert')

    ax.set_title(f"Detection Timeline Comparison for Event starting at t={event_to_plot['start']}")
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Normalized Score')
    ax.legend(loc='upper left')
    
    padding = (event_to_plot['end'] - event_to_plot['start']) * 0.5
    ax.set_xlim(event_to_plot['start'] - padding, event_to_plot['end'] + padding)
    
    save_tight(fig, out_path)


def plot_performance_tradeoff(thresholds, detection_rates, fp_rates, out_path):
    """REVISED: Plots the trade-off curve with log scale and non-overlapping labels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fp_rates, detection_rates, marker='o', linestyle='-')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())

    texts = []
    for i, threshold in enumerate(thresholds):
        texts.append(ax.text(fp_rates[i], detection_rates[i], f" τ={threshold:.2f}", fontsize=9))
    
    # Use adjust_text to prevent labels from overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        
    ax.set_title('Performance Trade-off: Detection Rate vs. False Positives')
    ax.set_xlabel('False Positive Rate (Alarms per Day, log scale)')
    ax.set_ylabel('Detection Rate (Recall)')
    ax.set_ylim(0, 1.05)
    min_fp_rate = min([r for r in fp_rates if r > 0], default=0.01)
    ax.set_xlim(left=min_fp_rate * 0.5)
    ax.grid(True, which="both", linestyle='--', alpha=0.6)
    save_tight(fig, out_path)

def plot_example_detection_timeline(t, raw_scores, ema_scores, alert_times, events, out_path, ema_threshold, ema_window, label_fontsize=20, legend_fontsize=20, tick_fontsize=20):
    """
    REVISED: Generates a two-panel plot with adjusted spacing and height for better clarity.
    """
    plt.rcParams["xtick.labelsize"] = tick_fontsize
    plt.rcParams["ytick.labelsize"] = tick_fontsize
    plt.rcParams["axes.labelsize"] = label_fontsize
    plt.rcParams["legend.fontsize"] = legend_fontsize

    # MODIFIED: Changed height_ratios to make the bottom panel taller
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False, gridspec_kw={'height_ratios': [1, 2]})
    
    # MODIFIED: Manually adjust the vertical space between the plots to be very small
    fig.subplots_adjust(hspace=0.15)
    
    ax_main, ax_zoom = axes[0], axes[1]

    # --- Top Panel: Overall Timeline (Unchanged) ---
    ax_main.plot(t, raw_scores, color='grey', alpha=0.4, lw=1.0)
    ax_main.plot(t, ema_scores, color='blue', lw=2.0)
    ax_main.axhline(y=ema_threshold, color='red', linestyle='--', lw=1.5)
    #ax_main.set_title('Example Detection Timeline from a Test Fold')
    ax_main.set_ylabel('Risk Score')
    ax_main.set_xlim(min(t), max(t))

    for i, event in enumerate(events):
        ax_main.axvspan(event['start'], event['end'], color='orange', alpha=0.1)
        alarms_in_window = alert_times[(alert_times >= event['start']) & (alert_times <= event['end'])]
        if len(alarms_in_window) > 0:
            ax_main.axvline(x=alarms_in_window[0], color='green', linestyle='-', lw=2)

    fp_found_main = False
    for alert_time in alert_times:
        if not any(event['start'] <= alert_time <= event['end'] for event in events):
            label = '' if not fp_found_main else ""
            ax_main.axvline(x=alert_time, color='magenta', linestyle=':', lw=2, label=label)
            fp_found_main = True
    #ax_main.legend(loc='upper right')

    # --- Bottom Panel: Zoomed-in View (Unchanged logic, but will now be taller) ---
    first_event = events[0] if events else None
    if first_event:
        ax_zoom.plot(t, raw_scores, color='grey', alpha=0.6, lw=1.5, marker='.', markersize=4, label='Raw Risk Score')
        ax_zoom.plot(t, ema_scores, color='blue', lw=2.5, label=f'Risk Score EMA ({ema_window} min)')
        ax_zoom.axhline(y=ema_threshold, color='red', linestyle='--', lw=2, label='EMA Alert Threshold')
        ax_zoom.axvspan(first_event['start'], first_event['end'], color='orange', alpha=0.1,label='Ground Truth Drift' if i == 0 else "")
        
        ax_zoom.axvline(x=first_event['start'], color='black', linestyle=':', label='Drift Start')
        alarms_in_window = alert_times[(alert_times >= first_event['start']) & (alert_times <= first_event['end'])]
        if len(alarms_in_window) > 0:
            ax_zoom.axvline(x=alarms_in_window[0], color='green', linestyle='-', label='Earliest Alert (TP)' if i == 0 else "")
        ax_zoom.axvline(x=first_event['failure_time'], color='red', linestyle=':', label='Failure Time')
        ax_zoom.axvline(x=first_event['end'], color='purple', linestyle=':', label='Restart Time')

        zoom_start = first_event['start'] - 20
        zoom_end = first_event['end'] + 20
        fp_found_zoom = False
        for alert_time in alert_times:
            if not any(e['start'] <= alert_time <= e['end'] for e in events) and (zoom_start <= alert_time <= zoom_end):
                label = 'False Positive' if not fp_found_zoom else ""
                ax_zoom.axvline(x=alert_time, color='magenta', linestyle=':', lw=2.5, label=label)
                fp_found_zoom = True

        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.set_ylim(bottom=-0.05)
        #ax_zoom.set_title(f"Zoomed-in View of '{first_event['type']}' Event")
        ax_zoom.set_xlabel('Time (minutes)')
        ax_zoom.set_ylabel('Risk Score')
        ax_zoom.legend(ncol=2)

        con = ConnectionPatch(xyA=(first_event['start'], ax_main.get_ylim()[0]), xyB=(first_event['start'], ax_zoom.get_ylim()[1]),
                              coordsA="data", coordsB="data",
                              axesA=ax_main, axesB=ax_zoom,
                              color="black", linestyle='--')
        fig.add_artist(con) # Add artist to the figure directly
        con2 = ConnectionPatch(xyA=(first_event['end'], ax_main.get_ylim()[0]), xyB=(first_event['end'], ax_zoom.get_ylim()[1]),
                               coordsA="data", coordsB="data",
                               axesA=ax_main, axesB=ax_zoom,
                               color="black", linestyle='--')
        fig.add_artist(con2)

    # Use regular savefig as we have manually adjusted spacing
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)

def plot_dataset_snapshot(df, events, out_path, label_fontsize=15, legend_fontsize=15, tick_fontsize=15):
    """NEW: Generates a multi-panel figure showing an example of each failure type."""

    plt.rcParams["xtick.labelsize"] = tick_fontsize
    plt.rcParams["ytick.labelsize"] = tick_fontsize
    #plt.rcParams["axes.labelsize"] = label_fontsize
    #plt.rcParams["legend.fontsize"] = legend_fontsize

    event_types = {e['type'] for e in events}
    fig, axes = plt.subplots(len(event_types), 1, figsize=(10, 3 * len(event_types)), sharex=False)
    if len(event_types) == 1: 
        axes = [axes]  # Ensure axes is iterable

    for ax, event_type in zip(axes, sorted(list(event_types))):
        event = next((e for e in events if e['type'] == event_type), None)
        if not event: 
            continue

        padding = (event['end'] - event['start']) * 0.5
        start_t, end_t = int(event['start'] - padding), int(event['end'] + padding)
        df_slice = df[(df['t'] >= start_t) & (df['t'] <= end_t)]
        
        ax.plot(df_slice['t'], df_slice['snet'], label='net_conn', color='green')
        ax.plot(df_slice['t'], df_slice['sri'], label='serv_resp', color='red')
        ax.plot(df_slice['t'], df_slice['cpu_pct'], label='cpu_pct', color='purple', linestyle='--')
        ax.axvspan(event['start'], event['end'], color='orange', alpha=0.1, label='Drift Window')
        
        ax.set_ylabel("KPI Value (%)", fontsize=label_fontsize)

    # Shared xlabel
    axes[-1].set_xlabel("Time (minutes)", fontsize=label_fontsize)

    # Add a single legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=legend_fontsize, loc="lower right", frameon=True)

    save_tight(fig, out_path)



def plot_lead_time_distribution(lead_times, out_path):
    """Unchanged but kept for completeness."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=lead_times, ax=ax)
    ax.set_title('Distribution of Proactive Detection Lead Times Across All Folds')
    ax.set_xlabel('Proactive Lead Time (minutes before failure)')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    save_tight(fig, out_path)

def plot_incident_lifecycle(t, raw_scores, ema_scores, ema_threshold, incident_log, out_path, ema_window):
    """
    Visualizes the risk score timeline and shades the periods where an incident was active.
    """
    fig, ax = plt.subplots(figsize=(15, 4))

    # Plot the scores and threshold
    ax.plot(t, raw_scores, label='Raw Risk Score', color='grey', alpha=0.5, lw=1.0)
    ax.plot(t, ema_scores, label=f'Risk Score EMA ({ema_window} min)', color='blue', lw=2.0)
    ax.axhline(y=ema_threshold, color='red', linestyle='--', lw=1.5, label=f'Alert Threshold ({ema_threshold:.2f})')

    # Shade the duration of each logged incident
    for i, incident in enumerate(incident_log):
        label = 'Detected Incident' if i == 0 else ""
        ax.axvspan(incident['start_time'], incident['end_time'], color='purple', alpha=0.25, label=label)
        # Add a marker for when the incident was opened
        ax.text(incident['start_time'], 1.0, f"Incident Opened\nt={incident['start_time']}", 
                ha='center', va='bottom', fontsize=9, color='purple', rotation=90)

    ax.set_title('Real-World Incident Detection Lifecycle')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Risk Score')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(t), max(t))
    ax.legend(loc='upper right')
    save_tight(fig, out_path)

def plot_proactive_detection_results(t, raw_scores, ema_scores, alert_times, events, out_path_main, out_path_zoom, ema_threshold, ema_window):
    """
    Generates two plots:
    1. An overall timeline of detection results with detailed annotations for TPs and FPs.
    2. A zoomed-in view of the first detected drift event, now including local FPs.
    """
    def get_first_alert(event):
        alarms_in_window = alert_times[(alert_times >= event['start']) & (alert_times <= event['end'])]
        return alarms_in_window[0] if len(alarms_in_window) > 0 else None

    # --- Plot 1: Overall Timeline ---
    fig_main, ax_main = plt.subplots(figsize=(15, 4))

    ax_main.plot(t, raw_scores, label='Raw Risk Score', color='grey', alpha=0.4, lw=1.0)
    ax_main.plot(t, ema_scores, label=f'Risk Score EMA ({ema_window} min)', color='blue', lw=2.0)
    ax_main.axhline(y=ema_threshold, color='red', linestyle='--', lw=1.5, label=f'EMA Alert Threshold ({ema_threshold:.2f})')

    for i, event in enumerate(events):
        ax_main.axvspan(event['start'], event['end'], color='orange', alpha=0.2, label='Ground Truth Drift' if i == 0 else "")
        ax_main.text(event['start'], -0.02, str(event['start']), ha='center', va='top', fontsize=8, color='black')
        ax_main.text(event['end'], -0.02, str(event['end']), ha='center', va='top', fontsize=8, color='black')

        first_alert_time = get_first_alert(event)
        if first_alert_time is not None:
            ax_main.axvline(x=first_alert_time, color='green', linestyle='-', lw=2, label='Earliest Alert (TP)' if i == 0 else "")
            ax_main.text(first_alert_time, 1.0, str(first_alert_time), ha='center', va='bottom', fontsize=9, color='green', rotation=90)
            
    fp_found_main = False
    for alert_time in alert_times:
        is_in_any_event = any(event['start'] <= alert_time <= event['end'] for event in events)
        if not is_in_any_event:
            label = 'False Positive' if not fp_found_main else ""
            ax_main.axvline(x=alert_time, color='magenta', linestyle=':', lw=2, label=label)
            ax_main.text(alert_time, 1.0, str(alert_time), ha='center', va='bottom', fontsize=8, color='magenta', rotation=90)
            fp_found_main = True

    ax_main.set_title('Proactive Detection Timeline with False Positives')
    ax_main.set_xlabel('Time (minutes)')
    ax_main.set_ylabel('Risk Score')
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.set_xlim(min(t), max(t))
    ax_main.legend(loc='upper right')
    save_tight(fig_main, out_path_main)

    # --- Plot 2: Zoomed-in View ---
    first_detected_event = next((event for event in events if get_first_alert(event) is not None), None)

    if first_detected_event:
        fig_zoom, ax_zoom = plt.subplots(figsize=(12, 4))
        
        ax_zoom.plot(t, raw_scores, label='Raw Risk Score', color='grey', alpha=0.6, lw=1.5, marker='.', markersize=4)
        ax_zoom.plot(t, ema_scores, label=f'Risk Score EMA ({ema_window} min)', color='blue', lw=2.5)
        ax_zoom.axhline(y=ema_threshold, color='red', linestyle='--', lw=2, label=f'EMA Alert Threshold ({ema_threshold:.2f})')

        ax_zoom.axvspan(first_detected_event['start'], first_detected_event['end'], color='orange', alpha=0.2, label='Ground Truth Drift')
        ax_zoom.text(first_detected_event['start'], 0, str(first_detected_event['start']), ha='center', va='bottom', fontsize=10)
        ax_zoom.text(first_detected_event['end'], 0, str(first_detected_event['end']), ha='center', va='bottom', fontsize=10)

        first_alert_time = get_first_alert(first_detected_event)
        ax_zoom.axvline(x=first_alert_time, color='green', linestyle='-', lw=2, label='Earliest Alert (TP)')
        ax_zoom.text(first_alert_time + 0.5, 0.5, f'Alert at t={first_alert_time}', ha='left', va='center', fontsize=11, color='green', rotation=90)

        padding = (first_detected_event['end'] - first_detected_event['start']) * 0.5
        zoom_start, zoom_end = first_detected_event['start'] - padding, first_detected_event['end'] + padding
        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.set_ylim(bottom=-0.05)
        
        # --- MODIFIED: Find and plot False Positives that are VISIBLE in the zoom window ---
        fp_found_zoom = False
        for alert_time in alert_times:
            # Check if alert is a false positive
            is_in_any_event = any(event['start'] <= alert_time <= event['end'] for event in events)
            if not is_in_any_event:
                # Check if the false positive is within the zoom plot's x-axis range
                if zoom_start <= alert_time <= zoom_end:
                    label = 'False Positive' if not fp_found_zoom else ""
                    ax_zoom.axvline(x=alert_time, color='magenta', linestyle=':', lw=2.5, label=label)
                    ax_zoom.text(alert_time, 0.9, str(alert_time), ha='center', va='bottom', fontsize=9, color='magenta', rotation=90)
                    fp_found_zoom = True

        ax_zoom.set_title('Zoomed-in View of First Detected Event')
        ax_zoom.set_xlabel('Time (minutes)')
        ax_zoom.set_ylabel('Risk Score')
        ax_zoom.legend(loc='upper left')
        save_tight(fig_zoom, out_path_zoom)

# --- NEW PLOTTING FUNCTION ---
def plot_intelligent_alerting(t, raw_scores, ema_scores, alerts, events, path, ema_threshold, ema_window):
    """
    Visualizes the raw risk score, its EMA, and the generated alerts.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))

    # Plot scores and EMA
    ax.plot(t, raw_scores, label='Raw Risk Score', color='grey', alpha=0.5, lw=1.0)
    ax.plot(t, ema_scores, label=f'Risk Score EMA ({ema_window} min)', color='blue', lw=2.0)
    ax.axhline(y=ema_threshold, color='red', linestyle='--', lw=1.5, label=f'EMA Alert Threshold ({ema_threshold:.2f})')

    # Plot the alert triggers
    alert_times = t[alerts]
    if len(alert_times) > 0:
        ax.plot(alert_times, ema_scores[alerts], 'rx', markersize=8, label='Alert Triggered')

    # Shade the ground truth drift periods
    for i, event in enumerate(events):
        start, end = event['start'], event['end']
        label = 'Ground Truth Drift Period' if i == 0 else ""
        ax.axvspan(start, end, color='orange', alpha=0.25, label=label, zorder=0)

    ax.set_title('Intelligent Alerting with Exponential Moving Average')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Risk Score')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(t), max(t))
    ax.legend(loc='upper right')
    save_tight(fig, path)

def plot_generated_data(df, events, path):
    """
    Visualizes the generated synthetic KPI data and ground truth drift events.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    # Plot the key KPIs
    ax.plot(df['t'], df['snet'], label='snet (%)', color='green', lw=1.5, alpha=0.8)
    ax.plot(df['t'], df['sri'], label='sri (%)', color='red', lw=1.5, alpha=0.8)
    ax.plot(df['t'], df['cpu_pct'], label='cpu_pct (%)', color='purple', linestyle='--', lw=1.2, alpha=0.7)
    
    ax.set_title('Generated Synthetic KPI Data with Ground Truth Events')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('KPI Value (%)')
    ax.set_ylim(-5, 110)
    ax.set_xlim(df['t'].min(), df['t'].max())
    
    # Use axvspan to show ground truth degradation periods (drift window)
    for i, event in enumerate(events):
        start_drift = event['start']
        end_drift = event['end']
        restart_point = end_drift # The point where recovery begins
        
        # Shade the degradation window
        label = 'Drift Window' if i == 0 else ""
        ax.axvspan(start_drift, end_drift, color='orange', alpha=0.3, label=label)
        
        # Mark the start of the drift
        ax.axvline(x=start_drift, color='red', linestyle=':', lw=1.5, ymax=0.9, label='Drift Start' if i == 0 else "")
        
        # Mark the restart point
        ax.axvline(x=restart_point, color='blue', linestyle=':', lw=1.5, ymax=0.9, label='Collector Restart' if i == 0 else "")

    ax.legend(loc='lower left', ncol=3)
    save_tight(fig, path)

def bar_attribution(vals, labels, path, title="", xlabel="", ylabel="", rotate=20, label_fontsize=15, legend_fontsize=15, tick_fontsize=15):
    """Generates a bar chart for visualizing feature importances."""

    plt.rcParams["xtick.labelsize"] = tick_fontsize
    plt.rcParams["ytick.labelsize"] = tick_fontsize
    plt.rcParams["axes.labelsize"] = label_fontsize
    plt.rcParams["legend.fontsize"] = legend_fontsize

    fig, ax = plt.subplots(figsize=(8, 4))
    labels[3] = "net_conn"
    labels[4] = "serv_resp"
    labels[6] = "serv_resp_delta"
    ax.bar(labels, vals)
    #ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0, top=max(vals) * 1.2 if vals else 1)
    ax.tick_params(axis='x', labelrotation=rotate)
    # Add text labels on top of bars
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01 * ax.get_ylim()[1], f"{v:.3f}", ha='center', va='bottom', fontsize=15)
    save_tight(fig, path)



# --- NEW PLOTTING FUNCTIONS ---

def plot_confusion_matrix(cm, class_names, path, title='Confusion Matrix'):
    """
    Creates a heatmap visualization of a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                          xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def plot_risk_scores(t, scores, events, path, threshold=0.5, title='Risk Score Over Time'):
    """
    Plots the predicted risk score and highlights ground truth drift periods.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, scores, label='Predicted Risk Score', color='blue', lw=1.5)
    ax.axhline(y=threshold, color='red', linestyle='--', lw=1.5, label=f'Risk Threshold ({threshold:.2f})')

    # Use axvspan to show ground truth drift periods
    for i, event in enumerate(events):
        start, end = event['start'], event['end']
        # Add a label only to the first span to avoid legend clutter
        label = 'Ground Truth Drift Period' if i == 0 else ""
        ax.axvspan(start, end, color='orange', alpha=0.3, label=label, zorder=0)
        # Mark start and end points
        ax.axvline(x=start, color='gray', linestyle=':', lw=1)
        ax.axvline(x=end, color='gray', linestyle=':', lw=1)

    ax.set_title(title)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Risk Score')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(t), max(t))
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

# --- Other functions from the original file can remain below ---
def kHs_plot(t, kHs, t_fail=None, out=None, title="kHs (operational)", unit="minutes"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.plot(t, kHs, lw=2.4, label='kHs (operational)')
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("kHs (%)")
    ax.set_ylim(0, 102)
    if t_fail is not None:
        ax.axvline(t_fail, color='r', ls=':', lw=2)
        ax.text(t_fail+0.5, 60, f"Failure ({unit})", color='r', rotation=90, va='center', ha='left')
    ax.legend(loc='upper left', frameon=True)
    save_tight(fig, out)

def overlay_kHs_health(
    t, kHs, snet, sri, t_fail, early_t=None, action_t=None, proactive_window=None,
    out=None, title="", unit="minutes", style=1, label_prefix=""):
    fig, ax1 = plt.subplots(figsize=(12, 3.0))
    # kHs
    ax1.plot(t, kHs, lw=2.6, color='#1f77b4', label=f'{label_prefix}kHs (%)', zorder=3)
    ax1.set_ylabel('kHs (%)')
    ax1.set_ylim(0, 102)
    ax1.set_xlabel(f'Time ({unit})')

    ax2 = ax1.twinx()
    if style == 1:
        ax2.plot(t, snet, lw=1.6, color='tab:green', label=f'{label_prefix}Collector2 snet (%)', alpha=0.9)
        ax2.plot(t, sri,  lw=1.6, color='tab:red',   label=f'{label_prefix}Collector2 sri (%)', alpha=0.9)
    else:
        ax2.plot(t, snet, lw=1.2, ls='--', color='tab:green', label=f'{label_prefix}Collector2 snet (%)', alpha=0.9)
        ax2.plot(t, sri,  lw=1.2, ls='--', color='tab:red',   label=f'{label_prefix}Collector2 sri (%)',  alpha=0.9)
    ax2.set_ylabel('Collector health (%)')
    ax2.set_ylim(-2, 104)  # slight vertical offset so 100 lines don't overlap

    # Markers
    if t_fail is not None:
        ax1.axvline(t_fail, color='r', ls=':', lw=2, zorder=2)
        ax1.text(t_fail+0.5, 25, 'Failure (minutes)', rotation=90, color='r', va='center', ha='left')
    if early_t is not None and proactive_window is not None:
        ax1.axvspan(early_t, proactive_window[1], color='orange', alpha=0.15, label='Proactive action window', zorder=1)
        ax1.text(early_t+0.3, 38, 'Early-warning', rotation=90, color='orange', va='center')
        ax1.text(proactive_window[1]+0.3, 38, 'Action-threshold', rotation=90, color='orange', va='center')

    # legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower left', frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

def overlay_two_methods(
    t, kHs_baseline, kHs_lead, snet_bl, sri_bl, snet_lead, sri_lead,
    t_fail, drift_start_time=None, outA=None, outB=None, unit="minutes"):
    
    # --- Style A Plot ---
    fig_a, axes = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, height_ratios=[1,1.1])
    ax1 = axes[0]
    ax1.plot(t, kHs_baseline, lw=2.2, label='kHs baseline (%)', color='#1f77b4')
    ax1.plot(t, kHs_lead,     lw=2.2, label='kHs LEAD-Drift (%)', color='#ff7f0e', ls='--')
    
    if drift_start_time is not None:
        ax1.axvline(drift_start_time, color='green', ls='--', lw=2, label='LEAD-Drift Detection')
        ax1.text(drift_start_time - 0.5, 50, 'Drift Detected', color='green', rotation=90, va='center', ha='right')

    if t_fail:
        ax1.axvline(t_fail, color='r', ls=':', lw=2)
        ax1.text(t_fail + 0.5, 55, 'Failure', color='r', rotation=90, va='center')
        
    ax1.set_ylabel('kHs (%)'); ax1.set_ylim(0,102)
    ax1.legend(loc='lower right', ncol=2)
    
    ax2 = axes[1]
    ax2.plot(t, snet_bl, lw=1.6, color='tab:green', label='BL c2 snet (%)')
    ax2.plot(t, sri_bl,  lw=1.6, color='tab:red',   label='BL c2 sri (%)')
    ax2.plot(t, snet_lead, lw=1.4, ls='--', color='tab:green', alpha=0.9, label='LEAD c2 snet (%)')
    ax2.plot(t, sri_lead,  lw=1.4, ls='--', color='tab:red', alpha=0.9, label='LEAD c2 sri (%)')
    ax2.set_ylabel('Collector health (%)'); ax2.set_ylim(-2,104)
    ax2.set_xlabel(f'Time ({unit})')
    ax2.legend(loc='lower left', ncol=2)
    
    fig_a.tight_layout(); fig_a.savefig(outA, dpi=140); plt.close(fig_a)

    # --- Style B Plot ---
    fig_b, ax1_b = plt.subplots(figsize=(12, 3.6))
    ax1_b.plot(t, kHs_baseline, lw=2.0, color='#1f77b4', label='kHs baseline (%)')
    ax1_b.plot(t, kHs_lead,     lw=2.0, color='#ff7f0e', ls='--', label='kHs LEAD-Drift (%)')
    ax1_b.set_ylabel('kHs (%)'); ax1_b.set_ylim(0,102); ax1_b.set_xlabel(f'Time ({unit})')

    if drift_start_time is not None:
        ax1_b.axvline(drift_start_time, color='green', ls='--', lw=2)
        ax1_b.text(drift_start_time - 0.5, 50, 'Drift Detected', color='green', rotation=90, va='center', ha='right')
    
    if t_fail:
        ax1_b.axvline(t_fail, color='r', ls=':', lw=2)
        ax1_b.text(t_fail + 0.5, 25, 'Failure', color='r', rotation=90, va='center')
        
    ax2_b = ax1_b.twinx()
    ax2_b.plot(t, snet_bl, lw=1.5, color='tab:green', label='BL c2 snet (%)', alpha=0.95)
    ax2_b.plot(t, sri_bl,  lw=1.5, color='tab:red',   label='BL c2 sri (%)',  alpha=0.95)
    ax2_b.plot(t, snet_lead, lw=1.3, ls='--', color='tab:green', label='LEAD c2 snet (%)', alpha=0.8)
    ax2_b.plot(t, sri_lead,  lw=1.3, ls='--', color='tab:red',   label='LEAD c2 sri (%)',  alpha=0.8)
    ax2_b.set_ylabel('Collector health (%)'); ax2_b.set_ylim(-2,104)

    h1, l1 = ax1_b.get_legend_handles_labels()
    h2, l2 = ax2_b.get_legend_handles_labels()
    ax1_b.legend(h1+h2, l1+l2, loc='lower left', ncol=2)
    
    fig_b.tight_layout(); fig_b.savefig(outB, dpi=140); plt.close(fig_b)