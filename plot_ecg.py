import os
import numpy as np
import pandas as pd

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
# from biosppy.signals import ecg
# import neurokit2 as nk

import sys
sys.path.append(os.path.abspath('src'))
# from data import load_ecgs_wfdb, load_sample

def plot_ecg(ecg_waveform, additional_waveforms=None, additional_labels=None, legend=None):
    if additional_waveforms is not None:
        cmap = plt.cm.Blues  # Use Matplotlib's built-in 'Blues' colormap
        n = len(additional_waveforms)
        additional_colors = [cmap(i/n) for i in range(1, n+1)]

    lead_mapping = {0: 'I',   1: 'II',  2: 'III',
                    3: 'aVR', 4: 'aVL', 5: 'aVF',
                    6: 'V1',  7: 'V2',  8: 'V3',
                    9: 'V4',  10: 'V5', 11: 'V6'}
    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    axs = axs.flatten()

    if additional_waveforms is not None:
        for i, waveform in enumerate(additional_waveforms):
            for j in range(12):
                if (legend is not None) & (j == 0):
                    axs[j].plot(waveform[:, j], color=additional_colors[i], linewidth=1, alpha=1, label=legend[i])
                else:
                    axs[j].plot(waveform[:, j], color=additional_colors[i], linewidth=1, alpha=1)

    for i in range(12):    
        if (legend is not None) & (i == 0):
            axs[i].plot(ecg_waveform[:, i], color='black', label="True")
        else:
            axs[i].plot(ecg_waveform[:, i], color='black')

        axs[i].text(0.95, 0.95, lead_mapping[i], transform=axs[i].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

    if additional_labels is not None:
        for i, label in enumerate(additional_labels):
            axs[0].text(0.05, 0.95-i*0.10, label, transform=axs[0].transAxes, fontsize=12, va='top', ha='left')

    for ax in axs:
        # Set X_lim to 0-ecg_length
        ax.set_xlim(0, ecg_waveform.shape[0])

    if legend is not None:
        handles, labels = axs[0].get_legend_handles_labels()
        # Assuming 'True' is the last label and you want it to be first
        order = [len(handles)-1] + list(range(len(handles)-1))
        axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize=10, frameon=False)

    return fig

def plot_lead_ii(data, title=None):
    if len(data.shape) == 2:
        lead_ii = data[:, 1]
    else:
        lead_ii = data

    full_lead_samples = lead_ii.shape[0]  # 10 seconds worth of samples for the full lead II

    sampling_rate = 500

    large_box_duration = 0.20  # seconds for a large box
    small_box_duration = 0.04  # seconds for a small box

    # Calculating samples per large and small box
    samples_per_large_box = int(large_box_duration * sampling_rate)
    samples_per_small_box = int(small_box_duration * sampling_rate)

    def draw_ecg_grid(ax, x_max):
        # Determine the number of large and small boxes to draw based on the axis limits
        xmin, xmax = 0, x_max
        ymin, ymax = -1, 1

        # Calculate the number of horizontal and vertical lines needed
        num_major_vlines = int(xmax / samples_per_large_box) + 1
        num_minor_vlines = int(xmax / samples_per_small_box) + 1
        num_major_hlines = int((ymax - ymin) / 0.5) + 1
        num_minor_hlines = int((ymax - ymin) / 0.1) + 1

        # Major grid lines (large boxes)
        for i in range(num_major_vlines):
            ax.vlines(i * samples_per_large_box, ymin, ymax, color='grey', linestyles='-', linewidth=0.3, alpha=0.7)
        for j in np.arange(ymin, ymax, 0.5):
            ax.hlines(j, xmin, xmax, color='grey', linestyles='-', linewidth=0.3, alpha=0.7)

        # Minor grid lines (small boxes)
        for i in range(num_minor_vlines):
            ax.vlines(i * samples_per_small_box, ymin, ymax, color='grey', linestyles='-', linewidth=0.1, alpha=0.2)
        for j in np.arange(ymin, ymax, 0.1):
            ax.hlines(j, xmin, xmax, color='grey', linestyles='-', linewidth=0.1, alpha=0.2)
        
        # Remove original ticks as grid is directly drawn
        ax.set_xticks([])
        ax.set_yticks([])
        
    # New figure and axes for the realistic ECG chart
    fig = plt.figure(figsize=(4, 3), dpi=600, constrained_layout=True) 
    gs = fig.add_gridspec(1, 1, hspace=0, wspace=0.0)  # 4 rows now, to include the 10s II lead

    ax = fig.add_subplot(gs[0, :])  # Span all columns for the last row
    draw_ecg_grid(ax, full_lead_samples)  # Draw the grid for the full lead II

    ax.plot(lead_ii, lw=0.5, color='black', alpha=1)
    # ax.text(0, 1, 'II', transform=ax.transAxes, fontsize=14, color='black', va='top', ha='left', weight='bold')

    # Set x lim
    ax.set_xlim(0, full_lead_samples)
    ax.set_ylim(-1, 1)  # Set y lim
    ax.set_aspect('auto')  # Adjust aspect ratio if needed
    ax.set_facecolor('#ffe6e6')  # Set the pink background color
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set plot facecolor to #ffe6e6
    fig.set_facecolor('#ffe6e6')

    if title is not None:
        fig.suptitle(title)

    # fig.tight_layout()
    return fig

def plot_ecg_strip(data, title=None, additional_ecg=None):
    samples_per_lead = int(2.5 * 500)
    full_lead_samples = 10 * 500  # 10 seconds worth of samples for the full lead II

    sampling_rate = 500

    # Define new lead arrangement for rows
    lead_mapping = {
        0: 'I',   1: 'II',  2: 'III',
        3: 'aVR', 4: 'aVL', 5: 'aVF',
        6: 'V1',  7: 'V2',  8: 'V3',
        9: 'V4',  10: 'V5', 11: 'V6'
    }

    lead_arrangement = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6']
    ]
    lead_mapping_rev = {v: k for k, v in lead_mapping.items()}

    # Let's create a more realistic ECG chart with grid lines and proper lead arrangement.
    # The provided image has a pink graph paper background with grid lines typical of an ECG report.
    # Each lead's graph is separated by a black line.
    large_box_duration = 0.20  # seconds for a large box
    small_box_duration = 0.04  # seconds for a small box

    # Calculating samples per large and small box
    samples_per_large_box = int(large_box_duration * sampling_rate)
    samples_per_small_box = int(small_box_duration * sampling_rate)


    def draw_ecg_grid(ax, x_max):
        # Determine the number of large and small boxes to draw based on the axis limits
        xmin, xmax = 0, x_max
        ymin, ymax = -1, 1

        # Calculate the number of horizontal and vertical lines needed
        num_major_vlines = int(xmax / samples_per_large_box) + 1
        num_minor_vlines = int(xmax / samples_per_small_box) + 1
        num_major_hlines = int((ymax - ymin) / 0.5) + 1
        num_minor_hlines = int((ymax - ymin) / 0.1) + 1

        # Major grid lines (large boxes)
        for i in range(num_major_vlines):
            ax.vlines(i * samples_per_large_box, ymin, ymax, color='grey', linestyles='-', linewidth=0.6, alpha=0.7)
        for j in np.arange(ymin, ymax, 0.5):
            ax.hlines(j, xmin, xmax, color='grey', linestyles='-', linewidth=0.6, alpha=0.7)

        # Minor grid lines (small boxes)
        for i in range(num_minor_vlines):
            ax.vlines(i * samples_per_small_box, ymin, ymax, color='grey', linestyles='-', linewidth=0.2, alpha=0.2)
        for j in np.arange(ymin, ymax, 0.1):
            ax.hlines(j, xmin, xmax, color='grey', linestyles='-', linewidth=0.2, alpha=0.2)
        
        # Remove original ticks as grid is directly drawn
        ax.set_xticks([])
        ax.set_yticks([])
        
    # New figure and axes for the realistic ECG chart
    fig = plt.figure(figsize=(12, 6), dpi=600, constrained_layout=True) 
    gs = fig.add_gridspec(4, 4, hspace=0, wspace=0.02)  # 4 rows now, to include the 10s II lead

    # Loop through the grid and populate each subplot with the respective lead data
    for row, leads in enumerate(lead_arrangement):
        for col, lead in enumerate(leads):
            # ax = axes[row, col]
            ax = fig.add_subplot(gs[row, col])
            lead_index = lead_mapping_rev[lead]  # Get the index of the lead
            lead_data = data[:samples_per_lead, lead_index]  # Get the data slice for the lead
            # Set up the grid to look like ECG paper
            draw_ecg_grid(ax, samples_per_lead)

            if additional_ecg is not None:
                add_lead_data = additional_ecg[:samples_per_lead, lead_index]
                ax.plot(add_lead_data, lw=0.5, color='red', linestyle='-', alpha=0.8)
                alpha= 1
            else:
                alpha = 1

            # Plot the lead data
            ax.plot(lead_data, lw=1, color='black', alpha=alpha)


            ax.set_xlim(0, samples_per_lead)  # Set x lim
            ax.set_ylim(-1, 1)  # Set y lim
            # Add the lead label as a text inset
            ax.text(0.03, 0.85, lead, transform=ax.transAxes, fontsize=14, color='black', va='top', ha='left', weight='bold')

            # Remove the spines (borders) of each subplot
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_aspect('auto')
            ax.set_facecolor('#ffe6e6')  # A light pink that resembles the color of ECG paper

    ax = fig.add_subplot(gs[3, :])  # Span all columns for the last row
    lead_II_data = data[:full_lead_samples, lead_mapping_rev['II']]  # Get full 10s data for lead II
    draw_ecg_grid(ax, full_lead_samples)  # Draw the grid for the full lead II

    if additional_ecg is not None:
        add_lead_data = additional_ecg[:full_lead_samples, lead_mapping_rev['II']]
        ax.plot(add_lead_data, lw=0.5, color='red', linestyle='-', alpha=0.5)

    ax.plot(lead_II_data, lw=1, color='black', alpha=alpha)
    ax.text(0.01, 0.8, 'II', transform=ax.transAxes, fontsize=14, color='black', va='top', ha='left', weight='bold')

    # Set x lim
    ax.set_xlim(0, full_lead_samples)
    ax.set_ylim(-1, 1)  # Set y lim
    ax.set_aspect('auto')  # Adjust aspect ratio if needed
    ax.set_facecolor('#ffe6e6')  # Set the pink background color
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set plot facecolor to #ffe6e6
    fig.set_facecolor('#ffe6e6')

    if title is not None:
        fig.suptitle(title)

    # fig.tight_layout()
    return fig

def segment_wf(waveform, hb_lead = 1):
    try:
        signals, info = nk.ecg_process(waveform[:,hb_lead], sampling_rate=500)
        # signals, del_waves = nk.ecg_delineate(waveform[:,hb_lead], rpeaks, sampling_rate=500, show=False, show_type="all")
        ecg_rpeaks = info["ECG_R_Peaks"]
        ecg_rpeaks = ecg_rpeaks[ecg_rpeaks < 4800]
        ecg_rpeaks = ecg_rpeaks[ecg_rpeaks > 100]
    except:
        ecg_out = ecg.ecg(signal=waveform[:, hb_lead], sampling_rate=500., show=False, interactive=False)
        ecg_out = ecg.correct_rpeaks(signal=waveform[:, hb_lead], rpeaks=ecg_out['rpeaks'], sampling_rate=500.)
        ecg_rpeaks = ecg_out['rpeaks']

    hb_templates = np.zeros((len(ecg_rpeaks), 300, 12))

    for i in range(12):
        hb_templates[:, :, i], _ = ecg.extract_heartbeats(signal=waveform[:, i], rpeaks=ecg_rpeaks, sampling_rate=500.)
    
    mean_hb = np.mean(hb_templates, axis=0)
    return mean_hb, hb_templates

def plot_segmented_ecg(waveform, additional_waveforms=None, additional_labels=None, show_components=True, legend=None):
    if additional_waveforms is not None:
        cmap = plt.cm.Blues #viridis  # Use Matplotlib's built-in 'Blues' colormap
        n = len(additional_waveforms)
        additional_colors = [cmap(i/n) for i in range(1, n+1)]

    fig, axs = plt.subplots(6, 2, figsize=(15, 15))
    axs = axs.flatten()

    lead_mapping = {0: 'I',   1: 'II',  2: 'III',
                3: 'aVR', 4: 'aVL', 5: 'aVF',
                6: 'V1',  7: 'V2',  8: 'V3',
                9: 'V4',  10: 'V5', 11: 'V6'}
    
    orig_mean_hb, orig_hb_templates = segment_wf(waveform)
    
    # Add line at zero
    for i in range(12):
        axs[i].axhline(0, color='black', linestyle='--', alpha=0.5)

    # Additional components
    if additional_waveforms is not None:
        for i, waveform in enumerate(additional_waveforms):
            mean_hb, hb_templates = segment_wf(waveform)
            # hb_templates = hb_templates[:5]
            for j in range(12):
                for k in range(len(hb_templates)):
                    if show_components:
                        axs[j].plot(hb_templates[k, :, j], color=additional_colors[i], linewidth=0.5, alpha=0.3)

    # Original components
    if show_components:
        for i in range(12):
            for j in range(len(orig_hb_templates)):
                axs[i].plot(orig_hb_templates[j, :, i], color='black', linewidth=0.5, alpha=0.3)

    # Additional average
    if additional_waveforms is not None:
        for i, waveform in enumerate(additional_waveforms):
            mean_hb, hb_templates = segment_wf(waveform)
            for j in range(12):
                if (legend is not None) & (j == 0):
                    axs[j].plot(mean_hb[:, j], color=additional_colors[i], linewidth=1.5, alpha=1, label=legend[i])
                else:
                    axs[j].plot(mean_hb[:, j] , color=additional_colors[i], linewidth=1.5, alpha=1)

    # Original average
    for i in range(12):
        if (legend is not None) & (i == 0):
            axs[i].plot(orig_mean_hb[:, i],  linewidth=1.5, color='black', label="True")
        else:
            axs[i].plot(orig_mean_hb[:, i],  linewidth=1.5, color='black')
        
        # Add lead labels
        axs[i].text(0.95, 0.95, lead_mapping[i], transform=axs[i].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

    # Additional labels
    if additional_labels is not None:
        for i, label in enumerate(additional_labels):
            axs[0].text(0.05, 0.95-i*0.10, label, transform=axs[0].transAxes, fontsize=12, va='top', ha='left')
        
    # Set X_lim to 0-ecg_length
    for ax in axs:
        ax.set_xlim(0, mean_hb.shape[0])
    
    if legend is not None:
        handles, labels = axs[0].get_legend_handles_labels()
        # Assuming 'True' is the last label and you want it to be first
        order = [len(handles)-1] + list(range(len(handles)-1))
        axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize=10, frameon=False)

    return fig


if __name__ == '__main__':
    # all, surgical, surgical_cardiac, surgical_noncardiac, icu, icu_cardiac, icu_noncardiac
    SAMPLE = 'surgical_noncardiac' #'surgical_noncardiac'
    SAMPLE_TYPE = '30d_before-day_before' #'after_admission-day_of' # 'all', '30d_before-day_before', '30d_before-day_of', 'after_admission-day_before', 'after_admission-day_of'

    sample_df = load_sample(SAMPLE, SAMPLE_TYPE)
    sample_df = sample_df[:10]
    wf_df = load_ecgs_wfdb(sample_df, batch_size=None)

    additional_labels=["hi", "hello"]
    additional_waveforms=[wf_df.iloc[1]['p_signal']]
    wf = wf_df.iloc[0]['p_signal']
    # fig = plot_ecg(wf, additional_waveforms, additional_labels)
    fig = plot_segmented_ecg(wf, additional_waveforms, additional_labels, legend=["series1", "series2"])
    fig.savefig('ecg.png', dpi=300, bbox_inches='tight')