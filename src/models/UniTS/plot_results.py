import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import argparse
import matplotlib.dates as mdates

from data.utils.anomalies import main_features, anomaly_types
from data.ATHOME.constants import ANOMALIES as ATHOME_ANOMALIES
from data.PREATHOME.constants import ANOMALIES as PREATHOME_ANOMALIES

def get_merged_data(pred: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """
    Identify contiguous regions where the prediction values exceed the threshold,
    marking detected anomalies.

    Parameters:
        pred (np.ndarray): Prediction array (e.g. anomaly scores).
        threshold (float): Threshold value to classify an anomaly.

    Returns:
        list[tuple[int, int]]: List of (start, end) indices of detected anomalies.
    """
    anomalies = []
    anomaly_start = -1
    th_pred = pred >= threshold  # Boolean mask: True when anomaly detected
    for i, anomaly in enumerate(th_pred):
        if anomaly:
            if anomaly_start == -1:
                anomaly_start = i
        else:
            if anomaly_start != -1:
                anomalies.append((anomaly_start, i))
                anomaly_start = -1

    if anomaly_start != -1:
        anomalies.append((anomaly_start, len(th_pred)))

    return anomalies

def plot_results(file_name: str, anomaly_id: int):
    """
    Plots the anomaly detection results with improved visualization.
    
    Each subplot shows a time series with:
      - Detected anomalies highlighted by a core region (dark red)
        and a padded region (lighter red) for clarity.
      - A threshold slider allowing interactive adjustment.
      - A multiline label on the left side. The label is constructed by splitting the
        feature name by underscores, capitalizing each word, and displaying them on separate lines.
    
    Args:
        file_name (str): Dataset name, e.g., 'ATHOME' or 'PREATHOME'.
        anomaly_id (int): Index of the anomaly type to plot.
    """
    # Define the directory containing result files
    script_dir = os.path.dirname(__file__)
    path = os.path.join(
        script_dir,
        "checkpoints/contextual_anomaly_detection_UniTS_sup__UniTS_All_ftM_dm128_el3_Exp_0/results"
    )
    
    # Select anomaly type from predefined constants
    if file_name == "PREATHOME":
        anomaly_type = PREATHOME_ANOMALIES[anomaly_id]
        rel_features = main_features[anomaly_type]
        freq = "10T"  # 10-minute intervals
    elif file_name.startswith("ATHOME_"):
        anomaly_type = ATHOME_ANOMALIES[anomaly_id]
        rel_features = main_features[anomaly_type]
        freq = "1T"   # 1-minute intervals
    else:
        raise ValueError(f"Data {file_name} not supported")
    len_features = len(rel_features)

    # For certain anomaly types, force daily frequency
    if anomaly_type.startswith("SLP_") or anomaly_type.startswith("ACT_"):
        freq = "1D"
    
    # Load prediction, data, and ground truth arrays
    pred = np.load(f"{path}/{file_name}_{anomaly_type}/pred.npy")
    data = np.load(f"{path}/{file_name}_{anomaly_type}/data.npy")
    gt = np.load(f"{path}/{file_name}_{anomaly_type}/gt.npy")
    print(f"Data shape: {data.shape}, Pred shape: {pred.shape}, GT shape: {gt.shape}")

    # Load the starting date from file and convert to Timestamp
    with open(f"{path}/{file_name}_{anomaly_type}/first_date.txt", "r") as f:
        first_date_str = f.read().strip()
    first_date = pd.Timestamp(first_date_str)

    # Create a datetime index for the x-axis using the determined frequency
    date_values = pd.date_range(first_date, periods=data.shape[0], freq=freq)
    
    # Extract the first sample of each window (for the relevant features) for plotting
    data_series_array = data[:, 0, :len_features]
    gt = gt[::data.shape[1]]  # Downsample ground truth to match data shape
    print(f"Anomalies (GT sum): {gt.sum()}")
    print(f"Anomalies (Pred sum): {pred.sum()}")

    # Aggregate predictions to match the shape of data_series_array
    pred = pred.reshape(-1, data.shape[1])
    new_pred = np.zeros(data_series_array.shape[0])
    for i in range(pred.shape[0] - pred.shape[1] + 1):
        for j in range(pred.shape[1]):
            if i + j < new_pred.shape[0]:
                new_pred[i + j] += pred[i, j]

    # Create figure and subplots
    fig, axes = plt.subplots(len_features, 1, sharex=True, figsize=(20, 5 * len_features))
    plt.subplots_adjust(left=0.15, bottom=0.25)  # Reserve space on left for labels and bottom for the slider
    if len_features == 1:
        axes = [axes]

    # Configure date locator and formatter for the x-axis
    if freq == "1D":
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        pad_k = 0  # Number of indices to pad around each anomaly region
    else:
        locator = mdates.AutoDateLocator(interval_multiples=True)
        formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
        pad_k = 5  # Number of indices to pad around each anomaly region

    # Identify ground truth anomaly regions using a threshold of 0.5
    gt_sequences = get_merged_data(gt, 0.5)

    # Plot each feature's time series along with ground truth anomaly regions
    for i in range(len_features):
        # Create a Pandas Series with a datetime index for plotting
        series = pd.Series(data_series_array[:, i], index=date_values)
        axes[i].plot(series.index, series.values, linewidth=2)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
        # Remove any default ylabel since we are adding custom multiline text on the left
        axes[i].set_ylabel('')
        
        # Remove legend as per requirement
        # axes[i].legend(loc='upper right', fontsize=12)

        # Annotate ground truth anomaly regions using axvspan
        for l, r in gt_sequences:
            start_date = date_values[int(np.floor(l))]
            end_date = date_values[min(int(np.ceil(r)), len(date_values) - 1)]
            axes[i].axvspan(start_date, end_date, color='green', alpha=0.3)
        
        # Apply the date locator and formatter to the x-axis
        axes[i].xaxis.set_major_locator(locator)
        axes[i].xaxis.set_major_formatter(formatter)
        
        #* Add Custom Multiline Label on the Left Side
        # Split feature name by underscore, capitalize each part, and join with newline
        label_text = "\n".join(word.capitalize() for word in rel_features[i].split("_"))
        # Place text in axis coordinates (x=-0.1 means just outside the left margin)
        axes[i].text(-0.035, 0.5, label_text, transform=axes[i].transAxes,
                     fontsize=12, horizontalalignment='right', verticalalignment='center')

    #* Anomaly Regions with Padded Visualization
    # Define the initial threshold and a padding constant (K) in index units
    THRESHOLD_INIT = 1

    # Lists to store the plotted regions for the core anomaly and the padded region
    anomaly_regions_core = [[] for _ in range(len_features)]
    anomaly_regions_pad = [[] for _ in range(len_features)]
    
    # Detect anomaly regions based on the initial threshold
    sequences = get_merged_data(new_pred, THRESHOLD_INIT)
    for l, r in sequences:
        for i in range(len_features):
            # Calculate core anomaly region indices
            core_start_idx = int(np.floor(l))
            core_end_idx = min(int(np.ceil(r)), len(date_values) - 1)
            # Calculate padded region indices, ensuring indices stay within bounds
            padded_start_idx = max(0, core_start_idx - pad_k)
            padded_end_idx = min(len(date_values) - 1, core_end_idx + pad_k)
            
            # Convert indices to datetime values
            unit = date_values[1] - date_values[0]
            core_start_date = date_values[core_start_idx] - unit / 2
            core_end_date = date_values[core_end_idx] - unit / 2
            pad_start_date = date_values[padded_start_idx] - unit / 2
            pad_end_date = date_values[padded_end_idx] - unit / 2
            
            # Plot the padded anomaly region (lighter red) first
            pad_region = axes[i].axvspan(pad_start_date, pad_end_date, color='lightcoral', alpha=0.2)
            anomaly_regions_pad[i].append(pad_region)
            
            # Plot the core anomaly region (darker red) on top
            core_region = axes[i].axvspan(core_start_date, core_end_date, color='red', alpha=0.8)
            anomaly_regions_core[i].append(core_region)

    #* Threshold Slider for Interactive Updates
    # The slider's range is from 1 to (window size / 2)
    window_size = data.shape[1]
    threshold_slider = Slider(
        ax=plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow'),
        label='THRESHOLD',
        valmin=1,
        valmax=window_size // 2,
        valinit=THRESHOLD_INIT,
        initcolor='blue',
        valstep=1,
    )

    def update(val):
        """
        Update the anomaly regions based on the new threshold value.
        Removes previous anomaly regions and redraws both the padded and core regions.
        """
        new_threshold = threshold_slider.val
        
        # Remove previously drawn anomaly regions (both core and padded)
        for i in range(len_features):
            for region in anomaly_regions_core[i]:
                region.remove()
            anomaly_regions_core[i].clear()
            for region in anomaly_regions_pad[i]:
                region.remove()
            anomaly_regions_pad[i].clear()
        
        # Recalculate anomaly regions with the updated threshold
        sequences = get_merged_data(new_pred, new_threshold)
        for l, r in sequences:
            for i in range(len_features):
                core_start_idx = int(np.floor(l))
                core_end_idx = min(int(np.ceil(r)), len(date_values) - 1)
                padded_start_idx = max(0, core_start_idx - pad_k)
                padded_end_idx = min(len(date_values) - 1, core_end_idx + pad_k)
                
                core_start_date = date_values[core_start_idx]
                core_end_date = date_values[core_end_idx]
                pad_start_date = date_values[padded_start_idx]
                pad_end_date = date_values[padded_end_idx]
                
                pad_region = axes[i].axvspan(pad_start_date, pad_end_date, color='lightcoral', alpha=0.3)
                anomaly_regions_pad[i].append(pad_region)
                core_region = axes[i].axvspan(core_start_date, core_end_date, color='red', alpha=0.8)
                anomaly_regions_core[i].append(core_region)
        fig.canvas.draw_idle()

    # Connect the update function to the slider
    threshold_slider.on_changed(update)

    # Add overall figure title and x-axis label for clarity
    fig.suptitle(f"{file_name} - {anomaly_types[anomaly_type]}", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    # Rotate x-axis labels and move to the right for better visibility
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the results of the anomaly detection model.")
    parser.add_argument("--idx", type=int, help="The index of the anomaly type to plot.", default=0)
    parser.add_argument("--data", type=str, help="The data type to plot.", default="ATHOME")
    args = parser.parse_args()
    plot_results(args.data, args.idx)
