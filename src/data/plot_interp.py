import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.ATHOME.constants import METHODS
from data.utils.interpolation import custom_interp
from data.eval_interp import get_data, WINDOW_SIZE, REMOVE_SIZE

def visualize_window(original: pd.DataFrame, interpolated: pd.DataFrame, window_idx: int):
    """
    Visualize the original and interpolated signals for a given window.
    
    Args:
        original (pd.DataFrame): The original signal. Columns: ['timestamp', 'systolic', 'diastolic', 'spo2', 'hrv', 'cvrr', 'breath', 'pulse']
        interpolated (pd.DataFrame): The interpolated signal. Columns: ['timestamp', 'systolic', 'diastolic', 'spo2', 'hrv', 'cvrr', 'breath', 'pulse']
        window_idx (int): The window index.
    """
    # Create a figure with subplots for each signal
    fig, axs = plt.subplots(len(original.columns[1:]), figsize=(15, 5*len(original.columns[1:])))
    fig.suptitle(f"Window {window_idx}")

    # Plot each signal
    for i, column in enumerate(original.columns[1:]):
        # Original signal
        axs[i].plot(original["timestamp"], original[column], label="Original", color="blue")

        # Interpolated signal
        axs[i].plot(interpolated["timestamp"], interpolated[column], label="Interpolated", color="red")

    # Add labels and legend
    for ax in axs.flat:
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()

    # Adjust layout and display the plot    
    plt.tight_layout()
    plt.show()

def visualize_random_windows(df_patient, method, k=5):
    """Visualize k random windows with original and interpolated signals."""
    windows = [df_patient.iloc[i:i+WINDOW_SIZE] 
               for i in range(len(df_patient) - WINDOW_SIZE + 1)]
    
    # Randomly select k windows
    selected_windows = random.sample(windows, k)
    
    for i, window in enumerate(selected_windows):
        # Create a test scenario (e.g., remove middle part)
        mask = slice(WINDOW_SIZE//2 - REMOVE_SIZE//2, WINDOW_SIZE//2 + REMOVE_SIZE//2)
        test_df: pd.DataFrame = window.copy()
        # Replace with NaNs all but the timestamp column
        test_df.iloc[mask, 1:] = np.nan
        
        # Interpolate the missing values
        interp_df, _ = custom_interp(test_df, method)
        
        # Visualize the original and interpolated signals
        visualize_window(window, interp_df, i+1)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize interpolated windows.')
    parser.add_argument('--method', type=str, required=True, choices=METHODS, help='Interpolation method to use.')
    parser.add_argument('--k', type=int, default=5, help='Number of random windows to visualize.')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load your data
    df_patient = get_data()
    
    # Visualize k random windows
    print(f"Visualizing {args.k} random windows with {args.method} interpolation...")
    visualize_random_windows(df_patient, args.method, args.k)