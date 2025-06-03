import numpy as np
import pandas as pd

from src.utils.bvp import extract_features as extract_features_BVP
from src.utils.ecg import extract_features as extract_features_ECG


def get_samples(
    signals: dict[str, np.ndarray],
    labels: list,
    sample_rates: dict[str, int],
    n: int,
    offset: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the samples with desired signals and labels from the pickle file.

    Args:
        signals (dict[str, np.ndarray]): The signals.
        labels (list): The labels.
        sample_rates (dict[str, int]): The sample rates for each signal.
        n (int): The number of seconds for the window.
        offset (int): The number of seconds to skip.

    Returns:
        pd.DataFrame: All the smartwatch signals and label.
        pd.DataFrame: The ECG signal and label.
    """
    # Pandas DataFrame with the ECG signal and the label
    ecg_df = pd.DataFrame(signals["ECG"], columns=["ECG"])
    ecg_df["Label"] = [int(l) for l in labels]

    # Compute the number of labels in seconds
    labels_seconds = int(len(labels) / sample_rates["label"])

    # Initialize the lists
    all_signals = []
    all_labels = []

    # Get sample every OFFSET seconds
    for end in range(n, labels_seconds, offset):
        # Start and end index
        start = end - n

        # Window signals
        w_signals = {
            "timestamp": start
        }

        # Compute the label for the window
        start_i = int(start * sample_rates["label"])
        end_i = int(end * sample_rates["label"])
        window_labels = labels[start_i:end_i]

        # Set as label the label with the most occurrences
        window_labels = [int(l) for l in window_labels]
        bin_count = np.bincount(window_labels)
        label = int(bin_count.argmax())

        # Crop signals with window
        for s in signals:
            # Start and end index
            start_i = int(start * sample_rates[s])
            end_i = int(end * sample_rates[s])

            # Extract the features from the BVP signal
            if s == "BVP":
                w_extra_signals = extract_features_BVP(
                    signals[s][start_i:end_i], sample_rates[s]
                )
                # Add the extracted features to the signals
                w_signals = w_signals | w_extra_signals
            elif s == "ACC":
                w_signals[f"{s}_x"] = signals[s][start_i, 0]
                w_signals[f"{s}_y"] = signals[s][start_i, 1]
                w_signals[f"{s}_z"] = signals[s][start_i, 2]
            elif s == "ECG":
                w_extra_signals = extract_features_ECG(
                    signals[s][start_i:end_i], sample_rates[s]
                )
                # Add the extracted features to the signals
                w_signals = w_signals | w_extra_signals
            else:
                if signals[s].ndim == 2:
                    w_signals[s] = signals[s][start_i, 0]
                else:
                    w_signals[s] = signals[s][start_i]
                    

        # Append the window signals and label
        all_signals.append(w_signals)
        all_labels.append(label)

    # Pandas DataFrame with the signals and the label
    extr_df = pd.DataFrame(all_signals)
    extr_df["Label"] = all_labels

    return extr_df, ecg_df
