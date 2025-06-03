import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import scipy.io

from src.data.DREAMER.constants import DATASET_PATH, PREPROCESSED_CSV, PREPROCESSED_ECG_CSV, ORIGINAL_SR, N, OFFSET
from src.utils.resample import get_samples

PATIENTS = 23
VIDEOS = 18

FEATURES = {
    'Age': 0,
    'Gender': 1,
    'EEG': 2,
    'ECG': 3,
    'Valence': 4,
    'Arousal': 5,
    'Dominance': 6
}


def retrieve_data(data, patient_idx: int, video_idx: int, feature: str) -> float:
    """
    Retrieve specific feature data for a given patient from the dataset.

    Parameters:
        patient_idx (int): Index of the patient (0 to 23).
        video_idx (int): Index of the video (0 to 17).
        feature (str): Name of the feature to retrieve. It should be one of the following:
                            'Age', 'Gender', 'EEG', 'ECG', 'Valence', 'Arousal', 'Dominance'.

    Returns:
        float: The value of the specified feature for the specified patient.

    Raises:
        KeyError: If Feature_name is not one of the supported features.
    """
    feature_idx = FEATURES[feature]
    return data['DREAMER']['Data'][0, 0][0][patient_idx][0][0][feature_idx][video_idx]


def retrieve_signals(data, patient_idx: int, signal: str, base: bool) -> float:
    """
    Retrieve specific signal data for a given patient from the dataset.

    Parameters:
        Sample_index (int): Index of the patient (0 to 23).
        signal (str): Type of signal ('EEG' or 'ECG').
        base (bool): Whether to retrieve the baseline or stimuli signal.

    Returns:
        float: The value of the specified signal for the specified patient.

    Raises:
        ValueError: If Feature_type is not 'EEG' or 'ECG'.
    """
    if signal not in ('EEG', 'ECG'):
        raise ValueError(
            f"Invalid Feature_type: {signal}. Feature_type must be 'EEG' or 'ECG'.")

    feature_idx = FEATURES[signal]
    base_idx = 0 if base else 1
    return data['DREAMER']['Data'][0, 0][0][patient_idx][0][0][feature_idx][0][0][base_idx]


def load_patient_data(mat: dict, patient_id: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load the patient data from the pickle file.

    Args:
        mat (dict): The loaded .mat file.
        patient_id (int): The ID of the patient (0 to 22).

    Returns:
        list[Data]: The signals (USED_SIGNALS + ECG).
        np.ndarray: The labels
    """
    # Get the patient data
    stimuli = retrieve_signals(mat, patient_id, 'ECG', base=False)
    ecg = []
    labels = []
    for j in range(VIDEOS):
        valence = retrieve_data(mat, patient_id, j, 'Valence')[0]
        anomaly = 1 if valence <= 2 else 0

        # Use left and right stimuli for the ECG signal
        left_channel = stimuli[j][0][:, 0]
        ecg.append(left_channel)
        labels.append([anomaly] * len(left_channel))
        right_channel = stimuli[j][0][:, 1]
        ecg.append(right_channel)
        labels.append([anomaly] * len(right_channel))

    # Concatenate the ECG signals and labels
    ecg = np.concatenate(ecg)
    labels = np.concatenate(labels)

    # Useful signals (ECG)
    signals: dict[str, np.ndarray] = {
        "ECG": ecg
    }

    return signals, labels


def main():
    """
    Preprocess DREAMER dataset for anomaly detection task and create two CSV file with the data.
    One CSV file contains the ECG signal and the other contains the HR and HRV features computed from the ECG signal.
    The CSV files also contain the Patient ID and the label of the frame.
    """
    # Check if the CSV file already exists
    if os.path.exists(PREPROCESSED_CSV) and os.path.exists(PREPROCESSED_ECG_CSV):
        print("DREAMER preprocessed data already exists. Skipping preprocessing.")
        return

    file = open(os.path.join(DATASET_PATH, "DREAMER.mat"), 'rb')
    mat = scipy.io.loadmat(file)

    # Process all patients
    extr_data: list[pd.DataFrame] = []
    ecg_data: list[pd.DataFrame] = []
    patients = range(PATIENTS)
    for i, patient in enumerate(tqdm(patients, desc="DREAMER preprocessing", position=0)):
        # Load the patient data
        signals, labels = load_patient_data(mat, i)

        # Extract the samples
        extr_df, ecg_df = get_samples(signals, labels, ORIGINAL_SR, N, OFFSET)
        extr_df["Patient"] = patient
        ecg_df["Patient"] = patient

        # Append the data to the list
        extr_data.append(extr_df)
        ecg_data.append(ecg_df)

    # Concatenate all data
    extr_df = pd.concat(extr_data)
    ecg_df = pd.concat(ecg_data)

    # Save the preprocessed data to a CSV file
    extr_df.to_csv(PREPROCESSED_CSV, index=False)
    ecg_df.to_csv(PREPROCESSED_ECG_CSV, index=False)
    print(f"Preprocessed DREAMER data saved to {PREPROCESSED_CSV} and {PREPROCESSED_ECG_CSV}.")


if __name__ == "__main__":
    main()
