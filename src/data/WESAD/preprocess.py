import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd

from src.data.WESAD.constants import DATASET_PATH, PREPROCESSED_CSV, PREPROCESSED_ECG_CSV, ORIGINAL_SR, N, OFFSET
from src.utils.resample import get_samples


def load_patient_data(pickle_path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load the patient data from the pickle file.

    Args:
        pickle_path (str): The path to the pickle file.

    Returns:
        list[Data]: The signals (USED_SIGNALS + ECG).
        np.ndarray: The labels
    """
    # Load the data from the pickle file
    file = open(pickle_path, 'rb')
    data = pickle.load(file, encoding="latin1")

    # Labels (0 = not defined/transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation)
    labels = data["label"]
    # Set the stress label to 1 and the others to 0
    labels = [1 if l == 2 else 0 for l in labels]

    # Chest signals (ACC, ECG, EMG, EDA, TEMP, RESP)
    chest_signals = data["signal"]["chest"]
    # Wrist signals (ACC, BVP, EDA, TEMP)
    wrist_signals = data["signal"]["wrist"]

    # Useful signals (all wrist signals which are in the used signals + ECG)
    signals: dict[str, np.ndarray] = {
        "ACC": wrist_signals["ACC"],
        "TEMP": wrist_signals["TEMP"],
        "BVP": wrist_signals["BVP"],
        "ECG": chest_signals["ECG"]
    }

    return signals, labels


def main():
    """
    Preprocess WESAD dataset for anomaly detection task and create two CSV file with the data.
    One CSV file contains the ECG signal and the other contains the HR and HRV features computed from the ECG and BVP signals.
    The CSV files also contain the Patient ID and the label of the frame.
    """
    # Check if the CSV file already exists
    if os.path.exists(PREPROCESSED_CSV) and os.path.exists(PREPROCESSED_ECG_CSV):
        print("WESAD preprocessed data already exists. Skipping preprocessing.")
        return

    # Process all patients
    extr_data: list[pd.DataFrame] = []
    ecg_data: list[pd.DataFrame] = []
    patients = os.listdir(DATASET_PATH)
    for patient in tqdm(patients, desc="WESAD preprocessing", position=0):
        # Patient path
        patient_path = os.path.join(DATASET_PATH, patient)
        pickle_path = os.path.join(patient_path,  f"{patient}.pkl")

        # If the pickle file does not exist, continue
        if not os.path.exists(pickle_path):
            print(f"Patient {patient} does not have a pickle file. Skipping.")
            continue

        # Load the patient data
        signals, labels = load_patient_data(pickle_path)

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
    print(
        f"Preprocessed WESAD data saved to {PREPROCESSED_CSV} and {PREPROCESSED_ECG_CSV}.")


if __name__ == "__main__":
    main()
