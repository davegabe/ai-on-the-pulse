import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import pyedflib
import numpy as np
import xml.etree.ElementTree as ET

from src.data.HCI.constants import DATASET_PATH, PREPROCESSED_CSV, PREPROCESSED_ECG_CSV, ORIGINAL_SR, N, OFFSET
from src.utils.resample import get_samples


def load_patient_data(session_path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load the patient data from the pickle file.

    Args:
        session_path (str): The path to the session folder.

    Returns:
        list[Data]: The signals (USED_SIGNALS + ECG).
        np.ndarray: The labels
    """
    # Get the patient data
    ecg = []
    temp = []
    labels = []

    # Get the first bdf file
    files = os.listdir(session_path)
    if len(files) != 2:
        print(f"Error: Expected 2 files in {session_path}, found {len(files)} files.")
        return None, None
    
    bdf_file = [f for f in files if f.endswith(".bdf")][0]

    # Get the session labels from xml file
    xml_file = [f for f in files if f.endswith(".xml")][0]
    # Parse the XML content
    xml = open(os.path.join(session_path, xml_file), "r").read()
    root = ET.fromstring(xml)
    # Extract the felt* attributes
    felt_attributes = {attr: root.attrib[attr] for attr in root.attrib if attr.startswith('felt')}
    # Extract the labels
    valence = int(felt_attributes["feltVlnc"])
    anomaly = 1 if valence <= 2 else 0

    # Load the ECG and Temp signals
    f = pyedflib.EdfReader(os.path.join(session_path, bdf_file))
    signal_labels = f.getSignalLabels()
    for i, signal in enumerate(signal_labels):
        if signal in ["EXG1", "EXG2", "EXG3"]: # Use all 3 ECG channels
            ecg.append(f.readSignal(i))
            labels.append([anomaly] * len(f.readSignal(i)))
        if signal == "Temp":
            temp.append(f.readSignal(i).repeat(3)) # Repeat the signal 3 times to match the ECG signal

    # Concatenate the ECG signals and labels
    ecg = np.concatenate(ecg)
    temp = np.concatenate(temp)
    labels = np.concatenate(labels)

    # Useful signals (ECG)
    signals: dict[str, np.ndarray] = {
        "ECG": ecg,
        "TEMP": temp
    }

    return signals, labels


def main():
    """
    Preprocess MAHNOB-HCI dataset for anomaly detection task and create two CSV file with the data.
    One CSV file contains the ECG signal and the other contains the HR and HRV features computed from the ECG and BVP signals.
    The CSV files also contain the Patient ID and the label of the frame.
    """
    # Check if the CSV file already exists
    if os.path.exists(PREPROCESSED_CSV) and os.path.exists(PREPROCESSED_ECG_CSV):
        print("MAHNOB-HCI preprocessed data already exists. Skipping preprocessing.")
        return

    # Process all patients
    extr_data: list[pd.DataFrame] = []
    ecg_data: list[pd.DataFrame] = []
    sessions = os.listdir(DATASET_PATH)
    for session in tqdm(sessions, desc="MAHNOB-HCI preprocessing", position=0):
        # Patient path
        patient_path = os.path.join(DATASET_PATH, session)

        # Load the patient data
        signals, labels = load_patient_data(patient_path)
        if signals is None or labels is None:
            continue

        # Extract the samples
        try:
            extr_df, ecg_df = get_samples(signals, labels, ORIGINAL_SR, N, OFFSET)
            extr_df["Patient"] = session
            ecg_df["Patient"] = session

            # # Append the data to the list
            extr_data.append(extr_df)
            ecg_data.append(ecg_df)
        except ValueError as e:
            print(f"Error processing patient {session}: {e}. Skipping patient.")

    # Concatenate all data
    extr_df = pd.concat(extr_data)
    ecg_df = pd.concat(ecg_data)

    # Save the preprocessed data to a CSV file
    extr_df.to_csv(PREPROCESSED_CSV, index=False)
    ecg_df.to_csv(PREPROCESSED_ECG_CSV, index=False)
    print(
        f"Preprocessed MAHNOB-HCI data saved to {PREPROCESSED_CSV} and {PREPROCESSED_ECG_CSV}.")


if __name__ == "__main__":
    main()
