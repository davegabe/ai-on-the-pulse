# Constants for the WESAD dataset
DATASET_PATH = "data/raw/HCI/Sessions"
PREPROCESSED_CSV = "data/preprocessed/HCI.csv"
PREPROCESSED_ECG_CSV = "data/preprocessed/HCI_ECG.csv"
SIGNALS = ["TEMP", "ECG"]
ORIGINAL_SR = {
    "TEMP": 4,  # Hz
    "ECG": 250,  # Hz
    "label": 250  # Hz
}

# We preprocess the dataset to have labels and HR, HRV features computed on the last N seconds
N = 60 # seconds

# We preprocess the dataset to have samples every OFFSET seconds
OFFSET = 10 # seconds