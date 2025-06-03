# Constants for the WESAD dataset
DATASET_PATH = "data/raw/WESAD"
PREPROCESSED_CSV = "data/preprocessed/WESAD.csv"
PREPROCESSED_ECG_CSV = "data/preprocessed/WESAD_ECG.csv"
SIGNALS = ["ACC", "BVP", "TEMP", "ECG"]
ORIGINAL_SR = {
    "ACC": 32,  # Hz
    "BVP": 64,  # Hz
    "TEMP": 4,  # Hz
    "ECG": 700,  # Hz
    "label": 700  # Hz
}

# We preprocess the dataset to have labels and HR, HRV features computed on the last N seconds
N = 60 # seconds

# We preprocess the dataset to have samples every OFFSET seconds
OFFSET = 10 # seconds