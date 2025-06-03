# Constants for the WESAD dataset
DATASET_PATH = "data/raw/dreamer"
PREPROCESSED_CSV = "data/preprocessed/DREAMER.csv"
PREPROCESSED_ECG_CSV = "data/preprocessed/DREAMER_ECG.csv"
SIGNALS = ["ECG"]
ORIGINAL_SR = {
    "ECG": 256,  # Hz
    "label": 256  # Hz
}

# We preprocess the dataset to have labels and HR, HRV features computed on the last N seconds
N = 60 # seconds

# We preprocess the dataset to have samples every OFFSET seconds
OFFSET = 10 # seconds