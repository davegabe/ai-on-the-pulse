from data.utils.anomalies import main_features

# Constants for the PREATHOME dataset
DATASET_PATH = "athome/data/PREATHOME/processed{patient_id}"
PREATHOME_CSV = DATASET_PATH + "/measurements.csv"
PREATHOME_SLEEP_CSV = DATASET_PATH + "/sleep_features.csv"
PREATHOME_STEPS_CSV = DATASET_PATH + "/steps_features.csv"
ANOMALIES = [ anomaly for anomaly in main_features.keys()]
W_LABELS = ["STRESS"]