try:
    from athome.data.utils.anomalies import main_features
except ImportError:
    from data.utils.anomalies import main_features

# Constants for the ATHOME dataset
DATASET_PATH = "athome/data/ATHOME/processed_{patient_id}"
ATHOME_CSV = DATASET_PATH + "/patient_data.csv"
ATHOME_SLEEP_CSV = DATASET_PATH + "/sleep_features.csv"
ATHOME_SLEEP_PHASES_CSV = DATASET_PATH + "/sleep_phases.csv"
ATHOME_STEPS_CSV = DATASET_PATH + "/steps_features.csv"
# ATHOME_GT = lambda t: DATASET_PATH + "/ground_truth_{t}.csv"
# LOCATION = pd.read_csv("src/data/ATHOME/location_mapping.csv")
# ID_BEACONS = LOCATION["location"].tolist().map(lambda x: str(x).lower())
ID_BEACONS = ["bagno", "soggiorno", "studio", "cucina", "camera da letto"]
ANOMALIES = [ anomaly for anomaly in main_features.keys()]
W_LABELS = []
LIMIT = 5 # Maximum number of consecutive NaNs to forward fill
MAX_DISTANCE = 15 # Maximum distance to consider for interpolation
INTERP_METHOD = "nearest_consec_1"
METHODS = ["nearest_consec", "nearest_consec_1", "nearest", "linear", "spline", "pchip", "loess", "savitzky_golay"]

PATIENTS: list[tuple[str, str]] = [ # (processInstanceId, hubId)
    ("95ABFA61-74FE-460D-9293-B22600D102E3", "0000000010857725"),
    ("888B4520-D70E-4D56-9881-B27300FA7F89", "000000000dc1ee89"),
    ("F6AD8BA2-C1F6-4AE9-B225-B27300FB84A5", "00000000fdc9f3b0"),
    ("9751957B-35DF-4EA9-923F-B27C008877DF", "0000000036496e56"),
    ("57F91DBB-9A4F-498C-87F6-B2B10092605C", "000000008b7c894b"),
    ("40A59AA6-0A34-4BBF-9689-B2BB008EE702", "0000000000e83ed0")
]