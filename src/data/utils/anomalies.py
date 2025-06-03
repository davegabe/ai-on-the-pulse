import numpy as np
import pandas as pd
# from data.ATHOME.constants import ID_BEACONS

# Define anomaly types and their corresponding names
anomaly_types = {
    # Vital signs
    "VS_HYPERHYPOTENSION": "Hypertension/Hypotension",
    "VS_STRESS": "Stress",
    "VS_ABN_HRV": "Abnormal HRV",
    # # Behavior
    # "BEH_ROOM_DURATION_ANOMALY": "Room Duration Anomaly",
    # "BEH_ABN_BATHROOM_USAGE": "Abnormal Bathroom Usage",
    # "BEH_ABN_KITCHEN_USAGE": "Abnormal Kitchen Usage",
    # "BEH_ABN_OUT_OF_HOME": "Out of Home",
    # # Activity
    # "ACT_ABN_STEPS": "Abnormal Steps",
    # # Sleep
    # "SLP_INSOMNIA_HYPERSOMNIA": "Insomnia/Hypersomnia",
    # "SLP_SLEEP_QUALITY": "Sleep Quality",
    # # Environment
    # "ENV_LUMINOSITY": "Luminosity"
}

# Define main features for each anomaly type
main_features = {
    # Vital signs
    "VS_HYPERHYPOTENSION": ['systolic', 'diastolic', 'pulse', 'hrv'],
    "VS_STRESS": ['pulse', 'hrv'],
    "VS_ABN_HRV": ['hrv'],
    # # Behavior
    # "BEH_ROOM_DURATION_ANOMALY": [f"{loc}_time" for loc in ID_BEACONS],
    # "BEH_ABN_BATHROOM_USAGE": ['bathroom_location', 'bathroom_duration'],
    # "BEH_ABN_KITCHEN_USAGE": ['kitchen_location', 'kitchen_duration'],
    # "BEH_ABN_OUT_OF_HOME": ['outside_location', 'outside_duration'],
    # # Activity
    # "ACT_ABN_STEPS": ['total_steps', 'steps_morning', 'steps_afternoon', 'steps_evening', 'steps_night'],
    # # Sleep
    # "SLP_INSOMNIA_HYPERSOMNIA": ['sleep_duration', 'deep_sleep_ratio', 'sleep_onset', 'wake_up'],
    # "SLP_SLEEP_QUALITY": ['fragmentation_index', 'avg_session_duration', 'max_session_duration', 'avg_deep_sleep_duration'],
    # # Environment
    # "ENV_LUMINOSITY": ['luminosity']
}

# Define context features for each anomaly type. If list is empty, all features are used. If not, only the features in the list are used.
context_features = {
    # Vital signs
    "VS_HYPERHYPOTENSION": [],
    "VS_STRESS": [],
    "VS_ABN_HRV": [],
    # # Behavior
    # "BEH_ROOM_DURATION_ANOMALY": [],
    # "BEH_ABN_BATHROOM_USAGE": [],
    # "BEH_ABN_KITCHEN_USAGE": [],
    # "BEH_ABN_OUT_OF_HOME": [],
    # # Activity
    # "ACT_ABN_STEPS": ["weekday", "day"],
    # # Sleep
    # "SLP_INSOMNIA_HYPERSOMNIA": ["weekday", "day"],
    # "SLP_SLEEP_QUALITY": ["weekday", "day"],
    # # Environment
    # "ENV_LUMINOSITY": []
}

# Define lm for POT method
anomaly_lm = {
    "PREATHOME": {
        # Vital signs
        "VS_HYPERHYPOTENSION": (0.99, 1),
        "VS_STRESS": (0.99, 1),
        "VS_ABN_HRV": (0.995, 1),
        # # Behavior
        # "BEH_ROOM_DURATION_ANOMALY": [],
        # "BEH_ABN_BATHROOM_USAGE": [],
        # "BEH_ABN_KITCHEN_USAGE": [],
        # "BEH_ABN_OUT_OF_HOME": [],
        # Activity
        "ACT_ABN_STEPS": (0.995, 1),
        # Sleep
        "SLP_INSOMNIA_HYPERSOMNIA": (0.99, 1),
        "SLP_SLEEP_QUALITY": (0.998, 1),
        # # Environment
        # "ENV_LUMINOSITY": []
    },
    "ATHOME": {
        # Vital signs
        "VS_HYPERHYPOTENSION": (0.9995, 1),
        "VS_STRESS": (0.9995, 1),
        "VS_ABN_HRV": (0.9995, 1),
        # # Behavior
        # "BEH_ROOM_DURATION_ANOMALY": [],
        # "BEH_ABN_BATHROOM_USAGE": [],
        # "BEH_ABN_KITCHEN_USAGE": [],
        # "BEH_ABN_OUT_OF_HOME": [],
        # Activity
        "ACT_ABN_STEPS": (0.95, 1),
        # Sleep
        "SLP_INSOMNIA_HYPERSOMNIA": (0.9995, 1),
        "SLP_SLEEP_QUALITY": (0.9995, 1),
        # # Environment
        # "ENV_LUMINOSITY": []
    },
}


def get_seq_to_impute(seq_type: str, win_size: int) -> int:
    """
    Get the sequence to impute for the anomaly type.

    Args:
        seq_type (str): The type of sequence to impute.
        win_size (int): The size of the sliding window.

    Returns:
        int: The sequence to impute.
    """
    if seq_type == "HALF":
        seq_to_impute = win_size // 2
    elif seq_type == "LAST":
        seq_to_impute = 1
    else:
        raise ValueError(f"Sequence type '{seq_type}' not recognized.")
    
    return seq_to_impute

def add_time_features(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Add time features to the signals DataFrame.

    Args:
        signals (pd.DataFrame): The signals DataFrame.

    Returns:
        pd.DataFrame: The signals DataFrame with the time features.
    """
    #* Create datetime features
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    # Sort the data by timestamp
    signals = signals.sort_values(by="timestamp")
    # Convert "timestamp" to time of the day, day of the week and day of the year
    seconds_in_day = 24 * 60 * 60
    signals["time"] = (signals["timestamp"].dt.second + signals["timestamp"].dt.minute * 60 + signals["timestamp"].dt.hour * 3600) / seconds_in_day
    signals["weekday"] = signals["timestamp"].dt.weekday / 6
    signals["day"] = signals["timestamp"].dt.dayofyear / 365

    return signals

def get_features(
        measurements: pd.DataFrame,
        sleep_data: pd.DataFrame,
        steps_data: pd.DataFrame,
        anomaly_type: str
    ) -> tuple[pd.DataFrame, list[bool], str]:
    """
    Remove the irrelevant features for the anomaly type.

    Args:
        measurements (pd.DataFrame): The measurements data if the anomaly is related to vital signs.
        sleep_data (pd.DataFrame): The sleep data if the anomaly is related to sleep.
        steps_data (pd.DataFrame): The steps data if the anomaly is related to activity.
        anomaly_type (str): The type of anomaly to determine the main features.

    Returns:
        pd.DataFrame: The signals with the main features for the anomaly type.
        list[bool]: List with True if the feature is a main feature, False otherwise.
        str: The sequence type to impute.
    """
    #* Get the main features for the anomaly type
    features = main_features[anomaly_type]
    context = context_features[anomaly_type]

    #* Select appropriate data
    if anomaly_type.startswith("SLP_"):
        signals = sleep_data.copy()
    elif anomaly_type.startswith("ACT_"):
        signals = steps_data.copy()
    else:
        signals = measurements.copy()
    
    # Strip all the columns
    signals.columns = signals.columns.str.strip()
    
    #* Add time features
    signals = add_time_features(signals)

    #* If there are no context features, use all other features
    if len(context) == 0:
        context = [col for col in signals.columns if col not in features]
    else:
        context += ["timestamp"]
    context = list(set(context))

    #* Load the features for SLEEP anomalies
    if anomaly_type.startswith("SLP_"):
        signals = signals[features + context].copy() # Only sleep features, no context
        seq_type = "LAST"

    #* Load the features for ACTIVITY anomalies
    elif anomaly_type.startswith("ACT_"):
        signals = signals[features + context].copy()
        seq_type = "LAST"
    
    #* Load the features for VITAL SIGNS anomalies
    elif anomaly_type.startswith("VS_"):
        signals = signals[features + context].copy()
        seq_type = "HALF"

    #* Load the features for ENVIRONMENT anomalies
    elif anomaly_type.startswith("ENV_"):
        seq_type = "HALF"
        raise NotImplementedError("Environment anomaly not implemented yet.")
    
    #* Load the features for BEHAVIOR anomalies
    elif anomaly_type.startswith("BEH_"):
        seq_type = "HALF"
        raise NotImplementedError("Behavior anomaly not implemented yet.")
    
    #* If the anomaly type is not recognized
    else:
        raise ValueError(f"Anomaly type '{anomaly_type}' not recognized.")

    #* List with True if the feature is a main feature, False otherwise
    columns = signals.columns.tolist()
    columns.remove("timestamp") # Remove timestamp from the list
    columns = np.array(columns)
    selected_main = [col in features for col in columns]
    selected_main = np.array(selected_main)

    #* Set labels, must be after selecting the main features
    signals["Label"] = 0

    return signals, selected_main, seq_type

def get_anomaly_types_info() -> dict:
    """
    Get the anomaly type information (name, main features, context features).

    Returns:
        dict: The anomaly type information.
    """
    anomaly_dict = {}
    for key, value in anomaly_types.items():
        anomaly_dict[key] = {
            "name": value,
            "main_features": main_features[key],
            "context_features": context_features[key],
        }
    return anomaly_dict