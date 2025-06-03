import numpy as np
from biosppy.signals import ecg
from pyhrv.tools import nn_intervals
from pyhrv.time_domain import rmssd

def extract_features(ecg_signal: np.ndarray, sample_rate: int):
    """
    Extract the features from the ECG signal.

    Args:
        ecg_signal (np.ndarray): The ECG signal.
        feature_names (list[str]): The feature names to extract.

    Returns:
        ExtractedFeatures: The HR and HRV features.
    """
    # Compute the HR
    _, _, rpeaks, _, _, _, hr = ecg.ecg(
        ecg_signal.flatten(),
        show=False,
        sampling_rate=sample_rate
    )

    # Compute NNI
    nni = nn_intervals(rpeaks)

    # Compute the HRV time domain features
    hrv = rmssd(nni)[0]
    
    return {"HR_ECG": hr.mean(), "HRV_ECG": hrv.mean()}
