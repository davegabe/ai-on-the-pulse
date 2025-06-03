# This code is inspired by https://github.com/Jegama/empaticaHRV/tree/master

from biosppy import bvp
import numpy as np
import peakutils
import math


def bvpPeaks(signal: list) -> list:
    """
    Find the peaks in the BVP signal.

    Args:
        signal (list): The BVP signal.

    Returns:
        list: The indexes of the peaks.
    """
    cb = np.array(signal)

    # Find the peaks
    x = peakutils.indexes(cb, thres=0.02/max(cb), min_dist=0.1)
    y = []

    # Remove peaks that are too close together
    i = 0
    while i < len(x)-1:
        if x[i+1] - x[i] < 15:
            y.append(x[i])
            x = np.delete(x, i+1)
        else:
            y.append(x[i])
        i += 1
    return y


def getRRI(signal: list, sample_rate: float):
    """
    Get the RRI from the BVP signal.

    Args:
        signal (list): The BVP signal.
        sample_rate (float): The sample rate.

    Returns:
        list[int]: IBI.
    """
    # Find the peaks
    peakIDX = bvpPeaks(signal)
    spr = 1 / sample_rate  # seconds between readings

    # Calculate the IBI
    ibi = [0, 0]
    for i in range(1, len(peakIDX)):
        ibi.append((peakIDX[i] - peakIDX[i-1]) * spr)
    return ibi


def getHRV(ibi: list[int], avg_heart_rate: float, method='SDNN'):
    """
    Get the HRV from the RRI.

    Args:
        ibi (list[int]): The IBI.
        avg_heart_rate (float): The average heart rate.
        method (str): The method to use. Default is 'SDNN'. [SDNN, RMSSD]

    Returns:
        pd.DataFrame: The HRV data.
    """
    rri = np.array(ibi) * 1000
    RR_list = rri.tolist()
    RR_diff = []
    RR_sqdiff = []
    cnt = 2

    # Calculate the RR differences and the square of the differences
    while (cnt < (len(RR_list)-1)):
        RR_diff.append(abs(RR_list[cnt+1] - RR_list[cnt]))
        RR_sqdiff.append(math.pow(RR_list[cnt+1] - RR_list[cnt], 2))
        cnt += 1

    # Set the window length
    hrv_window_length = 10
    window_length_samples = int(hrv_window_length*(avg_heart_rate/60))
    SDNN = []
    RMSSD = []
    index = 1

    # Calculate the HRV
    for val in RR_sqdiff:
        if index < int(window_length_samples):
            SDNNchunk = RR_diff[:index:]
            RMSSDchunk = RR_sqdiff[:index:]
        else:
            SDNNchunk = RR_diff[(index-window_length_samples):index:]
            RMSSDchunk = RR_sqdiff[(index-window_length_samples):index:]
        SDNN.append(np.std(SDNNchunk))
        RMSSD.append(math.sqrt(1. / len(RR_list) * np.std(RMSSDchunk)))
        index += 1
    SDNN = np.array(SDNN, dtype=np.float32)
    RMSSD = np.array(RMSSD, dtype=np.float32)

    if method == 'SDNN':
        return SDNN
    elif method == 'RMSSD':
        return RMSSD
    else:
        raise ValueError('Invalid method for HRV calculation')


def extract_features(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    """
    Extract the features from the BVP signal.

    Args:
        signal (np.ndarray): The BVP signal.

    Returns:
        dict[str, float]: The extracted features.
    """
    # Get the filtered signal and the heart rate
    _, filtered_signal, _, _, hr = bvp.bvp(signal.flatten(), sample_rate, show=False)
    hr = np.mean(hr)
    
    # Get the IBI
    ibi = getRRI(filtered_signal, sample_rate)
    
    # Get the HRV
    hrv = getHRV(ibi, hr, method='RMSSD')
    
    return {"HR_BVP": hr, "HRV_BVP": hrv.mean()}
