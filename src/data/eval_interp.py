from datetime import datetime
import multiprocessing as mp
import pandas as pd
import numpy as np
from scipy import stats, fft
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from fastdtw import fastdtw

from data.ATHOME.constants import METHODS
from data.ATHOME.retrieve import PATIENTS, retrieve_patient_data
from data.utils.interpolation import custom_interp

# Define date range and patient information
START_DATE = datetime(2024, 11, 1)
END_DATE = datetime(2025, 3, 5)
processInstanceId, hubId = PATIENTS[0]
WINDOW_SIZE = 16
REMOVE_SIZE = 5
PROCESSES = mp.cpu_count()
FS = 1.0  # Sampling frequency

def compute_dtw(original, reconstructed):
    distance, _ = fastdtw(original.values, reconstructed.values)
    return distance

def sample_entropy(series, m=2, r_factor=0.2):
    """Compute Sample Entropy with dynamic r based on series std."""
    series = series.dropna().values
    n = len(series)
    if n <= m + 1:
        return 0.0
    
    r = r_factor * np.std(series)  # Dynamic r based on series std
    
    def _maxdist(x, y):
        return np.max(np.abs(x - y))
    
    def _phi(m):
        x = np.array([series[i:i+m] for i in range(n - m + 1)])
        C = 0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if _maxdist(x[i], x[j]) <= r:
                    C += 1
        return C / (n - m + 1)

    phi_m_plus_1 = _phi(m + 1)
    phi_m = _phi(m)
    
    # Handle division by zero
    if np.isclose(phi_m, 0.0):
        return 0.0  # Return 0 or another default value when no matches are found
    
    if np.isclose(phi_m_plus_1, 0.0):
        return np.inf
    return -np.log(phi_m_plus_1 / phi_m)

def spectral_entropy(series, fs=1.0):
    """Compute Spectral Entropy of the series."""
    data = series - np.mean(series)
    fft_vals = np.abs(fft.rfft(data))**2
    psd = fft_vals / (np.sum(fft_vals) + 1e-10)  # Normalize to PDF, avoid division by zero
    psd = psd[psd > 0]  # Avoid log(0)
    return -np.sum(psd * np.log2(psd))  # Shannon entropy

def compute_variability_metrics(df: pd.DataFrame):
    """Compute variability metrics for each column."""
    metrics = {
        'std_diff': [],
        'spectral_entropy': [],
        'sample_entropy': []
    }
    
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 3:
            continue
            
        # Convert to numpy array for numerical operations
        data = series.to_numpy()
        
        # Standard deviation of first differences
        metrics['std_diff'].append(np.diff(data).std())
        
        # Spectral entropy
        metrics['spectral_entropy'].append(spectral_entropy(data))
        
        # Sample entropy
        se = sample_entropy(series)
        metrics['sample_entropy'].append(se)
    
    # Average across columns
    return {k: np.nanmean(v) if v else 0 for k, v in metrics.items()}

def compute_mase(original, reconstructed):
    """Compute Mean Absolute Scaled Error (MASE)."""
    naive_error = np.mean(np.abs(np.diff(original)))
    if naive_error == 0:
        return 0  # Avoid division by zero
    return np.mean(np.abs(original - reconstructed)) / naive_error

def process_window(df_window: pd.DataFrame, method: str):
    """Enhanced processing with multiple metrics."""
    if df_window.isnull().any().any():
        return (0,)*2  # Skip if there are NaN values

    df_nots = df_window.drop(columns=["timestamp"])
    # original_metrics = compute_variability_metrics(df_nots)
    
    # Create test scenarios
    test_cases = [
        ('start', slice(0, REMOVE_SIZE)),
        ('middle', slice(WINDOW_SIZE//2 - REMOVE_SIZE//2, WINDOW_SIZE//2 + REMOVE_SIZE//2)),
        ('end', slice(-REMOVE_SIZE, None))
    ]
    
    results = []
    for name, mask in test_cases:
        test_df = df_nots.copy()
        test_df.iloc[mask] = np.nan
        interp_df, _ = custom_interp(test_df, method)
        
        # Calculate errors
        mase = compute_mase(df_nots.values, interp_df.values)
        dtw = sum(compute_dtw(df_nots[col], interp_df[col]) for col in df_nots.columns)
        
        # # Calculate variability preservation
        # interp_metrics = compute_variability_metrics(interp_df.iloc[mask])
        # var_ratio = { 
        #     k: interp_metrics[k] / (original_metrics[k] + 1e-10)  # Smoothing
        #     for k in original_metrics
        # }
        
        results.append((mase, dtw))
    
    # Combine results from all test cases
    return (
        np.mean([r[0] for r in results]),  # MASE
        np.mean([r[1] for r in results]),  # DTW
        # np.mean([r[2]['std_diff'] for r in results]),
        # np.mean([r[2]['spectral_entropy'] for r in results]),
        # np.mean([r[2]['sample_entropy'] for r in results])
    )

def eval_interp(df_patient: pd.DataFrame, method: str):
    """Enhanced evaluation with multiple metrics."""
    metrics = {
        'mase': [],
        'dtw': [],
        'std_ratio': [],
        'spectral_entropy_ratio': [],
        'sample_entropy_ratio': []
    }
    
    windows = [df_patient.iloc[i:i+WINDOW_SIZE] 
               for i in range(len(df_patient) - WINDOW_SIZE + 1)]
    
    with mp.Pool(PROCESSES) as pool:
        process = partial(process_window, method=method)
        results = list(tqdm(pool.imap(process, windows), total=len(windows)))
    
    for mase, dtw in results:
        metrics['mase'].append(mase)
        metrics['dtw'].append(dtw)
        # metrics['std_ratio'].append(std_r)
        # metrics['spectral_entropy_ratio'].append(spec_r)
        # metrics['sample_entropy_ratio'].append(ent_r)
    
    # Calculate final scores
    return {
        'mase': np.nanmean(metrics['mase']),
        'dtw': np.nanmean(metrics['dtw']),
        # 'variability_score': (
        #     np.nanmean(metrics['std_ratio']) +
        #     np.nanmean(metrics['spectral_entropy_ratio']) +
        #     np.nanmean(metrics['sample_entropy_ratio'])
        # ) / 3
    }

def visualize_results(metrics):
    """Create comparison plots."""
    methods = list(metrics.keys())
    scores = [m['variability_score'] for m in metrics.values()]
    errors = [m['mase'] for m in metrics.values()]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].bar(methods, scores)
    ax[0].set_title('Variability Preservation Score')
    ax[0].set_ylim(0, 1)
    
    ax[1].bar(methods, errors)
    ax[1].set_title('Mean Absolute Scaled Error (MASE)')
    
    plt.tight_layout()
    plt.show()

def get_data():
    """
    Retrieve and preprocess patient data.
    
    Returns:
        pd.DataFrame: Processed patient dataset.
    """
    df_patient, _ = retrieve_patient_data(
        processInstanceId=processInstanceId,
        start_date=START_DATE,
        end_date=END_DATE,
        no_cache=True,
        interpolate=False,
    )
    return df_patient.drop(columns=["steps"])  # Remove unnecessary columns

if __name__ == "__main__":
    # Load your data
    df_patient = get_data()  # Implement your data loading
    
    # Evaluate methods
    metrics = {method: eval_interp(df_patient, method) for method in METHODS}
    
    # Print results
    print("\nMethod Comparison:")
    for method, vals in sorted(metrics.items(), key=lambda x: x[1]['dtw']):
        print(f"{method:8}: MASE={vals['mase']:.3f} | DTW={vals['dtw']:.3f}")
    
    # # Visualize
    # visualize_results(metrics)


"""
nearest_consec: MASE=0.012 | DTW=37.400
nearest_consec_1: MASE=0.012 | DTW=37.427
savitzky_golay: MASE=0.010 | DTW=41.138
nearest : MASE=0.011 | DTW=42.716
loess   : MASE=0.011 | DTW=44.361
linear  : MASE=0.031 | DTW=124.210
pchip   : MASE=0.135 | DTW=553.089
spline  : MASE=0.628 | DTW=2594.644
"""