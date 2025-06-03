import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

from tqdm import tqdm
from joblib import Parallel, delayed  # Added for parallelization


def generate_sliding_windows(
    data_df: pd.DataFrame,
    label_df: pd.DataFrame,
    win_size_records: int,
    step_size_records: int,
    min_valid_data_percentage: float,
    remove_before_first_valid: bool = True,
    classification: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding windows from time series data, filtering for windows with sufficient valid data.
    
    Args:
        data_df: DataFrame containing time series data
        label_df: DataFrame containing labels for each time step
        win_size_records: Number of records in each window
        step_size_records: Step size for sliding window
        min_valid_data_percentage: Minimum percentage of non-NaN values required in feature columns
        remove_before_first_valid: If True, skip data before first window with valid data
        classification: If True, treat labels as categorical (for classification tasks)
        
    Returns:
        Tuple containing:
        - numpy array of valid windows with shape (n_windows, win_size_records, n_features)
        - numpy array of corresponding labels
    """    
    # Get the indices where we'll start each window
    total_records = len(data_df)
    if total_records < win_size_records:
        return np.array([]), np.array([])
    
    # Vectorized calculation of valid data before parallel
    non_nan_counts = data_df.notna().sum(axis=1)
    rolling_valid_counts = non_nan_counts.rolling(win_size_records).sum()
    # Compare with threshold
    needed_count = min_valid_data_percentage * win_size_records
    valid_mask = rolling_valid_counts >= needed_count
    
    # Replace the existing start_indices list comprehension with a loop:
    start_indices = []
    first_valid_found = False if remove_before_first_valid else True
    for i in range(0, total_records - win_size_records + 1, step_size_records):
        if valid_mask.iloc[i + win_size_records - 1]:
            if not first_valid_found:
                first_valid_found = True
            start_indices.append(i) if first_valid_found else None
    
    # Initialize lists to hold valid windows and labels
    valid_windows = []
    valid_labels = []
    
    # Iterate over the data to create sliding windows
    def process_window(start_idx):
        # Extract window data
        end_idx = start_idx + win_size_records
        window_df = data_df.iloc[start_idx:end_idx]
        
        # Extract label using the provided function
        labels = label_df.iloc[start_idx:end_idx].values

        # If classification, we say 0 if most labels are 0, else 1
        if classification:
            n_zeros = np.sum(labels == 0)
            if n_zeros > len(labels) / 2:
                label = 0
            else:
                label = 1
        else:
            label = labels # For anomaly detection, we keep all labels
        
        return (window_df.values, label)

    # Use tqdm for progress bar in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_window)(start_idx) for start_idx in tqdm(start_indices, desc="Generating windows", unit="window")
    )
    
    for r in results:
        if r is not None:
            valid_windows.append(r[0])
            valid_labels.append(r[1])
    
    if not valid_windows:
        return np.array([]), np.array([])
    
    return np.array(valid_windows), np.array(valid_labels)


def apply_k_fold_split(
    all_windows: np.ndarray,
    all_labels: np.ndarray,
    k_split: int,
    current_fold_index: int,
    split_type: str,
    shuffle_data: bool = True,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply k-fold cross-validation split to the data.
    
    Args:
        all_windows: Array of all windows
        all_labels: Array of all labels
        k_split: Number of folds for cross-validation
        current_fold_index: Index of current fold (0 to k_split-1, or -1 for last fold)
        split_type: 'train' or 'test'
        shuffle_data: Whether to shuffle the data before splitting
        random_seed: Random seed for shuffling
        
    Returns:
        Tuple of (selected_windows, selected_labels) for the requested split
    """
    if len(all_windows) == 0:
        return np.array([]), np.array([])
        
    # Handle negative index (use last fold)
    if current_fold_index == -1:
        current_fold_index = k_split - 1
    
    if current_fold_index < 0 or current_fold_index >= k_split:
        raise ValueError(f"Fold index must be between 0 and {k_split-1}, got {current_fold_index}")
    
    # Create indices and shuffle if requested
    indices = np.arange(len(all_windows))
    if shuffle_data:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # Split indices into folds
    fold_indices = np.array_split(indices, k_split)
    
    # Get test indices for current fold
    test_indices = fold_indices[current_fold_index]
    
    # Get train indices (all indices except test indices)
    train_indices = np.concatenate([fold_indices[i] for i in range(k_split) if i != current_fold_index])
    
    # Return appropriate split
    if split_type.lower() == 'train':
        return all_windows[train_indices], all_labels[train_indices]
    elif split_type.lower() == 'test':
        return all_windows[test_indices], all_labels[test_indices]
    else:
        raise ValueError(f"Split type must be 'train' or 'test', got {split_type}")


def get_cache_filepath(
    base_dir: str,
    prefix: str,
    params_dict: Dict[str, Any],
    suffix: str = '.pkl'
) -> str:
    """
    Generate a cache file path based on parameters.
    
    Args:
        base_dir: Base directory for cache
        prefix: Prefix for the filename
        params_dict: Dictionary of parameters to include in the filename
        suffix: File extension
        
    Returns:
        Full file path for cache file
    """
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a string from the parameters
    params_str = '_'.join([f"{k}_{v}" for k, v in params_dict.items()])
    
    # Generate filename
    filename = f"{prefix}_{params_str}{suffix}"
    
    return os.path.join(base_dir, filename)


def save_to_cache(obj: Any, filepath: str) -> None:
    """
    Save an object to cache.
    
    Args:
        obj: Object to save
        filepath: File path to save to
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to cache: {os.path.basename(filepath)}")


def load_from_cache(filepath: str) -> Any:
    """
    Load an object from cache.
    
    Args:
        filepath: File path to load from
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded from cache: {os.path.basename(filepath)}")
    return obj


def scale_features(
    windows: np.ndarray,
    scaler: Any,
    exclude_indices: Optional[List[int]] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, Any]:
    """
    Scale features in windows, optionally excluding certain feature indices.
    
    Args:
        windows: Windows data with shape (n_windows, n_timesteps, n_features)
        scaler: Feature scaler instance
        exclude_indices: Indices of features to exclude from scaling
        fit_scaler: Whether to fit the scaler on the data
        
    Returns:
        Tuple of (scaled_windows, fitted_scaler)
    """
    if len(windows) == 0:
        return windows, scaler
        
    original_shape = windows.shape
    
    # Reshape for scaling (combine samples and timesteps)
    flat_windows = windows.reshape(-1, original_shape[2])
    
    if exclude_indices is None:
        exclude_indices = []
    
    # Get indices to scale (all except excluded)
    scale_indices = [i for i in range(flat_windows.shape[1]) if i not in exclude_indices]
    
    # Store excluded features
    excluded_features = flat_windows[:, exclude_indices].copy() if exclude_indices else None
    
    # Scale only the features that should be scaled
    if fit_scaler:
        scaler = scaler.fit(flat_windows[:, scale_indices])
    
    # Transform features (NaNs will remain NaNs)
    if scale_indices:
        scaled_features = scaler.transform(flat_windows[:, scale_indices])
    else:
        scaled_features = np.array([])  # No features to scale
    
    # Re-combine scaled and excluded features
    scaled_flat = np.zeros_like(flat_windows)
    
    if scale_indices:
        scaled_flat[:, scale_indices] = scaled_features
    
    if exclude_indices:
        scaled_flat[:, exclude_indices] = excluded_features
    
    # Reshape back to original shape
    scaled_windows = scaled_flat.reshape(original_shape)
    
    return scaled_windows, scaler


def print_class_distribution(labels: np.ndarray, class_mapping: Optional[Dict] = None) -> Dict:
    """
    Print the distribution of classes in the dataset.
    
    Args:
        labels: Array of labels
        class_mapping: Optional dictionary mapping numerical class values to names
        
    Returns:
        Dictionary with class counts
    """
    if len(labels) == 0:
        print("No labels in dataset")
        return {}
        
    classes, class_counts = np.unique(labels, return_counts=True)
    
    print(f"\nDataset: {len(labels)} samples")
    print("Class distribution:")
    
    class_info = {}
    for cls, count in zip(classes, class_counts):
        percentage = (count / len(labels)) * 100
        
        if class_mapping and cls in class_mapping.values():
            # Find the key for this value
            class_name = [k for k, v in class_mapping.items() if v == cls][0]
            print(f"  Class {cls} ({class_name}): {count} samples ({percentage:.2f}%)")
        else:
            print(f"  Class {cls}: {count} samples ({percentage:.2f}%)")
        
        class_info[int(cls)] = count
    
    return class_info


def calculate_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calculate class weights based on inverse frequency.
    
    Args:
        labels: Array of class labels
        num_classes: Total number of possible classes
        
    Returns:
        Array of class weights
    """
    if len(labels) == 0:
        return np.ones(num_classes, dtype=np.float32)
    
    # Get counts of each class in the dataset
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Initialize weights array
    weights = np.zeros(num_classes, dtype=np.float32)
    
    # Calculate inverse frequency
    total_samples = len(labels)
    for cls, count in zip(unique_classes, counts):
        weights[int(cls)] = total_samples / (num_classes * count)
    
    # Handle classes not present in the labels
    missing_classes = [c for c in range(num_classes) if c not in unique_classes]
    if missing_classes:
        print(f"Warning: Classes {missing_classes} not present in dataset. Assigning average weight.")
        if len(unique_classes) > 0:
            avg_weight = np.mean(weights[weights > 0])
            weights[missing_classes] = avg_weight
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.mean(weights[weights > 0])
    else:
        weights = np.ones(num_classes, dtype=np.float32)
    
    return weights
