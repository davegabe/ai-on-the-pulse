import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pytz
from datetime import timedelta

from data.ATHOME.constants import ATHOME_CSV, ATHOME_SLEEP_PHASES_CSV, PATIENTS
from data.utils.data_processing_utils import (
    generate_sliding_windows,
    apply_k_fold_split,
    get_cache_filepath,
    save_to_cache,
    load_from_cache,
    scale_features,
    print_class_distribution as util_print_class_distribution,
    calculate_class_weights
)

# Define timezone configuration
LOCAL_TIMEZONE = pytz.timezone('Europe/Rome')  # Use the appropriate timezone for your users
UTC_TIMEZONE = pytz.UTC

# Define sleep phase mappings
SLEEP_PHASES = {'awake': 0, 'light': 1, 'deep': 2, 'rem': 3}
BINARY_SLEEP_PHASES = {'awake': 0, 'light': 1, 'deep': 1, 'rem': 1}  # Binary: awake vs sleep
# Update feature columns to use cyclic time representation
FEATURE_COLS = ['systolic', 'diastolic', 'spo2', 'hrv', 'cvrr', 'breath', 'pulse', 'time_sin', 'time_cos']
PERC_THRESHOLD = 0.5 # Threshold for percentage of valid data in a window
REMOVE_BEFORE = True # Remove data before the first sleep phase

class ATHOMESleepDataset(Dataset):
    def __init__(
            self,
            patient_id: str | int = "0",
            data_csv: str = ATHOME_CSV,
            sleep_phases_csv: str = ATHOME_SLEEP_PHASES_CSV,
            win_size: int = 15,  # 15-minute windows by default
            step_size: int = 1,  # 1-minute step for sliding window
            flag="train",
            k_split=5,
            k=0,
            scaler=None,
            use_cache: bool = True,
            binary: bool = False,  # Parameter for binary classification
        ):
        """
        Load the sleep phase classification dataset.
        
        Args:
            patient_id: ID of the patient (str or int). Use "-1" or -1 to merge all patients.
            data_csv: Path to sensor data CSV file with {patient_id} placeholder
            sleep_phases_csv: Path to sleep phases CSV file with {patient_id} placeholder
            win_size: Size of sliding window in minutes
            step_size: Step size for sliding window in minutes
            flag: Split selection ('train' or 'test')
            k_split: Number of folds for cross-validation
            k: Current fold index (0 to k_split-1, or -1 for last fold)
            scaler: Feature scaler (StandardScaler, MinMaxScaler, or RobustScaler)
            use_cache: Whether to use cached preprocessed data
            binary: If True, use binary classification (awake=0 vs. sleep=1)
        """
        # Ensure patient_id is a string for consistency
        self.patient_id = str(patient_id)
        
        # Initialize parameters
        if scaler is None:
            scaler = RobustScaler()
            
        if not isinstance(scaler, (StandardScaler, MinMaxScaler, RobustScaler)):
            raise ValueError(f"Scaler must be an instance of StandardScaler, MinMaxScaler or RobustScaler, not {type(scaler)}")
        
        self.scaler = scaler
        self.flag = flag
        self.win_size = win_size
        self.step_size = step_size
        self.binary = binary  # Store binary classification flag
        self.k_split = k_split
        self.k = k
        
        # Select appropriate sleep phase mapping
        self.sleep_phase_mapping = BINARY_SLEEP_PHASES if binary else SLEEP_PHASES
        self.num_classes = 2 if binary else len(SLEEP_PHASES)
        
        # Initialize attributes for class counts and labels
        self.classes_ = None
        self.class_counts = None
        
        # Get the data folder
        self.base_path = os.getenv("BASE_PATH")
        if self.base_path is None:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            print(f"BASE_PATH not set, using {self.base_path}")

        # Try loading from cache first
        if use_cache and self._try_load_cache(flag):
            return

        # Process data from raw files
        self._load_and_process_data(self.patient_id, data_csv, sleep_phases_csv)
        self._split_and_scale_data(flag)
        self._cache_processed_data(use_cache)
        
        # Report class distribution
        self._print_class_distribution()
        print(f"Dataset prepared: {len(self.windows)} windows for patient(s) {patient_id}, binary={binary}, flag {flag}")
    
    def _try_load_cache(self, flag):
        """Try loading the dataset from cache"""
        cache_dir = os.path.join(self.base_path, "cache_processed")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create descriptive cache key
        prefix = "all_patients" if self.patient_id == "-1" else f"patient_{self.patient_id}"
        binary_suffix = "_binary" if self.binary else ""
        params_dict = {
            "win_size": f"{self.win_size}m",
            "step_size": f"{self.step_size}m",
            "features": f"f{len(FEATURE_COLS)}",
            "threshold": f"p{PERC_THRESHOLD}"
        }
        if self.binary:
            params_dict["type"] = "binary"
            
        cache_file = get_cache_filepath(
            cache_dir,
            f"sleep_data_{prefix}",
            params_dict
        )
        
        if os.path.exists(cache_file):
            try:
                cached_data = load_from_cache(cache_file)
                self.all_windows = cached_data['windows']
                self.all_labels = cached_data['labels']
                
                # Apply k-fold split
                self.windows, self.labels = apply_k_fold_split(
                    self.all_windows, self.all_labels, self.k_split, self.k, self.flag,
                    shuffle_data=True
                )
                
                # Scale features
                scaler_path = self._get_scaler_path(flag)
                if flag == 'train' or not os.path.exists(scaler_path):
                    fit = flag == 'train'
                    # Get indices of time features to skip scaling
                    time_feature_indices = [FEATURE_COLS.index('time_sin'), FEATURE_COLS.index('time_cos')]
                    self.windows, self.scaler = scale_features(
                        self.windows, self.scaler, 
                        exclude_indices=time_feature_indices,
                        fit_scaler=fit
                    )
                    if fit:
                        save_to_cache(self.scaler, scaler_path)
                else:
                    try:
                        self.scaler = load_from_cache(scaler_path)
                        time_feature_indices = [FEATURE_COLS.index('time_sin'), FEATURE_COLS.index('time_cos')]
                        self.windows, _ = scale_features(
                            self.windows, self.scaler, 
                            exclude_indices=time_feature_indices,
                            fit_scaler=False
                        )
                    except Exception as e:
                        print(f"Error loading scaler: {e}. Using unfitted scaler.")
                        time_feature_indices = [FEATURE_COLS.index('time_sin'), FEATURE_COLS.index('time_cos')]
                        self.windows, self.scaler = scale_features(
                            self.windows, self.scaler, 
                            exclude_indices=time_feature_indices,
                            fit_scaler=False
                        )
                
                return True
            except Exception as e:
                print(f"Error loading from cache: {e}. Processing from raw data.")
        
        return False

    def _load_and_process_data(self, patient_id, data_csv, sleep_phases_csv):
        """Load and process data for all patients or a single patient"""
        # Process all patients or single patient
        if patient_id == "-1":
            print(f"Processing data for all {len(PATIENTS)} patients...")
            all_windows, all_labels = [], []
            
            # Process each patient
            for pid in tqdm(range(len(PATIENTS)), desc="Processing patients"):
                windows, labels = self._process_patient_data(str(pid), data_csv, sleep_phases_csv)
                if len(windows) > 0:  # Only include patients with valid data
                    all_windows.append(windows)
                    all_labels.append(labels)
            
            # Merge all patient data
            self.all_windows = np.vstack(all_windows) if all_windows else np.array([])
            self.all_labels = np.concatenate(all_labels) if all_labels else np.array([])
            print(f"Merged data from {len(PATIENTS)} patients: {len(self.all_windows)} total windows")
        else:
            # Process single patient
            self.all_windows, self.all_labels = self._process_patient_data(patient_id, data_csv, sleep_phases_csv)
        
        # Validate data
        if len(self.all_windows) == 0:
            raise ValueError(f"No valid data windows found for patient(s)")

    def _process_patient_data(self, patient_id, data_csv, sleep_phases_csv):
        """Process raw data into windows with sleep phase labels for a single patient"""
        # Load sensor data
        print(f"Processing patient {patient_id}...")
        try:
            # Load sensor readings
            patient_data = pd.read_csv(os.path.join(self.base_path, data_csv.format(patient_id=patient_id)))
            patient_data['timestamp'] = pd.to_datetime(patient_data['timestamp'], utc=True)
            
            # Load sleep phases
            sleep_phases = pd.read_csv(os.path.join(self.base_path, sleep_phases_csv.format(patient_id=patient_id)))
            sleep_phases['start_timestamp'] = pd.to_datetime(sleep_phases['start_timestamp'], utc=True)
            sleep_phases['end_timestamp'] = pd.to_datetime(sleep_phases['end_timestamp'], utc=True)
            
            # Check for minimum data
            if len(patient_data) < 2 or len(sleep_phases) == 0:
                print(f"  Insufficient data for patient {patient_id}, skipping")
                return np.array([]), np.array([])
                
            # Make all timestamps timezone-naive for consistent comparison 
            patient_data['timestamp'] = patient_data['timestamp'].dt.tz_localize(None)
            sleep_phases['start_timestamp'] = sleep_phases['start_timestamp'].dt.tz_localize(None)
            sleep_phases['end_timestamp'] = sleep_phases['end_timestamp'].dt.tz_localize(None)
                
            # Remove data before the first sleep phase day if configured
            if REMOVE_BEFORE and len(sleep_phases) > 0:
                # Find date of first sleep phase
                first_sleep_date = sleep_phases['start_timestamp'].min().date()
                data_count_before = len(patient_data)
                
                # Filter data to only include dates on or after the first sleep phase
                patient_data = patient_data[patient_data['timestamp'].dt.date >= first_sleep_date]
                
                data_count_after = len(patient_data)
                removed_percentage = ((data_count_before - data_count_after) / data_count_before * 100) if data_count_before > 0 else 0
                print(f"  Removed {data_count_before - data_count_after} records ({removed_percentage:.1f}%) before first sleep phase date ({first_sleep_date})")
                
                # If all data was removed, return empty arrays
                if len(patient_data) < 2:
                    print(f"  All data removed for patient {patient_id} after filtering by first sleep date")
                    return np.array([]), np.array([])
            
            # Calculate cyclical time features instead of linear time
            time_of_day = patient_data['timestamp'].apply(
                lambda x: (x.hour * 3600 + x.minute * 60 + x.second) / 86400
            )
            # Convert time to cyclical features using sine and cosine
            patient_data['time_sin'] = np.sin(2 * np.pi * time_of_day)
            patient_data['time_cos'] = np.cos(2 * np.pi * time_of_day)
            
            # Check for minimum data after filtering
            if len(patient_data) < 2:
                print(f"  Insufficient data for patient {patient_id} after filtering, skipping")
                return np.array([]), np.array([])
            
            # Calculate window parameters
            # Helper function to safely get seconds from either Python timedelta or NumPy timedelta64
            def safe_total_seconds(time_diff):
                if hasattr(time_diff, 'total_seconds'):
                    return time_diff.total_seconds()
                else:
                    # Handle numpy.timedelta64
                    return time_diff.astype('timedelta64[s]').astype(float)
            
            # Calculate window parameters
            time_diff_obj = patient_data['timestamp'].iloc[1] - patient_data['timestamp'].iloc[0]
            time_diff = safe_total_seconds(time_diff_obj) / 60
                
            window_rows = int(self.win_size / time_diff)
            step_rows = max(1, int(self.step_size / time_diff))
            
            # Prepare sleep phases for labeling
            valid_phases = sleep_phases[sleep_phases['phase_type'].str.lower().isin(self.sleep_phase_mapping.keys())]
            valid_phases = valid_phases.sort_values('start_timestamp')
            valid_phases['numeric_phase'] = valid_phases['phase_type'].str.lower().map(self.sleep_phase_mapping)
            
            # Associate the label before generating sliding windows for more efficient processing.
            patient_data['label'] = self.sleep_phase_mapping['awake']
            for idx, row in valid_phases.iterrows():
                mask = (patient_data['timestamp'] >= row['start_timestamp']) & (patient_data['timestamp'] <= row['end_timestamp'])
                patient_data.loc[mask, 'label'] = row['numeric_phase']
            
            # Sort the data by timestamp
            patient_data = patient_data.sort_values('timestamp')
            
            # Store first and last date for reference
            self.first_date = patient_data['timestamp'].min()
            self.last_date = patient_data['timestamp'].max()
            
            # Calculate no data days
            dates = patient_data['timestamp'].dt.date
            all_dates = [self.first_date.date() + timedelta(days=i) 
                         for i in range((self.last_date.date() - self.first_date.date()).days + 1)]
            self.no_data_days = [date for date in all_dates if date not in dates.values]
            
            patient_data.drop(columns=['timestamp'], inplace=True)
            # Remove columns which are not in FEATURE_COLS
            label = patient_data['label']
            patient_data = patient_data[FEATURE_COLS]
            
            # Generate windows using the utility function
            windows, labels = generate_sliding_windows(
                data_df=patient_data,
                label_df=label,
                win_size_records=window_rows,
                step_size_records=step_rows,
                min_valid_data_percentage=PERC_THRESHOLD,
                remove_before_first_valid=REMOVE_BEFORE,
                classification=True,
            )
            
            print(f"  Created {len(windows)} valid windows after NaN filtering")
            
            return windows, labels
            
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])

    def _split_and_scale_data(self, flag):
        """Apply k-fold splitting and scale the data"""
        # Apply k-fold split
        self.windows, self.labels = apply_k_fold_split(
            self.all_windows, self.all_labels, self.k_split, self.k, flag,
            shuffle_data=True
        )
        
        # Scale features
        scaler_path = self._get_scaler_path(flag)
        if flag == 'train' or not os.path.exists(scaler_path):
            fit = flag == 'train'
            # Get indices of time features to skip scaling
            time_feature_indices = [FEATURE_COLS.index('time_sin'), FEATURE_COLS.index('time_cos')]
            self.windows, self.scaler = scale_features(
                self.windows, self.scaler, 
                exclude_indices=time_feature_indices,
                fit_scaler=fit
            )
            if fit:
                save_to_cache(self.scaler, scaler_path)
        else:
            try:
                self.scaler = load_from_cache(scaler_path)
                time_feature_indices = [FEATURE_COLS.index('time_sin'), FEATURE_COLS.index('time_cos')]
                self.windows, _ = scale_features(
                    self.windows, self.scaler, 
                    exclude_indices=time_feature_indices,
                    fit_scaler=False
                )
            except Exception as e:
                print(f"Error loading scaler: {e}. Using unfitted scaler.")
                time_feature_indices = [FEATURE_COLS.index('time_sin'), FEATURE_COLS.index('time_cos')]
                self.windows, self.scaler = scale_features(
                    self.windows, self.scaler, 
                    exclude_indices=time_feature_indices,
                    fit_scaler=False
                )
        
        print("Feature scaling applied (time features skipped).")

    def _cache_processed_data(self, use_cache):
        """Save processed data to cache if requested"""
        if use_cache:
            cache_dir = os.path.join(self.base_path, "cache_processed")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create descriptive cache key
            prefix = "all_patients" if self.patient_id == "-1" else f"patient_{self.patient_id}"
            binary_suffix = "_binary" if self.binary else ""
            params_dict = {
                "win_size": f"{self.win_size}m",
                "step_size": f"{self.step_size}m",
                "features": f"f{len(FEATURE_COLS)}",
                "threshold": f"p{PERC_THRESHOLD}"
            }
            if self.binary:
                params_dict["type"] = "binary"
                
            cache_file = get_cache_filepath(
                cache_dir,
                f"sleep_data_{prefix}",
                params_dict
            )
            
            cache_data = {
                'windows': self.all_windows,
                'labels': self.all_labels,
            }
            save_to_cache(cache_data, cache_file)

    def _get_scaler_path(self, flag):
        """Get path for scaler cache file"""
        cache_dir = os.path.join(self.base_path, "cache_processed")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create descriptive cache key
        prefix = "all_patients" if self.patient_id == "-1" else f"patient_{self.patient_id}"
        binary_suffix = "_binary" if self.binary else ""
        
        params_dict = {
            "win_size": f"{self.win_size}m",
            "step_size": f"{self.step_size}m",
            "k": self.k,
            "k_split": f"of_{self.k_split}",
            "features": f"f{len(FEATURE_COLS)}",
            "threshold": f"p{PERC_THRESHOLD}"
        }
        if self.binary:
            params_dict["type"] = "binary"
            
        return get_cache_filepath(
            cache_dir,
            f"scaler_{prefix}",
            params_dict
        )

    def _print_class_distribution(self):
        """Print distribution of classes in the dataset"""
        # Find the mapping to use based on whether we're doing binary classification
        phase_mapping = BINARY_SLEEP_PHASES if self.binary else SLEEP_PHASES
        
        # Use the utility function for consistent class distribution reporting
        class_info = util_print_class_distribution(
            self.labels, 
            class_mapping=phase_mapping
        )
        
        # Store the class information
        self.classes_ = np.array(list(class_info.keys()))
        self.class_counts = np.array(list(class_info.values()))
        
        # Add extra information specific to this dataset
        print(f"Window shape: {self.windows.shape[1]} timesteps, {self.windows.shape[2]} features")

    def get_class_weights(self):
        """Calculate class weights based on the current dataset split (self.labels)."""
        if self.labels is None or len(self.labels) == 0:
            print("Warning: No labels found for calculating class weights.")
            return None
            
        # For binary classification, we typically don't need class weights
        # as it's more straightforward to adjust the decision threshold
        if self.binary:
            print("Binary classification detected - returning None for class weights.")
            return None

        # Use the common utility function to calculate weights
        weights = calculate_class_weights(self.labels, self.num_classes)
        
        # Convert to torch tensor for model training
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x = torch.tensor(self.windows[index], dtype=torch.float)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


class ATHOMESleepUniTS(ATHOMESleepDataset):
    def __init__(
            self,
            win_size=15,
            step_size=1,
            flag="train",
            k_split=5,
            k=0,
            data_csv=ATHOME_CSV,
            sleep_phases_csv=ATHOME_SLEEP_PHASES_CSV,
            patient_id: str | int = "0",
            use_cache=True,
            binary=False,  # Add binary parameter
        ):
        """
        Sleep classification dataset formatted for UniTS model.
        Set patient_id="-1" or -1 to use all available patients.
        Set binary=True for binary classification (awake vs sleep).
        """
        super().__init__(
            patient_id=patient_id,  # Will be converted to string in parent constructor
            data_csv=data_csv,
            sleep_phases_csv=sleep_phases_csv,
            win_size=win_size,
            step_size=step_size,
            flag=flag,
            k_split=k_split,
            k=k,
            scaler=RobustScaler(),
            use_cache=use_cache,
            binary=binary,  # Pass binary parameter
        )


if __name__ == "__main__":
    # Test with a single patient (using int for patient_id)
    print("Testing with single patient (ID=0)")
    dataset = ATHOMESleepUniTS(patient_id=0, use_cache=True)
    dataset = ATHOMESleepUniTS(patient_id=0, flag="test", use_cache=True)
    
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"Sample shape: {x.shape}, Label: {y.item()}")
    
    # Test with all patients (using -1 as int)
    print("\nTesting with all patients")
    all_dataset = ATHOMESleepUniTS(patient_id=-1, use_cache=True)
    all_dataset = ATHOMESleepUniTS(patient_id=-1, flag="test", use_cache=True)

    if len(all_dataset) > 0:
        x, y = all_dataset[0]
        print(f"Sample shape: {x.shape}, Label: {y.item()}")
      
    # Test binary classification
    print("\nTesting binary classification")
    binary_dataset = ATHOMESleepUniTS(patient_id=-1, use_cache=True, binary=True)
    binary_dataset = ATHOMESleepUniTS(patient_id=-1, flag="test", use_cache=True, binary=True)

    if len(binary_dataset) > 0:
        x, y = binary_dataset[0]
        print(f"Sample shape: {x.shape}, Label: {y.item()} (binary)")
