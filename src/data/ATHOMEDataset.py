from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from data.ATHOME.constants import ATHOME_CSV, ATHOME_SLEEP_CSV, ATHOME_STEPS_CSV
from data.utils.anomalies import get_features, get_seq_to_impute, main_features
from data.utils.data_processing_utils import (
    generate_sliding_windows,
    apply_k_fold_split,
    get_cache_filepath,
    save_to_cache,
    load_from_cache,
    scale_features,
)


class ATHOMEFeatures(Dataset):
    def __init__(
            self,
            patient_id: str,
            data_csv: str,
            sleep_csv: str,
            steps_csv: str,
            anomaly_type: str,
            win_size: int,
            flag="train",
            k_split=5,
            k=0,
            scaler: StandardScaler | MinMaxScaler | RobustScaler = None,
            use_cache: bool = True,
            perc_threshold: float = 0.5,
        ):
        """
        Load the Features dataset for anomaly detection.

        Args:
            patient_id (str): ID of the patient to select
            data_csv (str): Path to the data CSV file.
            sleep_csv (str): Path to the sleep CSV file.
            steps_csv (str): Path to the steps CSV file.
            anomaly_type (str): Type of anomaly to generate.
            win_size (int): Size of the sliding window.
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 5)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            scaler: Feature scaler instance
            use_cache (bool): Whether to use cached data
            perc_threshold (float): Threshold for percentage of valid data in a window
        """
        # Check the scaler
        if scaler is None:
            scaler = RobustScaler()
        
        if not isinstance(scaler, (StandardScaler, MinMaxScaler, RobustScaler)):
            raise ValueError(f"Scaler must be an instance of StandardScaler, MinMaxScaler or RobustScaler, not {type(scaler)}")
        
        self.scaler = scaler
        self.anomaly_type = anomaly_type
        self.flag = flag
        self.win_size = win_size
        self.perc_threshold = perc_threshold
        self.patient_id = patient_id
        self.k = k
        self.k_split = k_split
        self.column_names = []

        # Get the data folder
        self.base_path = os.getenv("BASE_PATH")
        if self.base_path is None:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            print(f"BASE_PATH not set, using {self.base_path}")

        # Try loading from cache first
        if use_cache and self._try_load_cache(flag):
            return

        data, sleep_data, steps_data = self._load_csv_data(data_csv, sleep_csv, steps_csv)
        self._process_raw_data(data, sleep_data, steps_data)
        self._split_and_scale_data(flag)
        self._cache_processed_data(use_cache)
        print(f"Dataset prepared: {len(self.windows)} windows for patient {patient_id}, anomaly type {anomaly_type}, flag {flag}")
    
    def _try_load_cache(self, flag):
        """Try loading the dataset from cache"""
        cache_dir = os.path.join(self.base_path, "cache_processed")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create descriptive cache key
        params_dict = {
            "patient": self.patient_id,
            "anomaly_type": self.anomaly_type,
            "win_size": self.win_size,
            "threshold": f"p{self.perc_threshold}"
        }
        
        cache_file = get_cache_filepath(
            cache_dir,
            f"athome_features",
            params_dict
        )
        
        if os.path.exists(cache_file):
            try:
                cached_data = load_from_cache(cache_file)
                self.windows = cached_data['windows']
                self.labels = cached_data['labels']
                self.main_features_idxs = cached_data['main_features_idxs']
                self.column_names = cached_data['column_names']
                self.seq_to_impute = cached_data.get('seq_to_impute')
                self.first_date = cached_data.get('first_date')
                self.last_date = cached_data.get('last_date')
                self.no_data_days = cached_data.get('no_data_days', [])
                
                # Apply k-fold split
                self.windows, self.labels = apply_k_fold_split(
                    self.windows, self.labels, self.k_split, self.k, self.flag
                )
                
                # Scale features
                scaler_path = self._get_scaler_path(flag)
                if flag == 'train' or not os.path.exists(scaler_path):
                    fit = flag == 'train'
                    self.windows, self.scaler = scale_features(
                        self.windows, self.scaler, fit_scaler=fit
                    )
                    if fit:
                        save_to_cache(self.scaler, scaler_path)
                else:
                    self.scaler = load_from_cache(scaler_path)
                    self.windows, _ = scale_features(
                        self.windows, self.scaler, fit_scaler=False
                    )
                
                return True
            except Exception as e:
                print(f"Error loading from cache: {e}. Processing from raw data.")
        
        return False

    def _load_csv_data(self, data_csv, sleep_csv, steps_csv):
        """Load CSV data for the patient"""
        try:
            all_data = pd.read_csv(
                os.path.join(self.base_path, data_csv.format(patient_id=self.patient_id))
            )
            sleep_data = pd.read_csv(
                os.path.join(self.base_path, sleep_csv.format(patient_id=self.patient_id))
            )
            steps_data = pd.read_csv(
                os.path.join(self.base_path, steps_csv.format(patient_id=self.patient_id))
            )
        except Exception as e:
            print(f"Error loading data for patient {self.patient_id}: {e}")
            raise
        
        return all_data, sleep_data, steps_data

    def _process_raw_data(self, all_data, sleep_data, steps_data):
        """Process raw data into features, timestamps, windows, and labels"""
        # Get the features for the anomaly type
        data, self.main_features_idxs, seq_type = get_features(
            all_data, sleep_data, steps_data, self.anomaly_type
        )
        
        # Set the sequence to impute
        self.seq_to_impute = get_seq_to_impute(seq_type, self.win_size)
        
        # Store original timestamp data for first/last date
        if 'timestamp' in data.columns:
            self.first_date = data['timestamp'].min()
            self.last_date = data['timestamp'].max()
            
            # Calculate no data days
            dates = data['timestamp'].dt.date
            all_dates = [datetime.date(self.first_date + timedelta(days=i)) 
                         for i in range((self.last_date - self.first_date).days + 1)]
            self.no_data_days = [date for date in all_dates if date not in dates.values]
        else:
            self.first_date = None
            self.last_date = None
            self.no_data_days = []
        
        # Set step size (1/4 of window size, or 1 if window size < 4)
        step_size = max(1, self.win_size // 4)
                
        # Sort the data by timestamp
        data = data.sort_values('timestamp')
        labels = data['Label'] 
        data.drop(columns=['timestamp', 'Label'], inplace=True)

        # Store data columns for later use
        self.column_names = data.columns.tolist()
        
        windows, labels = generate_sliding_windows(
            data_df=data,
            label_df=labels,
            win_size_records=self.win_size,
            step_size_records=step_size,
            min_valid_data_percentage=self.perc_threshold,
            remove_before_first_valid=True
        )
        
        # Store the windows and labels before splitting
        self.all_windows = windows
        self.all_labels = labels

    def _split_and_scale_data(self, flag):
        """Apply k-fold splitting and scale the data"""
        # Apply k-fold split
        self.windows, self.labels = apply_k_fold_split(
            self.all_windows, self.all_labels, self.k_split, self.k, flag
        )
        
        # Scale features
        scaler_path = self._get_scaler_path(flag)
        if flag == 'train' or not os.path.exists(scaler_path):
            fit = flag == 'train'
            self.windows, self.scaler = scale_features(
                self.windows, self.scaler, fit_scaler=fit
            )
            if fit:
                save_to_cache(self.scaler, scaler_path)
        else:
            try:
                self.scaler = load_from_cache(scaler_path)
                self.windows, _ = scale_features(
                    self.windows, self.scaler, fit_scaler=False
                )
            except Exception as e:
                print(f"Error loading scaler: {e}. Using unfitted scaler.")
                self.windows, self.scaler = scale_features(
                    self.windows, self.scaler, fit_scaler=False
                )

    def _cache_processed_data(self, use_cache):
        """Save processed data to cache if requested"""
        if use_cache:
            cache_dir = os.path.join(self.base_path, "cache_processed")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create descriptive cache key
            params_dict = {
                "patient": self.patient_id,
                "anomaly_type": self.anomaly_type,
                "win_size": self.win_size,
                "threshold": f"p{self.perc_threshold}"
            }
            
            cache_file = get_cache_filepath(
                cache_dir,
                f"athome_features",
                params_dict
            )
            
            cache_data = {
                'windows': self.all_windows,
                'labels': self.all_labels,
                'main_features_idxs': self.main_features_idxs,
                'column_names': self.column_names,
                'seq_to_impute': self.seq_to_impute,
                'first_date': self.first_date,
                'last_date': self.last_date,
                'no_data_days': self.no_data_days
            }
            save_to_cache(cache_data, cache_file)

    def _get_scaler_path(self, flag):
        """Get path for scaler cache file"""
        cache_dir = os.path.join(self.base_path, "cache_processed")
        os.makedirs(cache_dir, exist_ok=True)
        
        params_dict = {
            "patient": self.patient_id,
            "anomaly_type": self.anomaly_type,
            "win_size": self.win_size,
            "k": self.k,
            "k_split": f"of_{self.k_split}",
            "threshold": f"p{self.perc_threshold}"
        }
        
        return get_cache_filepath(
            cache_dir,
            f"scaler_athome_features",
            params_dict
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        raise NotImplementedError


class ATHOMEFeaturesUniTS(ATHOMEFeatures):
    def __init__(
            self,
            win_size: int,
            anomaly_type: int,
            flag="train",
            k_split=5,
            k=0,
            data_csv=ATHOME_CSV,
            sleep_csv=ATHOME_SLEEP_CSV,
            steps_csv=ATHOME_STEPS_CSV,
            patient_id: str = "",
            use_cache=True,
            perc_threshold=0.5,
        ):
        """
        Load the Features dataset for anomaly detection, formatted for UniTS model.

        Args:
            win_size (int): Size of the sliding window.
            anomaly_type (int): Type of anomaly to generate.
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 5)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            data_csv (str): Path to the data CSV file.
            sleep_csv (str): Path to the sleep CSV file.
            steps_csv (str): Path to the steps CSV file.
            patient_id (str): ID of the patient to select
            use_cache (bool): Whether to use cached data
            perc_threshold (float): Threshold for percentage of valid data in a window
        """
        scaler = RobustScaler()
        super().__init__(
            patient_id=patient_id,
            data_csv=data_csv,
            sleep_csv=sleep_csv,
            steps_csv=steps_csv,     
            anomaly_type=anomaly_type,
            win_size=win_size,
            flag=flag,
            k_split=k_split,
            k=k,
            scaler=scaler,
            use_cache=use_cache,
            perc_threshold=perc_threshold,
        )

    def __getitem__(self, index):
        window = self.windows[index]
        # Convert NaNs to zeros for model compatibility (already done in generate_sliding_windows)
        window = np.nan_to_num(window, nan=0.0)
        
        # y is defined as a vector of zeros of length seq_to_impute.
        y = np.zeros(self.seq_to_impute)
        
        return torch.tensor(window, dtype=torch.float), torch.tensor(y, dtype=torch.float)


if __name__ == "__main__":
    # Test the ATHOMEFeaturesUniTS class
    for anomaly_type in main_features.keys():
        dataset = ATHOMEFeaturesUniTS(
            win_size=3, 
            anomaly_type=anomaly_type, 
            flag="train", 
            patient_id="0", 
            k=-1,
            use_cache=True,
        )
        dataset = ATHOMEFeaturesUniTS(
            win_size=3, 
            anomaly_type=anomaly_type, 
            flag="test", 
            patient_id="0", 
            k=-1,
            use_cache=True,
        )

        print(f"Anomaly type: {anomaly_type}")
        print(f"Number of samples: {len(dataset)}")
        print(f"First date: {dataset.first_date}")
        print(f"Last date: {dataset.last_date}")
        if dataset.first_date and dataset.last_date:
            n_days = (pd.to_datetime(dataset.last_date) - pd.to_datetime(dataset.first_date)).days
            print(f"Number of days: {n_days}")
        print(f"Number of days with no data: {len(dataset.no_data_days)}")
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"Sample shape: {x.shape}, Label shape: {y.shape}")
        print("#" * 80)
