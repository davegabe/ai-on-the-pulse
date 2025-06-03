import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import wandb

from WESAD.constants import PREPROCESSED_CSV as WESAD_CSV
from DREAMER.constants import PREPROCESSED_CSV as DREAMER_CSV
from HCI.constants import PREPROCESSED_CSV as HCI_CSV

BVP_FEATURES = ['HR_BVP', 'HRV_BVP']
ECG_FEATURES = ['HR_ECG', 'HRV_ECG']
CONF_FEATURES = ['TEMP', 'ACC_x', 'ACC_y', 'ACC_z']

class Features(Dataset):
    def __init__(self, dataset: str, flag="train", k_split=5, k=0, scaler=None, source="ECG", confounding=False, step=1):
        """
        Load the Features dataset for anomaly detection.

        Args:
            dataset (str): Dataset to use. Must be one of ['WESAD', 'DREAMER', 'HCI', etc.]
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            source (str): Source of the data. Must be one of ['ECG', 'BVP']
            confounding (bool): Whether to include confounding features (default: False)
            step (int): Step for downsampling the data (default: 1 - no downsampling)
        """
        self.flag = flag
        if scaler is None:
            raise ValueError("Scaler must be provided")
        self.scaler = scaler
        print(f"Features dataset with flag {flag}")

        # Get the data folder
        base_path = os.getenv("BASE_PATH")
        if base_path is None:
            raise ValueError("BASE_PATH environment variable not set")

        # Read the data from the correct dataset
        if dataset == "WESAD":
            path = os.path.join(base_path, WESAD_CSV)
        elif dataset == "DREAMER":
            path = os.path.join(base_path, DREAMER_CSV)
        elif dataset == "HCI":
            path = os.path.join(base_path, HCI_CSV)
        else:
            raise ValueError(f"Dataset must be one of ['WESAD', 'DREAMER', 'HCI']")
        data = pd.read_csv(path)

        # Use only HR and HRV column for the specified source signal
        if source == "ECG":
            # Remove BVP features if present
            for feature in BVP_FEATURES:
                if feature in data.columns:
                    data = data.drop(columns=[feature])
        elif source == "BVP":
            # Remove ECG features if present
            for feature in ECG_FEATURES:
                if feature in data.columns:
                    data = data.drop(columns=[feature])
        else:
            raise ValueError(f"Source must be one of ['ECG', 'BVP']")

        # Remove confounding columns if specified
        if not confounding:
            # Remove confounding features if present
            for feature in CONF_FEATURES:
                if feature in data.columns:
                    data = data.drop(columns=[feature])

        # Skip "Patient" and "timestamp" columns
        data = data.drop(columns=['Patient', 'timestamp'])
        # print("Remaining columns:", data.columns)
        data = np.nan_to_num(data.values)
        size = len(data)

        # Indices of the test split are the k-th split of data with k_split
        start_test = int(size * k / k_split)
        end_test = start_test + int(size / k_split)

        # Split the data
        if flag == 'train':
            # If k is the first split, take the rest of the data
            if k == 0:
                self.split = data[end_test:]
            # If k is the last split, take the beginning of the data
            elif k == k_split - 1:
                self.split = data[:start_test]
            # Otherwise, take the two parts of the data
            else:
                self.split = np.concatenate(
                    (data[:start_test], data[end_test:]))
            # Remove all elements with anomaly
            self.split = self.split[self.split[:, -1] == 0]
            # Split into labels and data
            self.labels = self.split[:, -1]
            self.split = self.split[:, :-1]
        elif flag == 'test':
            # Take the split
            self.split = data[start_test:end_test, :]
            # Split into labels and data
            self.labels = self.split[:, -1]
            self.split = self.split[:, :-1]
        else:
            raise ValueError(f"Flag must be one of ['train', 'test']")
        
        # Downsample the data
        self.step = step
        if self.step > 1:
            self.labels = self.labels[::self.step]
            self.split = self.split[::self.step]

        # Normalize the data
        if type(self.scaler) == MinMaxScaler or type(self.scaler) == StandardScaler:
            self.split = self.scaler.fit_transform(self.split)
        else:
            self.split = self.scaler(self.split)

        # Get the labels
        n_anomalies = len(np.where(self.labels == 1)[0])
        tot = len(self.split)
        # print(f"Features {flag} set - #anomalies: {n_anomalies}/{tot}")

        # Wandb logging
        if wandb.run is not None:
            wandb.log({
                f'n_anomalies_{flag}': n_anomalies,
                f'tot_{flag}': tot
            })

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class FeaturesUniTS(Features):
    def __init__(self, dataset: str, win_size: int, step=1, flag="train", k_split=5, k=0, source="ECG", confounding=False):
        """
        Load the Features dataset for anomaly detection.

        Args:
            dataset (str): Dataset to use. Must be one of ['WESAD', 'DREAMER', 'HCI', etc.]
            win_size (int): Size of the sliding window.
            step (int): Step of the sliding window (default: 1)
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            source (str): Source of the data. Must be one of ['ECG', 'BVP']
        """
        scaler = StandardScaler()
        super().__init__(
            dataset=dataset,
            flag=flag,
            k_split=k_split,
            k=k,
            scaler=scaler,
            source=source,
            confounding=confounding,
            step=step
        )
        self.win_size = win_size

    def __len__(self):
        return len(self.split) - self.win_size + 1

    def __getitem__(self, index):
        x = np.float32(self.split[index:index + self.win_size])
        if self.flag == 'test':
            y = np.float32(self.labels[index:index + self.win_size])
        else:
            y = np.zeros(self.win_size)
        return x, y


class FeaturesHypAD(Features):
    def __init__(self, dataset: str, flag="train", k_split=5, k=0, source="ECG", confounding=False):
        """
        Load the Features dataset for anomaly detection.

        Args:
            dataset (str): Dataset to use. Must be one of ['WESAD', 'DREAMER', 'HCI', etc.]
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            source (str): Source of the data. Must be one of ['ECG', 'BVP']
        """
        scaler = MinMaxScaler(feature_range=(-1, 1))
        super().__init__(
            dataset=dataset,
            flag=flag,
            k_split=k_split,
            k=k,
            scaler=scaler,
            source=source,
            confounding=confounding
        )
        self.intervals = self.labels_intervals()

    def labels_intervals(self) -> pd.DataFrame:
        """
        Create from the labels the intervals of anomalies as [start, end]
        """
        intervals = []
        start = None

        # For each label, check if it is an anomaly
        for i, l in enumerate(self.labels):
            # If it is an anomaly, start the interval
            if l == 1 and start is None:
                start = i
            # If it is not an anomaly and the interval has started, close it
            elif l == 0 and start is not None:
                intervals.append([start, i])
                start = None
        # If the interval has not been closed, close it
        if start is not None:
            intervals.append([start, len(self.labels)])

        return pd.DataFrame(intervals, columns=['start', 'end'])

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        # Get the data
        x = np.float32(self.split[index])
        x = torch.tensor(x)

        if self.flag == 'train':
            return x

        # Get the label
        y = np.float32(self.labels[index])
        y = torch.tensor(y)
        return x, y, [-1], [-1], [-1]


class FeaturesTranAD(Features):
    def __init__(self, dataset: str, flag="train", k_split=5, k=0, source="ECG", confounding=False):
        """
        Load the Features dataset for anomaly detection.

        Args:
            dataset (str): Dataset to use. Must be one of ['WESAD', 'DREAMER', 'HCI', etc.]
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
            source (str): Source of the data. Must be one of ['ECG', 'BVP']
        """
        def normalize(a):
            a = a / np.maximum(np.absolute(a.max(axis=0)),
                               np.absolute(a.min(axis=0)))
            return (a / 2 + 0.5)
        super().__init__(
            dataset=dataset,
            flag=flag,
            k_split=k_split,
            k=k,
            scaler=normalize,
            source=source,
            confounding=confounding
        )
        # Labels from [N] to [N, k] repeating the labels k times
        k = self.split.shape[-1]
        self.labels = self.labels[:, np.newaxis]
        self.labels = np.repeat(self.labels, k, axis=1)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        # Get the data
        x = np.float32(self.split[index])
        return x

if __name__ == "__main__":
    datasets = "HCI"
    scaler = StandardScaler()
    features = FeaturesTranAD(dataset=datasets, flag="train", source="BVP")
    print([features[i] for i in range(5)])
