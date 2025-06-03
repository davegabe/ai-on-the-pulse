import pandas as pd
from data.ATHOMEDataset import ATHOMEFeaturesUniTS
from data.PREATHOME.constants import PREATHOME_CSV, PREATHOME_SLEEP_CSV, PREATHOME_STEPS_CSV
from data.utils.anomalies import main_features

class PREATHOMEFeaturesUniTS(ATHOMEFeaturesUniTS):
    def __init__(self, win_size: int, anomaly_type: int, flag="train", k_split=5, k=0):
        """
        Load the Features dataset for anomaly detection.

        Args:
            win_size (int): Size of the sliding window.
            anomaly_type (int): Type of anomaly to generate.
            flag (str): Flag to select the split. Must be one of ['train', 'test']
            k_split (int): Number of splits to perform for k-fold cross-validation (default: 10)
            k (int): Index of the split to use for k-fold cross-validation (default: 0)
        """
        super().__init__(
            win_size=win_size,
            anomaly_type=anomaly_type,
            flag=flag,
            k_split=k_split,
            k=k,
            data_csv=PREATHOME_CSV,
            sleep_csv=PREATHOME_SLEEP_CSV,
            steps_csv=PREATHOME_STEPS_CSV,
        )


if __name__ == "__main__":
    # Test the ATHOMEFeaturesUniTS class
    for anomaly_type in main_features.keys():
        dataset = PREATHOMEFeaturesUniTS(win_size=3, anomaly_type=anomaly_type, flag="test", k=-1)

        print(f"Anomaly type: {anomaly_type}")
        print(f"Number of samples: {len(dataset)}")
        print(f"Number of non-NaN samples: {dataset.non_nan_count}")
        print(f"First date: {dataset.first_date}")
        print(f"Last date: {dataset.last_date}")
        n_days = (pd.to_datetime(dataset.last_date) - pd.to_datetime(dataset.first_date)).days
        print(f"Number of days: {n_days}")
        print(f"Number of days with no data: {len(dataset.no_data_days)}")
        # print(f"First sample:")
        # print(dataset[0])
        # print(f"Last sample:")
        # print(dataset[-1])
        print("#" * 80)
