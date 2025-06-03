import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # nopep8
import sys # nopep8
import torch
from torch.utils.data import DataLoader
import wandb
import random

base_path = os.getenv("BASE_PATH")  # nopep8
sys.path.append(os.path.join(base_path, "src/data"))  # nopep8
sys.path.append(os.path.join(base_path, "src/models/UniTS"))  # nopep8
from src.data.FeaturesDataset import Features, FeaturesUniTS
from exp.exp_sup import Exp_All_Task

# Get parent path
script_path = os.path.dirname(os.path.realpath(__file__))

# Save as environment variable for next scripts
os.environ["BASE_PATH"] = script_path
os.environ["WANDB_USER"] = "your_wandb_username"
WANDB_USER = os.getenv("WANDB_USER", "none")
k=1
FIXED_SEED = 444

def get_model_predictions(dataset_name: str, source: str, majority_threshold: float = 0):
    wandb.init(project="units_analysis", entity=WANDB_USER)
    # Units dataset
    train_dataset = FeaturesUniTS(dataset_name, 5, 1, "train", k_split=5, k=k, source=source)
    test_dataset = FeaturesUniTS(dataset_name, 5, 1, "test", k_split=5, k=k, source=source)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get model
    args = {
        "is_training": 1,
        "model_id": "UniTS_sup_",
        "model": "UniTS",
        "task_name": "anomaly_detection",
        "prompt_num": 10,
        "patch_len": 16,
        "stride": 16,
        "e_layers": 3,
        "d_model": 128,
        "n_heads": 8,
        "des": "Exp",
        "itr": 1,
        "lradj": "finetune_anl",
        "weight_decay": 5e-6,
        "train_epochs": 20,
        "batch_size": 64,
        "acc_it": 32,
        "dropout": 0,
        "debug": "online",
        "dataset_name": dataset_name,
        "project_name": f"{dataset_name}_units_pretrained_d128_kfold",
        "pretrained_weight": "src/models/UniTS/pretrain_checkpoint.pth",
        "clip_grad": 100,
        "task_data_config_path": f"{dataset_name}.yaml",
        "anomaly_ratio": 7,
        "confounding": False,
        "data": "All",
        "features": "M",
        "checkpoints": "src/models/UniTS/checkpoints/",
        "freq": "h",
        "win_size": 5,
        "k": k,
        "source": source,
        "step": 1,
        "subsample_pct": None,
        "num_workers": 0,
        "learning_rate": 1e-4 if source == "ECG" else 5e-5,
        "layer_decay": None,
        "memory_check": False,
        "prompt_tune_epoch": 0,
    }
    
    # Dict to object
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args)
    task_data_config = {'WESAD': {'dataset': 'WESAD', 'data': 'WESAD', 'embed': 'timeF', 'label_len': 0, 'pred_len': 0, 'features': 'M', 'enc_in': 2, 'context_size': 4, 'task_name': 'anomaly_detection', 'max_batch': 64}}
    
    # Load the model
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    exp = Exp_All_Task(args, infer=True, task_data_config=task_data_config)
    
    # Train the model
    setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des, 0)
    exp.train(setting, train_ddp=False)

    # Test the model
    f_score, precision, recall, accuracy, gt, pred, reconstructed = exp.test_anomaly_detection(
        {}, {}, (train_loader, test_loader), "", 0, ar=3
    )
    print(f"Precision: {precision}, Recall: {recall}, F1: {f_score}, Accuracy: {accuracy}")

    # Get test dataset as a numpy array
    data = []
    for x, y in test_loader:
        new_x = x.numpy()
        for i in range(new_x.shape[0]): # For each element in the batch
            new_x[i] = train_dataset.scaler.inverse_transform(new_x[i])
        data.append(x) # (batch_size, w_size, n_features)

    # Get the reconstructed data
    rec = []
    for i in range(len(reconstructed)):
        new_x = train_dataset.scaler.inverse_transform(reconstructed[i])
        rec.append(new_x)

    # Aggregate the data
    data = np.concatenate(data, axis=0) # (n_samples, w_size, n_features)
    rec = np.concatenate(rec, axis=0) # (n_samples, w_size, n_features)

    # Get the prediction by averaging the prediction of the sliding windows
    window_size = 5
    avg_pred = []

    # For each sample in the prediction
    for i in range(0, len(pred), window_size):
        # Get the first predictions for that sample from the sliding windows
        other_pred = []
        min_idx = max(0, i-window_size*window_size) - 1
        for j in range(i, min_idx, -window_size-1):
            other_pred.append(pred[j])
        p = np.mean(other_pred, axis=0)
        p = np.where(p > majority_threshold, 1, 0)
        avg_pred.append(p)

    # Flatten the data
    data = data.reshape(-1, data.shape[-1]) # (n_samples * w_size, n_features)
    rec = rec.reshape(-1, rec.shape[-1]) # (n_samples * w_size, n_features)

    # Get the ground truth and prediction
    avg_pred = np.array(avg_pred) # (n_samples, n_features)
    gt = gt[:-window_size+1:window_size] # (n_samples, n_features)
    data = data[:-window_size+1:window_size] # (n_samples, n_features)
    rec = rec[:-window_size+1:window_size] # (n_samples, n_features) 

    return gt, avg_pred, data, rec


def plot_anomaly(dataset_name: str, source: str, all_samples: bool = False, majority_threshold: float = 0):
    # Get the first anomaly
    if source == "ECG":
        n_samples = 500
        anomaly_idx = 200
    else:
        n_samples = 500
        anomaly_idx = 600

    if all_samples:
        n_samples = 2000
        anomaly_idx = 0

    print(f"Plotting anomaly for {source} from sample {anomaly_idx} to {anomaly_idx+n_samples}")

    # Create the directory to save the plots
    prefix = "all_" if all_samples else "part_"
    prefix += f"{source}_"
    path = os.path.join(base_path, f"plots/{k}")
    path = os.path.join(path, source)
    if majority_threshold > 0:
        path = os.path.join(path, f"majority_{majority_threshold}")
    os.makedirs(path, exist_ok=True)

    # Get the model prediciton
    gt, pred, data, outputs = get_model_predictions(dataset_name, source, majority_threshold)
    min_idx = max(anomaly_idx, 0)
    max_idx = min(min_idx + n_samples, len(gt))
    gt = gt[min_idx:max_idx]
    pred = pred[min_idx:max_idx]
    data = data[min_idx:max_idx]
    outputs = outputs[min_idx:max_idx]

    # Compute error between the data and the output MSE
    error_hr = np.abs(data[:, 0] - outputs[:, 0])
    error_hrv = np.abs(data[:, 1] - outputs[:, 1])
    tot_error = error_hr + error_hrv
    perc_hr = error_hr / tot_error
    perc_hrv = error_hrv / tot_error
    tot_error = tot_error / np.max(tot_error)
    error_hr = error_hr / np.max(error_hr)
    error_hrv = error_hrv / np.max(error_hrv)

    # Mask out the output signal where prediction is 0
    outputs[:, 0] = np.where(pred == 0, np.nan, outputs[:, 0])
    outputs[:, 1] = np.where(pred == 0, np.nan, outputs[:, 1])

    # Plot the data (HR, HRV) in 3 subplots
    plt.figure(figsize=(10, 3))
    x_ticks = np.arange(min_idx, max_idx)

    # Plot the HR data, every thick in x should have a description
    plt.plot(x_ticks, data[:, 0], color="#3499cd")
    plt.ylabel("HR")
    plt.xlabel("Time (s)")
    # Plot the anomaly prediction as red area
    min_h = data[:, 0].min()
    max_h = data[:, 0].max()
    h_margin = 0.1 * (max_h - min_h)
    plt.ylim(min_h - h_margin, max_h + h_margin)
    plt.fill_between(x_ticks, min_h, max_h+h_margin, where=pred == 1, color='red', alpha=0.3, linewidth=0.0)
    plt.fill_between(x_ticks, min_h-h_margin, min_h, where=gt == 1, color='green', alpha=0.3, linewidth=0.0)
    # Save HR plot in svg
    plt.tight_layout()
    plt.savefig(path + f"/{prefix}anomaly_hr.svg")
    plt.close()

    # Plot the HRV data
    plt.figure(figsize=(10, 3))
    plt.plot(x_ticks, data[:, 1], color="#3499cd")
    plt.ylabel("HRV")
    plt.xlabel("Time (s)")
    # Plot the anomaly prediction as red area
    min_h = data[:, 1].min()
    max_h = data[:, 1].max()
    h_margin = 0.1 * (max_h - min_h)
    plt.ylim(min_h - h_margin, max_h + h_margin)
    plt.fill_between(x_ticks, min_h, max_h+h_margin, where=pred == 1, color='red', alpha=0.3, linewidth=0.0)
    plt.fill_between(x_ticks, min_h-h_margin, min_h, where=gt == 1, color='green', alpha=0.3, linewidth=0.0)
    # Save HRV plot in svg
    plt.tight_layout()
    plt.savefig(path + f"/{prefix}anomaly_hrv.svg")
    plt.close()

    # Bar plot of the error
    plt.figure(figsize=(10, 3))
    # Set alpha for each sample based on its value for tot_error
    for i in range(len(tot_error)):
        alpha = min(1, tot_error[i])
        plt.bar(x_ticks[i], perc_hr[i], color="#3499cd", alpha=alpha)
        plt.bar(x_ticks[i], perc_hrv[i], bottom=perc_hr[i], color="#f89939", alpha=alpha)
    plt.ylabel("Contribution Probability")
    plt.xlabel("Time (s)")
    # Save the plot in svg
    plt.tight_layout()
    plt.savefig(path + f"/{prefix}anomaly_error_contrib.svg")
    plt.close()

    # Bar plot of the error
    plt.figure(figsize=(10, 3))
    plt.bar(x_ticks, error_hr, color="#3499cd", alpha=1)
    plt.ylabel("Rec Error HR")
    plt.xlabel("Time (s)")
    # Save the plot in svg
    plt.tight_layout()
    plt.savefig(path + f"/{prefix}anomaly_hr_error.svg")
    plt.close()

    # Bar plot of the error
    plt.figure(figsize=(10, 3))
    plt.bar(x_ticks, error_hrv, color="#3499cd", alpha=1)
    plt.ylabel("Rec Error HRV")
    plt.xlabel("Time (s)")
    # Save the plot in svg
    plt.tight_layout()
    plt.savefig(path + f"/{prefix}anomaly_hrv_error.svg")
    plt.close()


def plot_density(dataset_name: str, source: str):
    # Get the dataset
    dataset = Features(dataset_name, "test", k_split=5, k=k, source=source, scaler=lambda x: x)
    path = os.path.join(base_path, "plots")

    # Get anomaly samples
    anomaly_samples = dataset.split[dataset.labels == 1]
    non_anomaly_samples = dataset.split[dataset.labels == 0]

    # Window size 5
    w_size = 5
    lanomaly_samples = []
    for i in range(anomaly_samples.shape[0] - w_size):
        lanomaly_samples.append(np.mean(anomaly_samples[i:i+w_size], axis=0))
    anomaly_samples = np.array(lanomaly_samples)

    lnon_anomaly_samples = []
    for i in range(non_anomaly_samples.shape[0] - w_size):
        lnon_anomaly_samples.append(np.mean(non_anomaly_samples[i:i+w_size], axis=0))
    non_anomaly_samples = np.array(lnon_anomaly_samples)

    # Set the style of the plots
    plt.rcParams.update({'font.size': 22})
    plt.tight_layout()

    # Plot the density of anomalous and non-anomalous HR
    plt.figure(figsize=(10, 6.5))
    sns.kdeplot(anomaly_samples[:, 0], label="Anomalous", fill=True, color="red")
    sns.kdeplot(non_anomaly_samples[:, 0], label="Non-anomalous", fill=True, color="blue")
    plt.xlabel("HR")
    plt.ylabel("Density")
    plt.legend()

    # Save the plot in pdf
    plt.savefig(path + f"/hr_density_{source}.pdf")

    # Plot the density of anomalous and non-anomalous HRV
    plt.figure(figsize=(10, 6.5))
    plt.xlim(0, max(anomaly_samples[:, 1].max(), non_anomaly_samples[:, 1].max()))
    sns.kdeplot(anomaly_samples[:, 1], label="Anomalous", fill=True, color="red")
    sns.kdeplot(non_anomaly_samples[:, 1], label="Non-anomalous", fill=True, color="blue")
    plt.xlabel("HRV")
    plt.ylabel("Density")
    plt.legend()

    # Save the plot in pdf
    plt.savefig(path + f"/hrv_density_{source}.pdf")


def main():
    dataset_name = "WESAD"
    for source in ["ECG", "BVP"]:
        plot_anomaly(dataset_name, source, all_samples=True)
        plot_anomaly(dataset_name, source, all_samples=False)

    for source in ["ECG", "BVP"]:
        plot_density(dataset_name, source)


if __name__ == "__main__":
    main()