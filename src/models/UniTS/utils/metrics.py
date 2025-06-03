import numpy as np
import torch
from torcheval.metrics.functional import binary_auprc


def multilabel_fpr_fnr(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    # Initialize variables to store FPR and FNR for each label
    fpr_per_label = []
    fnr_per_label = []

    # Iterate over each label
    for label_idx in range(predictions.shape[1]):
        label_predictions = predictions[:, label_idx]
        label_targets = targets[:, label_idx]

        # Convert to tensor
        label_predictions = torch.tensor(label_predictions).int()
        label_targets = torch.tensor(label_targets).int()

        # FPR: predicted 1 (stress) but true label is 0 (no stress)
        fpr = torch.logical_and(label_predictions == 1, label_targets == 0)
        fpr = torch.mean(fpr.float())
        # FNR: predicted 0 (no stress) but true label is 1 (stress)
        fnr = torch.logical_and(label_predictions == 0, label_targets == 1)
        fnr = torch.mean(fnr.float())

        # Append FPR and FNR to the list
        fpr_per_label.append(fpr)
        fnr_per_label.append(fnr)

    # Compute average FPR and FNR across all labels
    avg_fpr = torch.mean(torch.tensor(fpr_per_label))
    avg_fnr = torch.mean(torch.tensor(fnr_per_label))

    # Return average FPR and FNR
    return avg_fpr.item(), avg_fnr.item()


def multilabel_auprc(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    # Initialize variables to store AUCPR for each label
    aucpr_per_label = []

    # Iterate over each label
    for label_idx in range(predictions.shape[1]):
        label_predictions = predictions[:, label_idx]
        label_targets = targets[:, label_idx]

        # Convert to tensor
        label_predictions = torch.tensor(label_predictions)
        label_targets = torch.tensor(label_targets)

        # Compute AUCPR for the current label
        aucpr = binary_auprc(label_predictions, label_targets)
        aucpr_per_label.append(aucpr)

    # Compute average AUCPR across all labels
    avg_aucpr = torch.mean(torch.tensor(aucpr_per_label))

    # Return average AUCPR
    return avg_aucpr.item()


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 *
                (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
