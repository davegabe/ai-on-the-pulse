#!usr/bin/bash python
# coding: utf-8

import argparse
import random

import yaml
from torch.utils.data import DataLoader

import anomaly_detection
import utils.data as od
from hyperspace.utils import *
from train import train
import wandb
import os

VERSION = os.getenv("VERSION", "none")
WANDB_USER = os.getenv("WANDB_USER", "none")
FIXED_SEED = 444
BVP_DATASETS = ["WESAD"]

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="HypAD")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        default="/your_default_config_file_path",
    )

    params = parser.parse_args()
    config_path = params.config
    params = yaml.load(open(params.config), Loader=yaml.FullLoader)
    params = argparse.Namespace(**params)
    dataset = params.dataset

    # For both hypad and tadgan
    for hyperbolic in [True, False]:
        # For each learning rate
        for lr in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
            # For each source signal (ECG, BVP)
            for source in ['ECG', 'BVP']:
                # For each fold of the dataset, train and test the model
                for k in range(5):
                    # Fix the seed
                    random.seed(FIXED_SEED)
                    torch.manual_seed(FIXED_SEED)
                    np.random.seed(FIXED_SEED)

                    # If the source signal is BVP and the dataset has no BVP signal, skip
                    if source == 'BVP' and dataset not in BVP_DATASETS:
                        break

                    params.k = k # Set fold number
                    params.hyperbolic = hyperbolic # Set hyperbolic flag
                    params.lr = lr # Set learning rate
                    params.source = source # Set source signal

                    # Initialize wandb
                    name = ""
                    tags = [f"lr{lr}", source]
                    # Add tags and name suffixes
                    if params.hyperbolic:
                        name += f"hypad-lr_{lr}-hyperbolic"
                        model_name = "hypad"
                        tags.append("hypad")
                    else:
                        name += f"tadgan-lr_{lr}"
                        model_name = "tadgan"
                        tags.append("tadgan")
                    if params.fixed_threshold:
                        name += "-ft"
                        tags.append("fixed-threshold")

                    # Add the source signal to the name
                    name += f"-{source}"

                    # Add the name with suffixes (but not the fold number) to the tags
                    group = name
                    tags.append(group)

                    # Add the fold number to the name
                    name += f"-fold_{k}"
                    project_name = f"{dataset.lower()}-{model_name}-kfold_{VERSION}"

                    # Initialize wandb
                    wandb.init(
                        project=project_name,
                        entity=WANDB_USER,
                        config=params,
                        name=name,
                        tags=tags,
                        group=group
                    )

                    # Select the dataset
                    train_dataset, test_dataset, read_path = od.dataset_selection(params)

                    # Create the data loaders
                    batch_size = params.batch_size
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        drop_last=True,
                        shuffle=True,
                        num_workers=2,
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        drop_last=False,
                        shuffle=False,
                        num_workers=2,
                    )

                    # Train the model
                    encoder, decoder, critic_x, critic_z, path = train(
                        train_loader,
                        test_loader,
                        params,
                        config_path,
                        read_path
                    )

                    # Close wandb
                    wandb.finish()
                    
