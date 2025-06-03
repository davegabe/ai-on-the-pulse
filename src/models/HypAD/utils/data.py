# -*- coding: utf-8 -*-
import logging
import os

# Import dataloader from main directory
import sys  # nopep8
import os  # nopep8
base_path = os.getenv("BASE_PATH")  # nopep8
sys.path.append(os.path.join(base_path, "src/data"))  # nopep8
from FeaturesDataset import FeaturesHypAD  # nopep8

LOGGER = logging.getLogger(__name__)


def dataset_selection(params):
    """
    DATASET SELECTION
    """
    read_path = ""
    train_dataset = FeaturesHypAD(
        dataset=params.dataset,
        flag="train",
        k=params.k,
        source=params.source
    )
    test_dataset = FeaturesHypAD(
        dataset=params.dataset,
        flag="test",
        k=params.k,
        source=params.source
    )
    return train_dataset, test_dataset, read_path  # type: ignore
