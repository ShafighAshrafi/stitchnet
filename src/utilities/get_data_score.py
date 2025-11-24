import random

import numpy as np
import torch

from utilities.dataloader_generator import generate_dataloader
from utilities.load_dataset_chest_xray import load_dataset


def get_data_score(batch_size=32, includeTarget=False):
    random.seed(51)
    np.random.seed(24)
    torch.manual_seed(77)

    dataset_train = load_dataset()

    dl_score = generate_dataloader(dataset_train, batch_size=batch_size)
    data_score, t = next(iter(dl_score))
    if includeTarget:
        return data_score, t
    data_score = data_score.numpy()
    return data_score
