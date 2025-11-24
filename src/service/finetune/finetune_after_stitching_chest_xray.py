from collections import Counter

import torch
import torch.nn as nn

from service.finetune.create_optimizer import create_optimizer
from service.finetune.train_model import train_model
from utilities.dataloader_generator import generate_dataloader
from utilities.load_dataset_chest_xray import load_dataset
from utilities.set_requires_grad_parameters import set_parameter_requires_grad


def finetune_after_stitching(
    model,
    num_classes=2,
    num_epochs=3,
    batch_size=64,
    val_batch_size=128,
    feature_extracting=False,
):
    """
    Fine-tune stitched model for chest X-ray pneumonia dataset (2 classes: NORMAL, PNEUMONIA).

    Args:
        model: PyTorch model (stitched network)
        num_classes: Number of classes (2 for chest X-ray: NORMAL, PNEUMONIA)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        val_batch_size: Validation batch size
        feature_extracting: If True, only train the last layer

    Returns:
        Training history (train_acc_history, train_loss_history, final_accuracy)
    """
    if not isinstance(model, torch.nn.Module):
        raise Exception("The model is not an instance of PyTorch.")

    set_parameter_requires_grad(model, feature_extracting)
    module_name, module = list(model.named_modules())[-1]
    setattr(
        model,
        module_name,
        nn.Linear(module.in_features, num_classes, bias=module.bias is not None),
    )

    dataloaders_dict = dict(
        train=generate_dataloader(load_dataset(), batch_size=batch_size),
        validation=generate_dataloader(
            load_dataset(folder="test"), batch_size=val_batch_size
        ),
    )
    optimizer = create_optimizer(model)
    dataset_info = dict(Counter(load_dataset().targets))
    total_train_data = sum(dataset_info.values())

    # Setup the loss func and weight list
    # forced to use the weight parameter because the dataset was imbalanced
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [total_train_data / count for count in dataset_info.values()]
        )
    ).cuda()

    # Train and evaluate
    return train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        is_inception=False,
    )
