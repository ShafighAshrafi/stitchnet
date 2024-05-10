import torch
import torch.nn as nn
from collections import Counter

from src.utilities.load_dataset import load_dataset
from src.service.finetune.train_model import train_model
from src.service.finetune.create_optimizer import create_optimizer
from src.utilities.dataloader_generator import generate_dataloader
from src.utilities.set_requires_grad_parameters import set_parameter_requires_grad


def finetune(model, num_classes=3, num_epochs=3, batch_size=64, val_batch_size=128, feature_extract=True):
    # The given model shoud be of type pytorch
    if not isinstance(model, torch.nn.Module):
        raise Exception("The model is not an instance of PyTorch.")
    set_parameter_requires_grad(model, feature_extract)
    module_name, module = list(model.named_modules())[-1]
    if module_name == "fc":
        model.fc = nn.Linear(module.in_features, num_classes, bias=module.bias is not None)
    else:
        module_name, module = list(model.classifier.named_modules())[-1]
        if module_name == "": # for densenet model
            model.classifier = nn.Linear(module.in_features, num_classes, bias=module.bias is not None)
        else:
            setattr(model.classifier, module_name, nn.Linear(module.in_features,
                    num_classes, bias=module.bias is not None))
        # print(module_name, module)
    dataloaders_dict = dict(
        train=generate_dataloader(load_dataset(), batch_size=batch_size),
        validation=generate_dataloader(load_dataset(
            folder="test"), batch_size=val_batch_size)
    )
    optimizer = create_optimizer(model)
    dataset_info = dict(Counter(load_dataset().targets))
    total_train_data = sum(dataset_info.values())
    # Setup the loss func and weight list
    # forced to use the weight parameter because the dataset was imbalanced
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([total_train_data / count for count in dataset_info.values()])).cuda()
    # Train and evaluate
    return train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=False)