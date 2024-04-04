#%%
import os
import torch
from tqdm import tqdm

from src.utilities.finetune import finetune


os.makedirs('./_results/finetune', exist_ok=True)
model_names = [
    "resnet50",
    "alexnet",
    "densenet121",
    "mobilenet_v3_small",
    "vgg16",
]
for index, model_name in tqdm(enumerate(model_names)):
    os.makedirs(f'_results/finetune/{model_name}', exist_ok=True)
    for run in range(10):
        try:
            model = torch.hub.load("pytorch/vision:v0.13.1", model_name, weights="IMAGENET1K_V2")
        except (ValueError, KeyError):
            model = torch.hub.load("pytorch/vision:v0.13.1", model_name, weights="IMAGENET1K_V1")
        num_samples, iter_val_acc_history, val_acc_history, val_loss_history, train_acc_history, train_loss_history  = finetune(model)
        with open(f'./_results/finetune/{model_name}/{run:03}.txt', 'w') as f:
            for n,acc in zip(num_samples, iter_val_acc_history):
                f.write(f'{n},{acc}\n')
        # torch.onnx.export(model_ft, torch.ones(1,3,224,224), f'_results/finetune/{namewithoutext}/{run:03}_ft.onnx')
# %%
