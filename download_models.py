# %%
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from src.utilities.load_dataset import load_dataset
from src.utilities.calculate_accuracy import calculate_model_accuracy

os.makedirs("../_models", exist_ok=True)

modelpaths = [
    ("resnet50", "../_models/resnet50.onnx"),
    ("alexnet", "../_models/alexnet.onnx"),
    ("densenet121", "../_models/densenet121.onnx"),
    ("mobilenet_v3_small", "../_models/mobilenet_v3_small.onnx"),
    ("vgg16", "../_models/vgg16.onnx"),
]
for modelpath in modelpaths:
    if not os.path.exists(modelpath[1]):
        try:
            model = torch.hub.load("pytorch/vision:v0.13.1", modelpath[0], weights="IMAGENET1K_V2")
        except (ValueError, KeyError):
            model = torch.hub.load("pytorch/vision:v0.13.1", modelpath[0], weights="IMAGENET1K_V1")
        model.eval()
        torch.onnx.export(model, torch.ones(1,3,224,224), modelpath[1], verbose=True)

# %%

# evaluate accuracy of the original networks before finetuning
batch_size = 256
validation_dataset = load_dataset("test")

for modelpath in modelpaths:
    try:
        model = torch.hub.load("pytorch/vision:v0.13.1", modelpath[0], weights="IMAGENET1K_V2")
    except (ValueError, KeyError):
        model = torch.hub.load("pytorch/vision:v0.13.1", modelpath[0], weights="IMAGENET1K_V1")
    print(model)
    continue
    for params in model.parameters():
        params.requires_grad = False
    model.fc = nn.Sequential(OrderedDict([('fc', nn.Linear(model.fc.in_features, 3))]))
    model.eval()
    
    acc = calculate_model_accuracy(model, validation_dataset, batch_size)
    print(acc, modelpath[0])
# %%
 