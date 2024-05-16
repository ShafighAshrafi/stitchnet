import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.utilities.get_device import Device


def eval_original_model(model, dataloaders):
    device = Device().get_device()
    model.to(device)
    model.eval()
    
    epoch_acc = defaultdict(int)
    with torch.no_grad():
        for phase in ['train', 'val']:
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], position=0, leave=True):
                inputs = inputs.to(device)
                labels = labels.to(device)
                                
                outputs = model(inputs)
                _, y = torch.max(outputs, 1)
                running_corrects += np.sum(y == labels.data)

                # cat 0, dog 1
                # preds = torch.where(reduce(torch.bitwise_or, [preds==y for y in catIds]), 0, -1)
                # preds[reduce(torch.bitwise_or, [preds==y for y in dogIds])] = 1
                
                # running_corrects += np.sum(y == t.numpy())
            
            epoch_acc[phase] = running_corrects.double() / len(dataloaders[phase].dataset)
    
    return epoch_acc['val'], epoch_acc['train']


def evaluate_validation(model, dataloader):
    device = Device().get_device()
    model.to(device)
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double() / len(dataloader.dataset)