from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tqdm import tqdm

from utilities.change_model_layers_dimension import change_layers_dimension
from utilities.fix_reshape_batch_dimension import fix_reshape_batch_dimension
from utilities.get_device import Device
from utilities.providers import PROVIDERS


def eval_original_model(model, dataloaders):
    device = Device().get_device()
    model.to(device)
    model.eval()

    epoch_acc = defaultdict(int)
    with torch.no_grad():
        for phase in ["train", "val"]:
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

            epoch_acc[phase] = running_corrects.double() / len(
                dataloaders[phase].dataset
            )

    return epoch_acc["val"], epoch_acc["train"]


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


def eval_original_model_onnx(model_onnx, dataloaders):
    """
    Evaluate an ONNX model using ONNX Runtime.

    Args:
        model_onnx: ONNX model (ModelProto)
        dataloaders: Dictionary with 'train' and 'val' dataloaders

    Returns:
        Tuple of (val_accuracy, train_accuracy)
    """
    # Make a copy of the model to avoid modifying the original
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(model_onnx)

    # Make batch dimension dynamic to support different batch sizes
    change_layers_dimension(model_copy, dimension="N")
    model_copy = fix_reshape_batch_dimension(model_copy)

    # Create ONNX Runtime inference session
    ort_session = ort.InferenceSession(
        model_copy.SerializeToString(), providers=PROVIDERS
    )

    # Get input name
    input_name = ort_session.get_inputs()[0].name

    epoch_acc = defaultdict(float)

    for phase in ["train", "val"]:
        running_corrects = 0
        for inputs, labels in tqdm(dataloaders[phase], position=0, leave=True):
            # Convert to numpy if needed
            if isinstance(inputs, torch.Tensor):
                data = inputs.numpy()
            else:
                data = inputs

            # Run inference
            outputs = ort_session.run(None, {input_name: data})
            y = np.argmax(outputs[0], axis=1)

            # Convert labels to numpy if needed
            if isinstance(labels, torch.Tensor):
                labels_np = labels.numpy()
            else:
                labels_np = labels

            running_corrects += np.sum(y == labels_np)

        epoch_acc[phase] = running_corrects / len(dataloaders[phase].dataset)

    return epoch_acc["val"], epoch_acc["train"]
