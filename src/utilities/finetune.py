import time
import copy
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from collections import Counter

from src.utilities.dataloader_generator import generate_dataloader
from src.utilities.load_dataset import load_dataset


def set_parameter_requires_grad(model, feature_extracting=False):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def evaluate_validation(device, model, dataloader):
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double() / len(dataloader.dataset)
    

def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    train_acc_history = []
    train_loss_history = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        running_corrects = 0.0
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Iterate over data.
        phase = "train"
        for inputs, labels in tqdm(dataloaders[phase], position=0, leave=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                if is_inception:
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, auxillary_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(auxillary_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.cpu().numpy() / len(dataloaders[phase].dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        print('Loss: {:.4f} Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))

    model.eval()
    final_accuracy = evaluate_validation(device, model, dataloaders['validation'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Final Accuracy is: {:4f}'.format(final_accuracy))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return train_acc_history, train_loss_history


def create_optimizer(model_to_fit, feature_extract=False):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_to_fit.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_to_fit.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_to_fit.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer

def finetune(model, num_classes=3, num_epochs=3, batch_size=64, val_batch_size=128, feature_extract=True):
    # The given model shoud be of type pytorch
    if not isinstance(model, torch.nn.Module):
        raise ("The model is an instance of PyTorch.")
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
        print(module_name, module)
    dataloaders_dict = dict(
        train=generate_dataloader(load_dataset(), batch_size=batch_size),
        validation=generate_dataloader(load_dataset(
            folder="test"), batch_size=val_batch_size)
    )
    optimizer = create_optimizer(model)
    dataset_info = dict(Counter(load_dataset().targets))
    total_train_data = sum(dataset_info.values())
    # Setup the loss func and weight list 
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([total_train_data / count for count in dataset_info.values()])).cuda()
    # Train and evaluate
    return train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=False)
