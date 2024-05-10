import time
import torch
from tqdm import tqdm
from src.utilities.evaluate_model import evaluate_validation


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
    final_accuracy = evaluate_validation(model, dataloaders['validation'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Final Accuracy is: {:4f}'.format(final_accuracy))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return train_acc_history, train_loss_history, final_accuracy