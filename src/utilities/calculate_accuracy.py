import numpy as np
import torch
from tqdm import tqdm

from utilities.dataloader_generator import generate_dataloader


@torch.no_grad()
def calculate_model_accuracy(model, dataset, batch_size=64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    count = 0
    model.eval()
    model.to(device)
    for x, label in tqdm(
        generate_dataloader(dataset, batch_size=batch_size, shuffle=False),
        position=0,
        leave=True,
    ):
        x = x.to(device)
        y = model(x)
        y = y.cpu()
        y = np.argmax(y.detach().numpy(), 1)
        print(x.shape, label, y)
        return
        count += np.sum(y == label.numpy())
    accuracy = 1.0 * count / len(dataset)
    return accuracy
