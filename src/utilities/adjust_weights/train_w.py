
import torch
from torch.autograd import Variable


def train_w(device, acts1, acts2, Winit, nepoch=1, batch_size=1024, learning_rate=1e-6, momentum=0.9):
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    Winit = Winit.to(device)
    W = Variable(Winit, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=learning_rate, momentum=momentum)
    dset = torch.utils.data.TensorDataset(acts1,acts2)
    prev_loss = None
    for _ in range(nepoch):
        running_loss = 0
        for x,y in torch.utils.data.DataLoader(dset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0):
            optimizer.zero_grad()
            y_pred = x.matmul(W.T)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        epoch_loss = running_loss/len(dset)
        # early stopping
        # print(f'epoch {epoch} loss', epoch_loss, acts1.shape, acts2.shape)
        if prev_loss is not None and prev_loss <= epoch_loss:
            break
        prev_loss = epoch_loss        
    return W.cpu().detach()