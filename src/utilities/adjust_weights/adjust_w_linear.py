import torch
from src.utilities.get_device import Device
from src.utilities.adjust_weights.train_w import train_w

def adjust_linear_weight(tensor_x, tensor_y, weight):
    device = Device().get_device()
    if tensor_x.ndim == 2:
        acts1 = tensor_x
        acts2 = tensor_y
    else:
        pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flat = torch.nn.Flatten()
        tensor_x = flat(pool(tensor_x))

        acts1 = tensor_x.reshape(tensor_x.shape[0], -1)
        acts2 = tensor_y
    
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    Ainit = acts2.T @ acts1.T.pinverse()
    # print(acts1.shape)
    # print(acts2.shape)
    # print('linear')
    A = train_w(device, acts1, acts2, Ainit)
    A = A.to(device)
    
    tw = torch.from_numpy(weight).to(device)
    nw = torch.einsum('ij, jk -> ik', tw, A)
    nw = nw.cpu().numpy()
    return nw