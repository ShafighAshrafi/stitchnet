import torch
from src.utilities.adjust_weights.train_w import train_w
from src.utilities.get_device import Device

def adjust_w_linear(tX, tY, w):
    device = Device().get_device()
    if tX.ndim == 2:
        acts1 = tX
        acts2 = tY
    else:
        pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flat = torch.nn.Flatten()
        tX = flat(pool(tX))

        acts1 = tX.reshape(tX.shape[0], -1)
        acts2 = tY
    
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    Ainit = acts2.T @ acts1.T.pinverse()
    # print(acts1.shape)
    # print(acts2.shape)
    # print('linear')
    A = train_w(device, acts1, acts2, Ainit)
    A = A.to(device)
    
    tw = torch.from_numpy(w).to(device)
    nw = torch.einsum('ij, jk -> ik', tw, A)
    nw = nw.cpu().numpy()
    return nw