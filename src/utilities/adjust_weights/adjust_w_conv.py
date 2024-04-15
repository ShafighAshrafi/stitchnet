import torch
from src.utilities.get_device import Device
from src.utilities.sample_index import sample_index
from src.utilities.adjust_weights.train_w import train_w


def adjust_w_conv(tX, tY, w):
    device = Device().get_device()
    if tY.shape[-2]!=tX.shape[-2] and tY.shape[-1]!=tX.shape[-1]:
        up = torch.nn.UpsamplingBilinear2d((tY.shape[-2],tY.shape[-1]))
        tX = up(tX)
        # addLayers += [up]

    acts1 = tX.permute((0,2,3,1)).reshape(-1,tX.shape[1])
    acts2 = tY.permute((0,2,3,1)).reshape(-1,tY.shape[1])

    indexX = sample_index(acts1, acts2, nsample=min(acts1.shape[0], acts2.shape[0])*10)
    acts1sampled = acts1[indexX,:]
    acts2sampled = acts2[indexX,:]

    # print('diff sampled', (acts1sampled - acts2sampled).pow(2).sum())
    
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    
    Ainit = acts2sampled.T @ acts1sampled.T.pinverse()
    # print(Ainit)
    # print(acts2.shape)
    A = train_w(device, acts1, acts2, Ainit)
    A = A.to(device)
    
    tw = torch.from_numpy(w).to(device)
    nw = torch.einsum('ijkl, jn -> inkl', tw, A)
    nw = nw.cpu().numpy()
    return nw