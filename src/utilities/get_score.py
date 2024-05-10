import torch
from src.utilities.get_device import Device
from src.utilities.scoring_functions.pt_R2 import pt_R2
from src.utilities.scoring_functions.CKA import linear_CKA


@torch.no_grad()
def get_score(X, Y, num_samples=1000, scoring_method='CKA'):
    device = Device().get_device()
    if scoring_method == 'R2':
        score_function = pt_R2
    if scoring_method == 'CKA':
        score_function = linear_CKA

    # X = torch.from_numpy(X)
    # Y = torch.from_numpy(Y)
    # print('X1', X.shape)
    # print('Y1', Y.shape)
    X = X.to(device)
    Y = Y.to(device)
    if X.ndim == 4 and Y.ndim == 4:
        if X.shape[2] > Y.shape[2]:
            up = torch.nn.UpsamplingBilinear2d((Y.shape[-2], Y.shape[-1]))
            X = up(X)
        else:
            up = torch.nn.UpsamplingBilinear2d((X.shape[-2], X.shape[-1]))
            Y = up(Y)
        # print('X2', X.shape)
        # print('Y2', Y.shape)
        X = X.permute((0, 2, 3, 1)).reshape(-1, X.shape[1])
        Y = Y.permute((0, 2, 3, 1)).reshape(-1, Y.shape[1])
        # only sampling 10 times the channels
        num_samples = min(num_samples, X.shape[0], Y.shape[0])
        # print(num_samples, X.shape, Y.shape)
        p = torch.ones(X.shape[0])
        indexX = p.multinomial(num_samples=num_samples)
        # p = torch.ones(Y.shape[0])
        # indexY = p.multinomial(num_samples=num_samples)
        # print(X[indexX,:].shape, Y[indexY,:].shape)
        s = score_function(X[indexX, :], Y[indexX, :]).cpu().item()
    elif X.ndim == 4 and Y.ndim == 2:
        pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flat = torch.nn.Flatten()
        X = flat(pool(X))
        b = X.shape[0]
        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)
        num_samples = min(num_samples, X.shape[1], Y.shape[1])
        p = torch.ones(X.shape[1])
        indexX = p.multinomial(num_samples=num_samples)
        p = torch.ones(Y.shape[1])
        indexY = p.multinomial(num_samples=num_samples)
        X = X[:, indexX]
        Y = Y[:, indexY]
        s = score_function(X, Y).cpu().item()
    else:
        # TODO: check b > 2*min(X.shape[1],Y.shape[1])
        b = X.shape[0]
        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)
        num_samples = min(num_samples, X.shape[1], Y.shape[1])
        p = torch.ones(X.shape[1])
        indexX = p.multinomial(num_samples=num_samples)
        p = torch.ones(Y.shape[1])
        indexY = p.multinomial(num_samples=num_samples)
        X = X[:, indexX]
        Y = Y[:, indexY]
        s = score_function(X, Y).cpu().item()

    # CKA = linear_CKA(X, Y).cpu().item()
    # CKA = linear_CKA(X.cuda(), Y.cuda()).cpu().item()
    return s
