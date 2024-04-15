import torch


def sample_index(X,Y, nsample=1000):
    num_samples = min(min(X.shape[0], Y.shape[0]), nsample)
    # print(X.shape, Y.shape)
    p = torch.ones(X.shape[0])
    indexX = p.multinomial(num_samples=num_samples)
    return indexX