import torch


def pt_R2(X, Y):
    # QR decomposition of Y
    Q, R = torch.linalg.qr(Y)

    # Compute the Frobenius norm of Q^T * X
    # print('Q', Q.shape)
    # print('X', X.shape)
    # print('Q^T * X', torch.matmul(Q.T, X).shape)
    numerator = torch.norm(torch.matmul(Q.T, X), p='fro') ** 2
    # Compute the Frobenius norm of X
    denominator = torch.norm(X, p='fro') ** 2
    # print('numerator', numerator.item(), 'denominator', denominator.item(), 'numerator / denominator', numerator.item() / denominator.item())

    # Compute and return R^2_LR
    r2 = (numerator / denominator)
    # if r2 >= 1:
    #     print('X',X)
    #     print('Y',Y)
    # print('r2', r2.item(), numerator.item(), denominator.item())
    return r2