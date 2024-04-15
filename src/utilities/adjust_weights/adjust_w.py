from src.utilities.adjust_weights.adjust_w_conv import adjust_w_conv
from src.utilities.adjust_weights.adjust_w_linear import adjust_w_linear


def adjust_w(tX, tY, w):
    if w.ndim == 2:
        return adjust_w_linear(tX, tY, w)
    else:
        # linear to conv, unsupport
        if tX.ndim == 2 and tY.ndim == 4:
            raise Exception("unsupport linear to conv stitching")
        # print('tX.shape', tX.shape, 'tY.shape', tY.shape)
        return adjust_w_conv(tX, tY, w)