from src.utilities.adjust_weights.adjust_w_conv import adjust_conv_weight
from src.utilities.adjust_weights.adjust_w_linear import adjust_linear_weight


def adjust_weight(tensor_x, tensor_y, weight):
    if weight.ndim == 2:
        return adjust_linear_weight(tensor_x, tensor_y, weight)
    else:
        # linear to conv, unsupport
        if tensor_x.ndim == 2 and tensor_y.ndim == 4:
            raise Exception("unsupport linear to conv stitching")
        # print('tX.shape', tX.shape, 'tY.shape', tY.shape)
        return adjust_conv_weight(tensor_x, tensor_y, weight)