import onnx.numpy_helper as numpy_helper


def to_tensor_proto(array, name=None):
    tensor_proto = numpy_helper.from_array(array, name=name)
    return tensor_proto