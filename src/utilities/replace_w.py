from src.utilities.to_tensor import to_tensor_proto
from src.utilities.create_onnx_model import create_onnx_model


def replace_w(model_onnx, name, w):
    initializer = [] 
    for init in model_onnx.graph.initializer:
        if init.name == name:
            initializer.append(to_tensor_proto(w, name=name))
        else:
            initializer.append(init)        
    return create_onnx_model(model_onnx, initializer=initializer)