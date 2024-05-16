from onnx_tool.graph import Graph
from onnx_tool.utils import ModelConfig
from onnx_tool import create_ndarray_f32


def graph_profile(onnxgraph, dynamic_shapes=None):
    mcfg = ModelConfig({'verbose':True,"constant_folding":False})
    graph = Graph(onnxgraph, mcfg)
    graph.shape_infer(dynamic_shapes)
    graph.profile()
    return graph.macs[0], graph.params


def get_macs_params(fragment, inputName=None, inputSize=(1, 3, 224, 224)):
    inputs= {}
    if inputName is None:
        inputName = fragment.fragment.graph.input[0].name
    inputs[inputName] = create_ndarray_f32(inputSize)
    # change_input_dim(fragment.fragment,1)
    macs,params=graph_profile(fragment.fragment.graph, inputs)
    return macs, params


def get_macs_params_onnx(onnx_model, inputName=None, inputSize=(1, 3, 224, 224)):
    inputs= {}
    if inputName is None:
        inputName = onnx_model.graph.input[0].name
    inputs[inputName] = create_ndarray_f32(inputSize)
    # change_input_dim(fragment.fragment,1)
    macs, params=graph_profile(onnx_model.graph, inputs)
    return macs, params