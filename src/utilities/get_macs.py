from onnx_tool.graph import Graph
from onnx_tool.utils import ModelConfig
from onnx_tool import create_ndarray_f32
import traceback


def graph_profile(onnxgraph, dynamic_shapes=None):
    try:
        mcfg = ModelConfig({'verbose':True,"constant_folding":False})
        g = Graph(onnxgraph, mcfg)
        g.shape_infer(dynamic_shapes)
        g.profile()
    except Exception as e:
        traceback.print_exc()
    return g.macs,g.params


def get_macs_params(fragment, inputName=None, inputSize=(1, 3, 224, 224)):
    inputs= {}
    if inputName is None:
        inputName = fragment.fragment.graph.input[0].name
    inputs[inputName] = create_ndarray_f32(inputSize)
    # change_input_dim(fragment.fragment,1)
    macs,params=graph_profile(fragment.fragment.graph, inputs)
    return macs, params