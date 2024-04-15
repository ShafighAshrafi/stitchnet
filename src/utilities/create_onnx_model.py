import onnx
import onnxoptimizer
from onnx.helper import make_graph,make_model


def create_onnx_model(model, nodes=None, name=None, inputs=None, outputs=None, initializer=None):
    if nodes is None:
        nodes = model.graph.node
    if name is None:
        name = model.graph.name
    if inputs is None:
        inputs = model.graph.input
    if outputs is None:
        outputs = model.graph.output
    if initializer is None:
        initializer = model.graph.initializer
    
    graph = make_graph(nodes, name, inputs,
                       outputs, initializer)
    
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    # fix opset import
    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
        
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    # optimized_model = onnx_model

    return optimized_model