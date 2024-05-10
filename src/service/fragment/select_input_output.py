
import onnxoptimizer
from onnx import ValueInfoProto
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from onnx.helper import make_tensor_value_info, make_graph, make_model, set_model_props


def select_model_inputs_outputs2(model, outputs=None, inputs=None, inputs_types=None, inputs_shapes=None, name=None):
    """
    Takes a model and changes its outputs.

    :param model: *ONNX* model
    :param inputs: new inputs
    :param outputs: new outputs
    :return: modified model

    The function removes unneeded files.
    """
    # if inputs is not None:
    #     raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise RuntimeError("Parameter outputs cannot be None.")
    if not isinstance(outputs, list):
        outputs = [outputs]
    if inputs is not None and not isinstance(inputs, list):
        inputs = [inputs]
    if inputs_types is None:
        inputs_types = []
    elif not isinstance(inputs_types, list):
        inputs_types = [inputs_types]
    if inputs_shapes is None:
        inputs_shapes = []
    elif not isinstance(inputs_shapes, list):
        inputs_shapes = [inputs_shapes]
        
    mark_nodes = {}
    for out in enumerate_model_node_outputs(model):
        mark_nodes[out] = 0
    for inp in model.graph.input:
        mark_nodes[inp.name] = 0
    for out in outputs:
        if out not in mark_nodes:
            raise ValueError(f"Output '{out}' not found in model.")
        mark_nodes[out] = 1
    
    mark_input_nodes = {}
    if inputs is not None:
        for inp in inputs:
            mark_input_nodes[inp] = 1
        
    nodes = model.graph.node[::-1]
    mark_operation = {}
    for node in nodes:
        mark_operation[node.name] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            # decide whether to include this node
            if mark_operation[node.name] == 1:
                continue
            mod = False
            # if output of this node is in the graph, include the node and check its input as well
            for out in node.output:
                if mark_nodes[out] == 1:
                    mark_operation[node.name] = 1
                    mod = True
                    break
            if not mod:
                continue

            nb += 1
            for inp in node.input:
                # stop
                if mark_input_nodes.get(inp, 0) == 1:
                    mark_operation[node.name] = 1
                    continue
                if mark_nodes.get(inp, 0) == 1:
                    continue
                mark_nodes[inp] = 1
                nb += 1
        
    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes if mark_operation[node.name] == 1]
    keep_nodes = keep_nodes[::-1]

    var_out = []
    for out in outputs:
        value_info = ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
            
    if inputs is None:
        var_int = model.graph.input
    else:
        var_int = []
        for i, inp in enumerate(inputs):
            ttype = inputs_types[i]
            tshape = inputs_shapes[i]
            var_int.append(make_tensor_value_info(inp, 
                                  ttype, tshape))
    if name is None:
        name = model.graph.name
    graph = make_graph(keep_nodes, name, var_int,
                       var_out, model.graph.initializer)
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)

    # if len(onnx_model.graph.input) != len(model.graph.input):
    #     raise RuntimeError("Input mismatch {} != {}".format(
    #         len(onnx_model.input), len(model.input)))

    # fix opset import
    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
        
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    passes = ["eliminate_unused_initializer"]
    # print('inputname', [n.name for n in onnx_model.graph.input], 'inputs', inputs)
    # list_ops(onnx_model)
    # onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    # save_onnx_model(onnx_model, 'tmp.onnx')
    # onnx_model = load_onnx_model('tmp.onnx')
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    # except Exception as e:
    #     from skl2onnx.helpers.onnx_helper import save_onnx_model
    #     import string
    #     import random
    #     print("outputs: ", outputs)
    #     print("inputs: ", inputs)
    #     save_onnx_model(onnx_model, f"../test/corupted_model_{''.join(random.choice(string.ascii_uppercase) for _ in range(5))}.onnx")
    #     raise Exception("error")

    return optimized_model