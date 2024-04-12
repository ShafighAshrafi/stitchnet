from onnx import helper


def find_conv_inputs(model_onnx):
    inputs = []
    for n in model_onnx.graph.node:
        if n.op_type == 'Conv':
            groups = [helper.get_attribute_value(att) for att in n.attribute if att.name == 'group' and helper.get_attribute_value(att) > 1]
            if len(groups) == 0:
                inputs.append(n.input[0])
        elif n.op_type == 'Gemm':
            inputs.append(n.input[0])
    return inputs