import operator
from functools import reduce

from service.fragment.find_conv_layers import find_conv_inputs
from service.fragment.split_model import split_model_at
from utilities.change_model_layers_dimension import change_layers_dimension
from utilities.change_model_name import change_model_name
from utilities.fix_reshape_batch_dimension import fix_reshape_batch_dimension
from utilities.get_input_nodes import get_input_nodes


# TODO: move the functionality in the loop to new function
# TODO: find out what the reduce line does
def get_fragments(model, input):
    """chop up onnx models into fragments at conv and linear(gemm) layers"""
    inputs = find_conv_inputs(model)
    inputs = inputs[1:]
    fragments = []
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input = [input]
    for i, inp in enumerate(inputs):
        # print(i, inp)
        try:
            first_part_model, remaining_model, first_part_output = split_model_at(
                model, inp, input[0]
            )
            # print('len(get_input_nodes(newf1))', len(get_input_nodes(remaining_model)), [n.name for n in get_input_nodes(remaining_model)])
            if len(first_part_model.graph.node) == 0:
                continue
            inpnodes = get_input_nodes(remaining_model)
            # print('inputnodes', i, inp, len(inpnodes), [n.op_type for n in inpnodes], [n.input for n in inpnodes], reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]))
            # It prevents the fully connected layer to be the
            if len(inpnodes) > 1 and not reduce(
                operator.and_, ["Conv" == n.op_type for n in inpnodes]
            ):
                continue
            model = remaining_model
            input = first_part_output
            change_layers_dimension(first_part_model)
            first_part_model = fix_reshape_batch_dimension(first_part_model)
            # print(input_name, f1.graph.input[0].name)
            if first_part_model.graph.input[0].name == input_name:
                change_model_name(first_part_model, "start")
            fragments.append(first_part_model)
        except IndexError:
            pass
    # print('len(get_input_nodes(model))', len(get_input_nodes(model)), [n.name for n in get_input_nodes(model)])
    inpnodes = get_input_nodes(model)
    # print('inputnodes', inp, len(inpnodes), [n.op_type for n in inpnodes], [n.input for n in inpnodes])
    if len(inpnodes) > 1 and not reduce(
        operator.and_, ["Conv" == n.op_type for n in inpnodes]
    ):
        return fragments
    change_layers_dimension(model)
    model = fix_reshape_batch_dimension(model)
    if model.graph.output[0].name == output_name:
        change_model_name(model, "end")
    fragments.append(model)
    return fragments
