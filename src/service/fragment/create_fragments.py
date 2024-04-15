import operator
import traceback
from functools import reduce
from src.utilities.get_input_nodes import get_input_nodes
from src.service.fragment.split_model import split_model_at
from src.utilities.change_model_name import change_model_name
from src.service.fragment.find_conv_layers import find_conv_inputs
from src.utilities.change_model_layers_dimension import change_layers_dimension


#TODO: move the functionality in the loop to new function
#TODO: find out what the reduce line does
def get_fragments(model, input):
    '''chop up onnx models into fragments at conv and linear(gemm) layers'''
    inputs = find_conv_inputs(model)
    print(inputs)
    inputs = inputs[1:]
    fragments = []
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input = [input]
    for i,inp in enumerate(inputs):
        print(i, inp)
        try:
            newf1, newmodel, newx = split_model_at(model, inp, input[0], str(i))
            print('len(get_input_nodes(newf1))', len(get_input_nodes(newmodel)), [n.name for n in get_input_nodes(newmodel)])
            if len(newf1.graph.node)==0:
                continue
            inpnodes = get_input_nodes(newmodel)
            print('inputnodes', i, inp, len(inpnodes), [n.op_type for n in inpnodes], [n.input for n in inpnodes])
            if len(inpnodes) > 1 and not reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]):
                continue
            model = newmodel
            input = newx
            change_layers_dimension(newf1)
            # print(input_name, f1.graph.input[0].name)
            if newf1.graph.input[0].name == input_name:
                change_model_name(newf1, "start")
            fragments.append(newf1)
        except Exception:
            # skip multiple input dependencies
            # print('[WARNING]:', i, inp, e)
            traceback.print_exc()
            pass
    # print('len(get_input_nodes(model))', len(get_input_nodes(model)), [n.name for n in get_input_nodes(model)])
    inpnodes = get_input_nodes(model)
    # print('inputnodes', inp, len(inpnodes), [n.op_type for n in inpnodes], [n.input for n in inpnodes])
    if len(inpnodes) > 1 and not reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]):
        return fragments
    change_layers_dimension(model)
    if model.graph.output[0].name == output_name:
        change_model_name(model, "end")
    fragments.append(model)
    return fragments