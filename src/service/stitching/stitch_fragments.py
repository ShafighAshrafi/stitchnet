import time
import copy
import torch
import operator
import traceback
import onnxoptimizer
from onnx import helper
from functools import reduce
from src.utilities.get_input_nodes import get_input_nodes
from src.utilities.adjust_weights.adjust_w import adjust_w
from src.utilities.get_output_nodes import get_output_nodes
from src.utilities.create_onnx_model import create_onnx_model
from src.utilities.change_model_layers_dimension import change_layers_dimension


def stitch_fragments(fragment1, fragment2, data):
    # list_ops(fragment1.fragment)
    # fragment1 = copy.deepcopy(fragment1)
    # fragment2 = copy.deepcopy(fragment2)

    try:
        x1 = fragment1.get_output(data)
        x2 = fragment2.get_input(data)
    except Exception as e:
        traceback.print_exc()
        # print('-------fragment1 ops-------')
        # list_ops(fragment1.fragment)
        # print('-------fragment2 ops-------')
        # list_ops(fragment2.fragment)
        raise e
    tX = torch.from_numpy(x1)
    tY = torch.from_numpy(x2)
    # score = get_score(tX, tY)
    # if score < 0.5:
    #     return None
    # print('score', score)

    ws = fragment2.get_ws()
    nws = []
    for i, w in enumerate(ws):
        nw = adjust_w(tX, tY, w)
        nws.append(nw)
    newFragment = fragment2.replace_ws(nws)
    # newFragment = fragment2.fragment

    oldinputname = newFragment.graph.input[0].name
    newname = oldinputname+"_timestamp_"+str((time.time()))

    # find and replace all the input name
    for n in newFragment.graph.node:
        for i, inp in enumerate(n.input):
            if inp == oldinputname:
                n.input[i] = newname
    newFragment.graph.input[0].name = newname

    # begin stitching
    exitNodes = get_output_nodes(fragment1.fragment)
    # print('exitNodes', [n.name for n in exitNodes])
    exitNode = exitNodes[0]

    nodes = []
    for node in fragment1.fragment.graph.node:
        if node.name == exitNode.name:
            # print('skipping...', node.name)
            continue
        node = copy.deepcopy(node)
        nodes.append(node)

    # newOutput = newname
    node = exitNode
    kwargs = {}
    for att in node.attribute:
        # print(att)
        kwargs[att.name] = helper.get_attribute_value(att)

    newNodeOutName = newname
    if tX.ndim == 4 and tY.ndim == 2:
        newNodeOutName += "_pool"
        poolNode = helper.make_node(
            'GlobalAveragePool',
            inputs=[f'{newNodeOutName}'],
            outputs=[f'{newNodeOutName}_poolflat'],
        )
        flatNode = helper.make_node(
            'Flatten',
            inputs=[f'{newNodeOutName}_poolflat'],
            outputs=[f'{newname}'],  # Default value for axis: axis=1
        )

    newNode = helper.make_node(
        node.op_type,                  # optype
        node.input,  # inputs
        [newNodeOutName],               # outputs
        node.name+"_nodeexit_timestamp_"+str((time.time())),  # name
        node.doc_string,
        node.domain,
        # mode='constant',        # attributes
        # name=newname+"_nodeexit_"+str((time.time()))
        # doc_string=exitNode.doc_string,
        # domain=exitNode.domain,
        # *node.attribute
        **kwargs
    )

    # newNode = helper.make_node(
    #     exitNode.op_type,                  # name
    #     exitNode.input, # inputs
    #     [newname],                  # outputs
    #     # mode='constant',        # attributes
    #     name=newname+"_nodeexit_"+str((time.time()))
    # )

    nodes.append(newNode)
    if tX.ndim == 4 and tY.ndim == 2:
        nodes.append(poolNode)
        nodes.append(flatNode)

    inpnodes = get_input_nodes(newFragment)
    # print('[n.op_type for n in inputnodes]', [n.op_type for n in inpnodes])
    if len(inpnodes) > 1 and not reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]):
        raise Exception(
            "There are more than one inputs to stitch and it is not all going into Conv.")

    if len(inpnodes) == 0:
        # print('len(inpnodes)', len(inpnodes))
        # print('newFragment', newFragment, fragment2.fragment)
        # list_ops('newFragment', newFragment)
        # list_ops('fragment2', fragment2.fragment)
        raise Exception("No inputs.")

    enterNode = inpnodes[0]

    node = enterNode
    kwargs = {}
    for att in node.attribute:
        # print(att)
        kwargs[att.name] = helper.get_attribute_value(att)

    newNode = helper.make_node(
        node.op_type,                  # optype
        [newname]+enterNode.input[1:],  # inputs
        node.output,               # outputs
        node.name+"_nodeenter_timestamp_"+str((time.time())),  # name
        node.doc_string,
        node.domain,
        # mode='constant',        # attributes
        # name=newname+"_nodeexit_"+str((time.time()))
        # doc_string=exitNode.doc_string,
        # domain=exitNode.domain,
        # *node.attribute
        **kwargs
    )

    # newNode = helper.make_node(
    #     enterNode.op_type,                  # name
    #     [newname]+enterNode.input[1:],    # inputs
    #     enterNode.output,              # outputs
    #     # mode='constant',        # attributes
    #     name=newname+"_nodeenter_"+str((time.time()))
    # )
    existingNames = {}
    for n in nodes:
        for item in n.input:
            existingNames[item] = True
        for item in n.output:
            existingNames[item] = True

    # change node names
    newnodes = []
    newnodes.append(newNode)
    for node in newFragment.graph.node:
        if node.name == enterNode.name:
            # print('skipping...', node.name)
            continue
        node = copy.deepcopy(node)
        node.name += "_timestamp_"+str((time.time()))
        newnodes.append(node)

    initializer = []
    for init in fragment1.fragment.graph.initializer:
        initializer.append(init)

    newinitname = {}
    for init in newFragment.graph.initializer:
        init = copy.deepcopy(init)
        oldname = init.name
        init.name += "_timestamp_"+str((time.time()))
        newinitname[oldname] = init.name
        initializer.append(init)

    for node in newnodes:
        for i, inp in enumerate(node.input):
            if inp in existingNames:
                # print('existingNames', inp)
                if inp not in newinitname:
                    if 'timestamp' not in inp:
                        newinitname[inp] = inp+"_timestamp_"+str((time.time()))
                    # else:
                    #     newinitname[inp] = inp.split('_')[0]+"_timestamp_"+str((time.time()))

                    # node.input[i] = newNames[inp]

        for i, out in enumerate(node.output):
            if out in existingNames:
                # print('existingNames', out)
                if out not in newinitname:
                    if 'timestamp' not in out:
                        newinitname[out] = out+"_timestamp_"+str((time.time()))
                    # else:
                    #     newinitname[out] = out.split('_')[0]+"_timestamp_"+str((time.time()))
                    # node.output[i] = newNames[out]
                # if inp not in newouts:
                #     newouts[inp] = inp+"_"+str((time.time()))
                # node.output[i] = newouts[inp]

    # replace input output with new names
    for node in newnodes:
        for i, inp in enumerate(node.input):
            if inp in newinitname:
                node.input[i] = newinitname[inp]
        for i, out in enumerate(node.output):
            if out in newinitname:
                node.output[i] = newinitname[out]
        if 'timestamp' not in node.name:
            node.name += "_timestamp_"+str((time.time()))
        else:
            node.name = node.name.split(
                '_')[0]+"_timestamp_"+str((time.time()))

    inputs = copy.deepcopy(fragment1.fragment.graph.input)
    for inp in inputs:
        if inp.name in newinitname:
            inp.name = newinitname[inp.name]

    outputs = newFragment.graph.output
    for out in outputs:
        if out.name in newinitname:
            out.name = newinitname[out.name]

    nodes += newnodes
    # print('end stitching')
    name = 'stitch'
    try:
        newnet = create_onnx_model(fragment1.fragment, nodes=nodes, name=name,
                                   inputs=inputs, outputs=outputs, initializer=initializer)
        newnet = onnxoptimizer.optimize(newnet, passes=[
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm'
        ])
        change_layers_dimension(newnet)
        return newnet
    except Exception as e:
        print(e)
        traceback.print_exc()
        for node in nodes:
            print(node.name, node.input, node.output)
