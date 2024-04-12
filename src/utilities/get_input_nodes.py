

def get_input_nodes(model):
    nodes = []
    for node in model.graph.node:
        for input in model.graph.input:
            if input.name in node.input:
                nodes.append(node)
    return nodes