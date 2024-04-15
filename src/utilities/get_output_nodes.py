def get_output_nodes(f):
    # print('f.graph.output', [n.name for n in f.graph.output])
    nodes = []
    for n in f.graph.node:
        for out in f.graph.output:
            if out.name in n.output:
                nodes.append(n)
    return nodes