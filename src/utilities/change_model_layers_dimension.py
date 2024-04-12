

def change_layers_dimension(model, dimension="N"):
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = dimension
    outputs = model.graph.output
    for output in outputs:
        if len(output.type.tensor_type.shape.dim)>0:
            dim1 = output.type.tensor_type.shape.dim[0]
            dim1.dim_param = dimension