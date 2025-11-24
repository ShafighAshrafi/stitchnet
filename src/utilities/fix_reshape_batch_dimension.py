"""
Fix reshape nodes to support dynamic batch dimensions.
Reshape nodes with constant shape values that have batch size 1 are modified
to use -1 (dynamic) for the batch dimension.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx import helper


def fix_reshape_batch_dimension(model, verbose=False):
    """
    Fix reshape nodes to support dynamic batch dimensions.
    Modifies reshape nodes that have constant shape values with batch size 1
    to use -1 for the batch dimension.
    Handles both initializers and Constant nodes.

    Args:
        model: ONNX model
        verbose: If True, print debug information

    Returns:
        Modified ONNX model with dynamic batch dimensions in reshape nodes
    """
    # First, try to extract constants to initializers to simplify handling
    from onnxoptimizer import optimize

    try:
        # Extract constants to initializers first
        model = optimize(model, passes=["extract_constant_to_initializer"])
    except Exception:
        # If optimization fails, continue with original model
        pass

    # Find all reshape nodes
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]

    if not reshape_nodes:
        return model

    # Create a mapping of initializer names to their indices
    initializer_map = {
        init.name: idx for idx, init in enumerate(model.graph.initializer)
    }

    # Create a mapping of node outputs to Constant nodes
    constant_node_map = {}
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) > 0:
            constant_node_map[node.output[0]] = node

    modified = False
    new_initializers = list(model.graph.initializer)
    new_nodes = list(model.graph.node)

    for node in reshape_nodes:
        if len(node.input) < 2:
            continue

        shape_input_name = node.input[1]
        shape_values = None
        init_idx = None
        const_node_idx = None

        # Check if shape is an initializer
        if shape_input_name in initializer_map:
            init_idx = initializer_map[shape_input_name]
            init = new_initializers[init_idx]

            # Get shape values
            shape_array = numpy_helper.to_array(init)
            if shape_array.dtype != np.int64:
                shape_array = shape_array.astype(np.int64)

            shape_values = shape_array.flatten().tolist()

        # Check if shape comes from a Constant node
        elif shape_input_name in constant_node_map:
            const_node = constant_node_map[shape_input_name]
            # Find the constant node in the node list
            for idx, n in enumerate(new_nodes):
                if n == const_node:
                    const_node_idx = idx
                    break

            # Extract value from Constant node
            for attr in const_node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    tensor = attr.t
                    shape_array = numpy_helper.to_array(tensor)
                    if shape_array.dtype != np.int64:
                        shape_array = shape_array.astype(np.int64)
                    shape_values = shape_array.flatten().tolist()
                    break

        # If first dimension is 1, change it to -1 (dynamic)
        if shape_values and len(shape_values) > 0 and shape_values[0] == 1:
            if verbose:
                print(
                    f"Fixing reshape node '{node.name}': changing shape {shape_values} to ",
                    end="",
                )
            shape_values[0] = -1
            if verbose:
                print(f"{shape_values}")
            modified = True

            if init_idx is not None:
                # Update the initializer
                new_init = numpy_helper.from_array(
                    np.array(shape_values, dtype=np.int64),
                    name=new_initializers[init_idx].name,
                )
                new_initializers[init_idx] = new_init
            elif const_node_idx is not None:
                # Update the Constant node - create a new node with updated value
                const_node = new_nodes[const_node_idx]
                new_tensor = numpy_helper.from_array(
                    np.array(shape_values, dtype=np.int64)
                )
                # Create a new Constant node with updated value
                new_const_node = helper.make_node(
                    "Constant",
                    [],
                    const_node.output,
                    value=new_tensor,
                    name=const_node.name if const_node.name else None,
                )
                new_nodes[const_node_idx] = new_const_node

    if modified:
        # Recreate the model with updated initializers and nodes
        from utilities.create_onnx_model import create_onnx_model

        return create_onnx_model(model, nodes=new_nodes, initializer=new_initializers)

    return model
