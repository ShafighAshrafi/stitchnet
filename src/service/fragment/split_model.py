import onnxruntime as ort

from service.fragment.select_input_output import select_model_inputs_outputs2
from utilities.providers import PROVIDERS


def split_model_at(model_onnx, output, x):
    first_part_model = select_model_inputs_outputs2(model_onnx, output)
    session = ort.InferenceSession(
        first_part_model.SerializeToString(), providers=PROVIDERS
    )
    inputs = {}
    inputs[first_part_model.graph.input[0].name] = x
    try:
        first_part_output = session.run(None, inputs)
    except Exception as e:
        # Fallback to CPU if CUDA fails (e.g., CUDNN_STATUS_BAD_PARAM)
        if "CUDA" in str(e) or "CUDNN" in str(e):
            session = ort.InferenceSession(
                first_part_model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            first_part_output = session.run(None, inputs)
        else:
            raise

    # Make batch dimension dynamic ("N") to support different batch sizes
    output_shape = list(first_part_output[0].shape)
    if len(output_shape) > 0:
        output_shape[0] = (
            "N"  # Dynamic batch dimension (matches change_layers_dimension)
        )

    remaining_model = select_model_inputs_outputs2(
        model_onnx,
        [o.name for o in model_onnx.graph.output],
        output,
        1,
        tuple(output_shape),
    )

    return first_part_model, remaining_model, first_part_output
