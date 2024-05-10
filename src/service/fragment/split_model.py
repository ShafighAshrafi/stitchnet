import onnxruntime as ort
from src.utilities.providers import PROVIDERS
from src.service.fragment.select_input_output import select_model_inputs_outputs2


def split_model_at(model_onnx, output, x):
    first_part_model = select_model_inputs_outputs2(model_onnx, output)
    session = ort.InferenceSession(
        first_part_model.SerializeToString(), providers=PROVIDERS)
    inputs = {}
    inputs[first_part_model.graph.input[0].name] = x
    first_part_output = session.run(None, inputs)

    remaining_model = select_model_inputs_outputs2(
        model_onnx, [o.name for o in model_onnx.graph.output], output, 1, first_part_output[0].shape)
    
    return first_part_model, remaining_model, first_part_output