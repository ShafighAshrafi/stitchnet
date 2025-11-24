import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from service.fragment.net import Net
from utilities.change_model_layers_dimension import change_layers_dimension
from utilities.fix_reshape_batch_dimension import fix_reshape_batch_dimension
from utilities.providers import PROVIDERS


def accuracy_score_net(net: Net, dataset, bs=64):
    assert len(net) == 1, "stitchnet should have only one fragment"
    count = 0
    fragmentC = net[0]
    change_layers_dimension(fragmentC.fragment)
    fragmentC.fragment = fix_reshape_batch_dimension(fragmentC.fragment)
    ort_sess1 = ort.InferenceSession(
        fragmentC.fragment.SerializeToString(), providers=PROVIDERS
    )

    for x, t in tqdm(dataset, position=0, leave=True):
        data = x.numpy()

        # Add batch dimension if missing (dataset yields single samples, model expects batches)
        if data.ndim == 3:  # [C, H, W] -> [1, C, H, W]
            data = np.expand_dims(data, axis=0)

        # y = evalulate_stitchnet(net, x)
        outputs = ort_sess1.run(None, {fragmentC.fragment.graph.input[0].name: data})
        y = outputs[0]
        # y = ys[0]
        # print('y.shape', y.shape)
        y = np.argmax(y, 1)

        # Convert target to Python int - handle both tensor and scalar cases
        if hasattr(t, "numpy"):
            target_value = t.numpy()
            # If it's an array, get the scalar value and convert to Python int
            if isinstance(target_value, np.ndarray):
                target_value = int(
                    target_value.item() if target_value.ndim == 0 else target_value[0]
                )
            else:
                target_value = int(target_value)
        else:
            # t is already a scalar (int/float)
            target_value = int(t)

        # y is shape [1] from argmax (batch size 1), compare y[0] with target
        count += int(y[0] == target_value)
    accuracy = 1.0 * count / len(dataset)
    return accuracy
