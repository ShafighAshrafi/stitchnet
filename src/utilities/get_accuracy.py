import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from src.utilities.providers import PROVIDERS
from src.utilities.load_dataset import load_dataset
from src.utilities.dataloader_generator import generate_dataloader
from src.utilities.change_model_layers_dimension import change_layers_dimension


def accuracy_score_net(net, dataset, bs=64):
    assert len(net)==1, 'stitchnet should have only one fragment'
    count = 0
    fragmentC = net[0]
    change_layers_dimension(fragmentC.fragment)
    ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
    
    for x,t in tqdm(generate_dataloader(load_dataset(), batch_size=bs)):
        data = x.numpy()

        # y = evalulate_stitchnet(net, x)
        inputs = {}
        inputs[fragmentC.fragment.graph.input[0].name] = data
        outputs = ort_sess1.run(None, inputs)
        y = outputs[0]
        # y = ys[0]
        # print('y.shape', y.shape)
        y = np.argmax(y, 1)
        # print('y.shape', y.shape)
        # y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy