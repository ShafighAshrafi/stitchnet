import torch
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from collections import defaultdict
from onnx_tool import create_ndarray_f32
from src.utilities.providers import PROVIDERS
from src.utilities.change_model_layers_dimension import change_layers_dimension
from src.utilities.get_score import get_score


class Net:
    def __init__(self, fragments, nId=None):
        self.id = nId
        self.fragments = fragments
        # for f in fragments:
        #     hashf = hash_model(f)
        #     print(hashf)
        fragmentCs = []
        for fId,f in enumerate(fragments):
            change_layers_dimension(f)
            fragmentC = Fragment(f, self, fId)
            fragmentCs.append(fragmentC)
        self.fragmentCs = fragmentCs
        self.results = defaultdict(dict)
        self.knn = None
        self.p = None
    
    # def get_macs_params(self):
    #     macs,params = get_macs_params(self[0])
    #     return dict(macs=macs,params=params)

    # def fit(self, train_dataset, label_column="label", batch_size=32):
    #     label = label_column
    #     net = self
        
    #     fragmentC = net[0]
    #     change_layers_dimension(fragmentC.fragment)
    #     self.ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
    #     self.input_name = fragmentC.fragment.graph.input[0].name
        
    #     nnX = []
    #     nnY = []
    #     ptdset_train = TensorDataset(train_dataset['pixel_values'], 
    #                          train_dataset[label])
    #     dl = load_dl(ptdset_train, batch_size=batch_size, shuffle=False, num_workers=0)
    #     for dataItem in tqdm(dl):
    #         try:
    #             X = dataItem[0].squeeze(1).numpy()
    #             t = dataItem[1].unsqueeze(1).numpy()
    #             inputs = {}
    #             inputs[self.input_name] = X
    #             outputs = self.ort_sess1.run(None, inputs)
    #             y = outputs[0]
    #             nnX.append(y)
    #             nnY.append(t)
    #         except Exception as e:
    #             print('ERROR TRAIN', e)
    #             traceback.print_exc()
        
    #     nnX = np.vstack(nnX)
    #     nnY = np.vstack(nnY).squeeze()
        
    #     knn = KNeighborsClassifier(n_neighbors=5)
    #     knn.fit(nnX, nnY)
        
    #     self.knn = knn
    #     self.task = "image-classification"
    
    # def __call__(self, pixel_values, **kwargs):
    #     return self.predict(pixel_values)
    # def predict(self, pixel_values):
    #     result = []
    #     for pixel_value in tqdm(load_dl(pixel_values, batch_size=32, shuffle=False, num_workers=0)):
    #     # for pixel_value in tqdm(pixel_values):
    #         try:
    #             X = pixel_value.squeeze(1).numpy()
    #             inputs = {}
    #             inputs[self.input_name] = X
    #             outputs = self.ort_sess1.run(None, inputs)
    #             y = outputs[0]
    #             y = np.array(y)
    #             # print('y.shape', y.shape)
    #             labels = self.knn.predict(y)
    #             scores = self.knn.predict_proba(y)
    #             # print('score',score)
    #             # print('label',label)
    #             for score,label in zip(scores,labels):
    #                 result += [{
    #                     "score": list(score),
    #                     "label": label
    #                 }]
    #         except Exception as e:
    #             print('ERROR EVAL', e)
    #             traceback.print_exc()
    #             result += [{
    #                 "score": 0,
    #                 "label": -1
    #             }]
    #     return result
    
    # def evaluate_dataset(self, dataset_val, label_column='labels'):
    #     result = self(dataset_val['pixel_values'])
    #     total = len(dataset_val)
    #     count = 0
    #     for r,t in zip(result,dataset_val['labels']):
    #         if r['label']==t:
    #             count+=1
    #     return {'accuracy': 1.*count/total}  
        
    # def predict_files(self, filenames):
    #     if not isinstance(filenames, list):
    #         filenames = [filenames]            
    #     if self.knn is None:
    #         raise Exception("classifier is not trained, please call fit before")
    #     if self.p is None:
    #         self.p = AutoProcessor.from_pretrained('microsoft/resnet-50')
    #     result = []
    #     try:
    #         from PIL import Image
    #         data = []
    #         for filename in filenames:
    #             im = Image.open(filename)
    #             data += self.p(im)['pixel_values']
    #         X = np.stack(data)
    #         inputs = {}
    #         inputs[self.input_name] = X
    #         outputs = self.ort_sess1.run(None, inputs)
    #         y = outputs[0]
    #         y = np.array(y)
    #         # print('y.shape', y.shape)
    #         labels = self.knn.predict(y)
    #         scores = self.knn.predict_proba(y)
    #         for score,label in zip(scores,labels):
    #             result += [{
    #                 "score": list(score),
    #                 "label": label
    #             }]
    #     except Exception as e:
    #         print('ERROR EVAL', e)
    #         traceback.print_exc()
    #         result += [{
    #             "score": 0,
    #             "label": -1
    #         }]
    #     return result
        
    # def draw_svg(self, path):
    #     draw_net(self, path)
        
    # def save_onnx(self, path):
    #     self.save(path)
        
    # def save(self, path):
    #     # print('len(self.fragments)', len(self.fragments))
    #     if len(self.fragments) == 1:
    #         print('saving to', f'{path}.onnx')
    #         save_onnx_model(self.fragments[0], f'{path}.onnx')
    #     else:
    #         os.makedirs(os.path.dirname(path), exist_ok=True)
    #         for i,fragment in enumerate(self.fragments):
    #             save_onnx_model(fragment, f'{path}_{i:03}.onnx')
        
    # def get_id(self):
    #     return self.id
    def get_scores(self, x1, data, scoring_method='CKA'):
        tX = torch.from_numpy(x1)
        score_fragments = []
        for i,f in enumerate(self.fragments[:-1]):
            x2 = self.get_output(f,data)
            tY = torch.from_numpy(x2)
            score = get_score(tX, tY, min(tX.shape[1],tY.shape[1])*10, scoring_method)
            score_fragments.append((score, self.fragmentCs[i+1]))
        return score_fragments
    
    # def evaluate(self, x):
    #     if isinstance(x, torch.Tensor):
    #         x = x.numpy()
    #     hashx = hashlib.md5(x.tobytes()).hexdigest()
    #     os = execute_fragments(self.fragments, x)
    #     for f,o in zip(self.fragments,os):
    #         hashf = hash_model(f)
    #         # print(hashf, hashx)
    #         self.results[hashf][hashx] = o
    
    # def get_outputs(self, x):
    #     return [self.get_output(f,x) for f in self.fragments]
    
    # def get_output(self, f, x):
    #     hashf = hash_model(f)
    #     # print('hashf', hashf)
    #     if isinstance(x, torch.Tensor):
    #         x = x.numpy()
    #     hashx = hashlib.md5(x.tobytes()).hexdigest()
    #     if hashx in self.results[hashf]:
    #         return self.results[hashf][hashx]
    #     else:
    #         self.evaluate(x)
    #         return self.results[hashf][hashx]
        
    # def get_input(self, f, x):
    #     curr = None
    #     for nf in self.fragments:
    #         prev = curr
    #         curr = nf
    #         if hash_model(nf) == hash_model(f):
    #             if prev is None:
    #                 return x
    #             return self.get_output(prev, x)
            
    def __iter__(self):
        self.index = 0
        return iter(self.fragmentCs)
    # def __next__(self):
    #     if self.index < len(self):
    #         item = self.fragmentCs[self.index]
    #         self.index+=1
    #         return item
    #     else:
    #         raise StopIteration
    def __getitem__(self,i):
        return self.fragmentCs[i]
    def __len__(self):
        return len(self.fragmentCs)


class Fragment: 
    def __init__(self, fragment, net: Net, fId=None):
        self.id = fId
        self.fragment = fragment
        self.net = net
    def get_last_id(self):
        if type(self.net.id) is tuple:
            return self.net.id[-1]
        else:
            return self.get_id()
    def get_id(self):
        return (self.net.id,self.id)
    def get_output(self, x):
        return self.net.get_output(self.fragment, x)
    def get_input(self, x):
        return self.net.get_input(self.fragment, x)
    # def get_w(self):
    #     f = self.fragment
        
    #     nodes = get_input_nodes(f)
        
    #     node = nodes[0]
    #     name = node.input[1]
    #     w = get_numpy_matrix(f, name)
    #     return np.copy(w)
    
    # def get_ws(self):
    #     f = self.fragment
        
    #     nodes = get_input_nodes(f)
        
    #     ws = []
    #     for node in nodes:
    #         name = node.input[1]
    #         w = get_numpy_matrix(f, name)
    #         ws.append(np.copy(w))
    #     return ws
    
    # def replace_w(self, w, i=0):
    #     f = self.fragment
    #     nodes = get_input_nodes(f)
    #     node = nodes[i]
    #     name = node.input[1]
    #     return replace_w(f, name, w)
    
    # def replace_ws(self, ws):
    #     f = self.fragment
    #     nodes = get_input_nodes(f)
    #     names = [node.input[1] for node in nodes]
    #     return replace_ws(f, names, ws)


# def get_macs_params(fragment: Fragment, inputName=None, inputSize=(1, 3, 224, 224)):
#     inputs= {}
#     if inputName is None:
#         inputName = fragment.fragment.graph.input[0].name
#     inputs[inputName] = create_ndarray_f32(inputSize)
#     # change_input_dim(fragment.fragment,1)
#     macs,params=graph_profile(fragment.fragment.graph, inputs)
#     return macs, params