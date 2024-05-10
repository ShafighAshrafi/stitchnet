import os
import traceback

from src.utilities.visual import draw_stitchNet
from src.utilities.get_macs import get_macs_params
from src.utilities.get_accuracy import accuracy_score_net


class Report:
    def __init__(self, bs=128, fname=f'../_results_with_finetune/result_val.txt', mode='a'):
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        self.fname = fname
        self.mode = mode
        self.bs = bs
        
    def evaluate(self, nets, net, netname, score, dataset):
        dot, gname = draw_stitchNet(nets, net, name=netname)
        accuracy = accuracy_score_net(net, dataset, bs=self.bs)
        print('accuracy', accuracy)
        macs, params = get_macs_params(net[0])
        network = net.get_id()
        self.f.write(f'{score},{accuracy},{macs},{params},{gname},"{network}"\n')
        self.f.flush()
        
    def close(self):
        self.f.close()
    
    def __enter__(self):
        self.f = open(self.fname, self.mode)
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        self.close()
        return True