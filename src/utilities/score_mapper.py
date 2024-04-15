from torch import Tensor
from threading import Thread


class ScoreMapper:
    def __init__(self, nets, data, scoring_method='CKA'):
        self.nets = nets
        self.scoreMap = {}
        self.data = data
        self.scoring_method = scoring_method

    def score(self, x1, curDepth, maxDepth=10):
        if isinstance(x1, Tensor):
            x1 = x1.numpy()
        
        scores = []
        def net_scores(net2, x1, data, scores):
            nextscores = net2.get_scores(x1,data,self.scoring_method)
            # only take ending segments
            if curDepth >= maxDepth-1:
                scores += nextscores[-1:]
            else:
                scores += nextscores        

        threads = [Thread(
                    target=net_scores, 
                    args=(net,x1,self.data,scores)
              ) for net in self.nets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        scores = sorted(scores, key=lambda x:x[0], reverse=True)
        # disabled caching for now since it is taking up too much memory
        # self.scoreMap[hashx] = scores
        # return self.scoreMap[hashx]
        return scores