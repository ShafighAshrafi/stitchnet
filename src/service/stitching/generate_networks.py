import numpy as np
from src.service.stitching.recursive_stitching import recursive_stitching

 
def generate_networks(nets, scoreMapper, data, threshold=0.9, totalThreshold=0.5, maxDepth=10, sample=False, K=None):
    fragments = [f for net in nets for f in net]
    starts = [f for f in fragments if f.fragment.graph.name == 'start']
    ends = [f for f in fragments if f.fragment.graph.name == 'end']
    middles = [f for f in fragments if f.fragment.graph.name not in ['start','end']]
    if sample:
        starts = [np.random.choice(starts)]
    for start in starts:
        for score, curr in recursive_stitching(start, scoreMapper, data, threshold, totalThreshold, 1.0, maxDepth=maxDepth, sample=sample, K=K):
            yield score, curr.net