from src.service.fragment.net import Net
from src.service.stitching.stitch_fragments import stitch_fragments
from src.service.stitching.find_next_fragment import find_next_fragment


def get_net_id(curr, nextf):
    fId = curr.get_id()
    if type(fId[0]) is tuple:
        nId = fId[0] + (nextf.get_id(),)
    else:
        nId = (fId,) + (nextf.get_id(),)
    return nId


def recursive_stitching(curr, scoreMapper, data, threshold=0.9, totalThreshold=0.5, totalscore=1, maxDepth=10, sample=False, K=None):
    for score,nextf in find_next_fragment(curr, scoreMapper, data, threshold, maxDepth, sample, K):
        # skip fragment if it is the same as current fragment
        # print('curr', curr.fragment)
        # print('nextf', nextf.fragment)
        totalscore_nextf = totalscore*score
        print(f'totalscore before thresholding of {totalThreshold}: {totalscore_nextf}');
        if totalscore_nextf < totalThreshold:
            continue
        try:
            if nextf.fragment.graph.name=='end':
                newcurr_fragment = stitch_fragments(curr, nextf, data)
                newcurr_net = Net([newcurr_fragment], get_net_id(curr, nextf))
                newcurr = newcurr_net[0]
                yield totalscore_nextf, newcurr
            else:
                newcurr_fragment = stitch_fragments(curr, nextf, data)
                newcurr_net = Net([newcurr_fragment], get_net_id(curr, nextf))
                newcurr = newcurr_net[0]
                for _score, _curr in recursive_stitching(newcurr, scoreMapper, data, threshold, totalThreshold, totalscore_nextf, maxDepth, sample, K):
                    yield _score, _curr
        except Exception as e:
            # catch death end path with errors
            print('[WARNING]', e)
            # traceback.print_exc()
            # raise e
            # pass