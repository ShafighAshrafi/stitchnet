from src.service.fragment.net import Fragment, Net
from src.utilities.score_mapper import ScoreMapper
from src.service.stitching.stitch_fragments import stitch_fragments
from src.service.stitching.find_next_fragment import find_next_fragment


def get_net_id(current_network, next_fragment):
    fId = current_network.get_id()
    if type(fId[0]) is tuple:
        nId = fId[0] + (next_fragment.get_id(),)
    else:
        nId = (fId,) + (next_fragment.get_id(),)
    return nId


def recursive_stitching(current_network: Fragment, scoreMapper: ScoreMapper, data, threshold=0.9, totalThreshold=0.5, totalscore=1, maxDepth=10, sample=False, K=None):
    for score, next_fragment in find_next_fragment(current_network, scoreMapper, data, threshold, maxDepth, sample, K):
        # skip fragment if it is the same as current fragment
        # print('curr', curr.fragment)
        # print('nextf', nextf.fragment)
        totalscore_with_new_fragment = totalscore * score
        print(f'totalscore before thresholding of {totalThreshold}: {totalscore_with_new_fragment}');
        if totalscore_with_new_fragment < totalThreshold:
            continue
        try:
            newcurr_fragment = stitch_fragments(current_network, next_fragment, data)
            newcurr_net = Net([newcurr_fragment], get_net_id(current_network, next_fragment))
            new_network = newcurr_net[0]
            if next_fragment.fragment.graph.name=='end':
                yield totalscore_with_new_fragment, new_network
            else:
                for _score, _curr in recursive_stitching(new_network, scoreMapper, data, threshold, totalThreshold, totalscore_with_new_fragment, maxDepth, sample, K):
                    yield _score, _curr
        except Exception as e:
            # catch death end path with errors
            print('[WARNING]', e)
            # traceback.print_exc()
            # raise e
            # pass