import numpy as np
from src.service.fragment.net import Fragment
from src.utilities.score_mapper import ScoreMapper


# check if next fragment already in curr
def check_fragment_exists(current_network, next_fragment):
    fragment_id = current_network.get_id()
    next_fragment_id = next_fragment.get_id()
    if type(fragment_id[0]) is tuple:
        return next_fragment_id in fragment_id[0]
    else:
        return next_fragment_id in (fragment_id,)


def find_next_fragment(current_network: Fragment, scoreMapper: ScoreMapper, data, threshold=0.5, maxDepth=10, sample=False, K=None):
    x1 = current_network.get_output(data)
    fragment_id = current_network.get_id()
    if type(fragment_id[0]) is tuple:
        curDepth = len(fragment_id[0])
    else:
        curDepth = 1
    # print('current depth:', curDepth)
    scores = scoreMapper.score(x1, curDepth, max_depth=maxDepth)
    if K is not None:
        scores = scores[:K]
    # print('potential next fragments:', len(scores))
    # print(f'potential next fragments before thresholding of {threshold}:', len(scores), [f'{s[0]:.2}' for s in scores])
    # filter out the previous in curr

    scores = [(score, next_fragment) for score, next_fragment in scores if not check_fragment_exists(
        current_network, next_fragment)]

    # print(f'potential next fragments after filter duplicated fragments:', len(scores), [f'{s[0]:.2}' for s in scores])
    print(f'potential next fragments before thresholding of {threshold}:', len(
        scores), [f'{s[0]:.2}' for s in scores])

    if threshold is not None:
        scores = [score for score in scores if score[0] > threshold]
    # print('scores', [f'{s[0]:.2}' for s in scores])
    if sample and len(scores) > 0:
        i = np.random.choice(range(len(scores)))
        scores = [scores[i]]
    print(f'potential next fragments after thresholding of {threshold}:', len(
        scores), [f'{s[0]:.2}' for s in scores])
    for score, next_fragment in scores:
        # score = get_score_fragments(curr, nextf, data)
        # if score < threshold:
        #     continue
        # yield score,nextf
        # if s > threshold:
        #     yield s,nextf
        yield score, next_fragment
