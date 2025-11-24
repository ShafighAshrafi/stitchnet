from tqdm import tqdm

from utilities.get_fragments_score import get_score_fragments


def get_score_net(net, data_score, num_samples=1000):
    totalscore = 1
    for i, fragment1 in tqdm(enumerate(net[:-1]), position=0, leave=True):
        fragment2 = net[i + 1]
        score = get_score_fragments(
            fragment1, fragment2, data_score, num_samples=num_samples
        )
        # print(fragment1.fragment.graph.output[0].name, fragment2.fragment.graph.input[0].name, score)
        totalscore *= score
    return totalscore
