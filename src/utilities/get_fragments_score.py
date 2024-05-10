import torch
from src.service.fragment.net import Fragment
from src.utilities.get_score import get_score


def get_score_fragments(fragment1: Fragment, fragment2: Fragment, data, num_samples=10000):
    x1 = fragment1.get_output(data)
    x2 = fragment2.get_input(data)
    tX = torch.from_numpy(x1)
    tY = torch.from_numpy(x2)
    score = get_score(tX, tY, num_samples=num_samples)
    return score