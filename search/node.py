import torch

class PoolSearchStateNode:

    def __init__(self, state, feat=None, logit=None):
        self.state = state
        self.feat = feat
        self.logit = logit
        self.score = torch.FloatTensor([0.])[0]
        self.entropy = None

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self):
        return ','.join([str(a) for a in self.state])