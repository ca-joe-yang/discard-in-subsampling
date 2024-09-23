import copy
import numpy as np

from .node import PoolSearchStateNode

class PoolSearchProgress:
    # Save the searching progress for each image
    def __init__(self, budget, num_downsample):
        self.budget = budget
        self.num_downsample = num_downsample
        node_0 = PoolSearchStateNode(
            state=[-1] * self.num_downsample, 
            feat=None, logit=None)
        self.candidates = []
        self.frontier = [ node_0 ]

    def __str__(self):
        ret = ''
        ret += 'Frontier: ' + '+'.join([ f'[{node}]' for node in self.frontier ])
        ret += '\n'
        ret += 'Candidates: ' + '+'.join([ f'[{node}]' for node in self.candidates ])
        return ret

    def is_end(self):
        if len(self.candidates) >= self.budget:
            return True
        
        if len(self.frontier) == 0:
            return True

        return False

    def next(self):
        if self.is_end():
            self.expand_node = None
            self.expand_idx = False
            return
        
        while True:
            if self.is_end():
                self.expand_node = None
                self.expand_idx = False
                return

            cur_node = self.frontier.pop(0)
            unexpanded_indices = np.where(np.array(cur_node.state) == -1)[0]
            if len(unexpanded_indices) == 0: continue
            break
        
        self.expand_idx = np.min(unexpanded_indices)
        self.expand_node = cur_node

    def get_expand_neighbors(self, num_expand):
        """
        neighboring states including itself
        """
        self.neighboring_states = []
        self.num_expand = num_expand
        state = self.expand_node.state
        for j in range(num_expand):
            self.neighboring_states.append(state[:self.expand_idx] + [j] + state[self.expand_idx+1:])
    
    def expand(self, 
        feat, logit,
    ):
        start = 0
        if len(self.candidates) != 0:
            # Ignore the first one since it is already in the candidates
            start = 1

        for k in range(start, self.num_expand):
            state = self.neighboring_states[k]
            new_node = PoolSearchStateNode(
                state=state, feat=feat[k], 
                logit=logit[k])
            self.candidates.append(new_node)
            self.frontier.append(new_node)#copy.deepcopy(new_node))
            self.frontier = sorted(self.frontier) 
            if len(self.candidates) >= self.budget:
                break           
        
        self.candidates = self.candidates[:self.budget]