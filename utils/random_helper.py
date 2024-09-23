import random
import numpy as np
import torch

def set_random_seed(seed: int) -> None:
    r'''set randoom seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    