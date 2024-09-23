import numpy as np

def unique(x, sort=False):
    y, indices = np.unique(x, return_index=True)
    if sort:
        return y
    return np.array([x[index] for index in sorted(indices)])