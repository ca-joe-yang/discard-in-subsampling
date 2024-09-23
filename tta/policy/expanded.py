import itertools
import numpy as np

from .base import BasePolicy
from .ops import *

class ExpandedPolicy(BasePolicy):
    
    name = 'expanded'
    max_budget = 200
    ops_pool = [
        (Identity, 1),
        (FlipLR, 1),
        (FlipUD, 1),
        (Invert, 1),
        (PIL_Blur, 1),
        (PIL_Smooth, 1),
        (AutoContrast, 1),
        (Equalize, 1),
        (Posterize, 4, 1, 4),
        (Rotate, 10, -30, 30),
        (CropBilinear, 10, 1, 10),
        (Solarize, 10, 0., 1.),
        (Contrast, 10, 0.1, 1.9),
        (Saturation, 10, 0.1, 1.9),
        (Brightness, 10, 0.1, 1.9),
        (Sharpness, 10, 0.1, 1.9),
        (ShearX, 10, -0.3, 0.3),
        (ShearY, 10, -0.3, 0.3), 
        (TranslateX, 10, -9, 9), 
        (TranslateY, 10, -9, 9), 
        (Cutout, 10, 2, 20),
    ]

    def __init__(self, dm):
        self.crop_size = dm.crop_size

        fns = []
        for i in range(len(self.ops_pool)):
            op_func, n_levels = self.ops_pool[i][:2]
            op_name = op_func.__name__
            if n_levels == 1:
                if op_name == 'Equalize':
                    transform = Equalize(dm)
                else:
                    transform = op_func()
                fns.append((transform, 0))
            else:
                min_val, max_val = self.ops_pool[i][2:]
                vals = np.linspace(min_val, max_val, n_levels)
                vals = sorted(vals, key=lambda x: np.abs(x))
                # closest_level = np.argmin(np.abs(vals))
                for level, val in enumerate(vals):
                    if op_name == 'CutOut':
                        transform = CutOut(length=val)
                    elif op_name in ['Solarize', 'Posterize']:
                        transform = op_func(dm, val)
                    else:
                        transform = op_func(val)                                        
                    fns.append((transform, level))
        fns = sorted(fns, key=lambda x: x[1])
        fns = [ t for t, l in fns ]
        fns = [ Sequential([t], self.crop_size) for t in fns ]
            
        fns2 = []
        for i, j in itertools.combinations(range(len(fns)), 2):
            transform = Sequential([fns[i], fns[j]], self.crop_size)
            fns2.append(transform)
        np.random.shuffle(fns2)
        
        self.fns = fns + fns2
        self.fns = self.fns[:self.max_budget]

        self.tta_transforms = self.fns

        self.transforms_names = [ p.name for p in self.fns ]
        self.num_augs = len(self.transforms_names)

        self.name2fns = {}
        for i in range(len(self.fns)):
            self.name2fns[self.transforms_names[i]] = self.fns[i]
