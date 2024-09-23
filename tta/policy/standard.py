import numpy as np

from .base import BasePolicy
from .utils import unique
from .ops import PoolState, FlipLR, Scale, FiveCrops, Sequential

class StandardPolicy(BasePolicy):

    name = 'standard'
    # max_budget = 150

    def __init__(self, dm):
        super().__init__()
        self.crop_size = dm.crop_size
        s = list(np.arange(0.08, 1.11, 0.06))
        s = [ round(x, 2) for x in s ]
        s = sorted(s, key=lambda x: np.abs(x - 1.00) )
        s2 = list(np.arange(0.08, 1.11, 0.03))
        s2 = [ round(x, 2) for x in s2 ]
        s2 = sorted(s2, key=lambda x: np.abs(x - 1.00) )
        self.scale_params = list(unique(
            [1.00, 1.04, 1.10] + list(s) + list(s2),
            sort=False))[:19]
        print(self.scale_params)
        
        self.five_crops_params = [
            'center_crop', 'crop_lt', 'crop_lb', 'crop_rb', 'crop_rt']

        self.max_budget = 2 * 5 * len(self.scale_params)
        print(self.max_budget, self.scale_params)

        tta_transforms = [
            [FlipLR(0), FlipLR(1)],
            [Scale(s) for s in self.scale_params],
            [FiveCrops(0, self.crop_size, self.crop_size), 
             FiveCrops(1, self.crop_size, self.crop_size), 
             FiveCrops(2, self.crop_size, self.crop_size), 
             FiveCrops(3, self.crop_size, self.crop_size), 
             FiveCrops(4, self.crop_size, self.crop_size)],
        ]

        transforms = []
        for i in range(2):
            t1 = tta_transforms[0][i]
            p1 = t1.val
            for j in range(len(self.scale_params)):
                t2 = tta_transforms[1][j]
                p2 = t2.val
                for k in range(5):
                    t3 = tta_transforms[2][k]
                    p3 = t3.val

                    t = Sequential([t1, t2, t3])
                    transforms.append([t, t.name, p1, p2, p3])

        transforms = sorted(transforms, 
            key=lambda x: self.five_crops_params.index(x[4]))
        transforms = sorted(transforms, 
            key=lambda x: self.scale_params.index(x[3]))
        # transforms = sorted(transforms, 
        #     key=lambda x: x[5][0])
        transforms = [ x[0] for x in transforms ]
        
        self.fns = transforms
        
        self.transforms_names = [fn.name for fn in self.fns]
        print(self.transforms_names[:30])
        self.num_augs = len(self.transforms_names)
        self.tta_transforms = self.fns
        # ttach.Compose(
        #     [TransformCustomFunctions(self.fns, self.crop_size)])

        self.name2fns = {}
        for i in range(len(self.fns)):
            self.name2fns[self.transforms_names[i]] = self.fns[i]


# def get_tta_functions_from_aug_order(aug_order, dataset, budget=None):
#     crop_size = 224
#     if dataset == 'cifar100':
#         crop_size = 32 
#     elif dataset == 'stl10':
#         crop_size = 96
#     if aug_order[0] == 'pil':
#         tta_functions = tta.base.Compose(
#             [AllPIL(crop_size, dataset, budget)])
#         return tta_functions
#     transform_map = {
#         'hflip': tta.transforms.HorizontalFlip(),
#         # 'vflip': tta.transforms.VerticalFlip(),        
#         #  'flips': tta.transforms.Flips(),
#         'five_crop': tta.transforms.FiveCrops(crop_size, crop_size),
#         'scale': tta.transforms.Scale([1.04, 1.10]),
#         #  'modified_five_crop': tta.transforms.ModifiedFiveCrops(crop_size, crop_size)
#     }
#     fns = [transform_map[x] for x in aug_order]
#     tta_functions = tta.base.Compose(fns)
#     return tta_functions

if __name__ == '__main__':
    from PIL import Image
    import torchvision
    import sys
    sys.path.append('.')
    import poolsearch
    dm = poolsearch.data.toy.Toy()
    tm = StandardPolicy(dm)
    x = Image.open('lena.png')
    x = dm.transform(x)
    
    for i in range(10):
        fn = tm.fns[i]
        y = fn(x.unsqueeze(0))
        print(fn)
        y = dm.unnormalize(y).squeeze(0)
        torchvision.utils.save_image(y, f'img/lena-{i}.jpg')