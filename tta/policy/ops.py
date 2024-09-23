import math

import torch
import numpy as np
import torchvision.transforms as T

import ttach

class BaseOp:

    def __init__(self, dm=None):
        if dm is not None:
            self.mean = torch.FloatTensor(dm.mean).view(3, 1, 1)
            self.std = torch.FloatTensor(dm.std).view(3, 1, 1)

    def __repr__(self):
        return self.name

    @property
    def name(self):
        name = type(self).__name__
        if hasattr(self, 'val'):
            if isinstance(self.val, float):
                name += f'={self.val:.3f}'
            else:
                name += f'={self.val}'
        return name
        
    def unnormalize(self, img):
        device = img.device
        return img * self.std.to(device) + self.mean.to(device)
    
    def normalize(self, img):
        device = img.device
        return (img - self.mean.to(device)) / self.std.to(device) 

    # def augment_image(self, img):
    #     self.__call__(img)

class Sequential:

    def __init__(self, ops, crop_size=224):
        self.ops = ops
        self.crop_size = crop_size

    @property
    def name(self):
        name = [ op.name for op in self.ops ]
        return ','.join(name)

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        # x = T.functional.resize(x, (self.crop_size, self.crop_size), antialias=True)
        x = T.functional.center_crop(x, (self.crop_size, self.crop_size))
        return x

class Identity(BaseOp):

    def __call__(self, img, *args, **kwargs):
        return img

class FlipLR(BaseOp):

    def __init__(self, val=1):
        super().__init__()
        self.val = val

    @property
    def name(self):
        if self.val == 1:
            name = type(self).__name__
        else:
            name = 'Identity'
        return name

    def __call__(self, img, *args, **kwargs):
        if self.val == 1:
            return T.functional.hflip(img)
        return img

class FlipUD(BaseOp):

    def __call__(self, img, *args, **kwargs):
        return T.functional.vflip(img)

class Invert(BaseOp):

    def __call__(self, img, *args, **kwargs):
        return T.functional.invert(img)

class AutoContrast(BaseOp):

    def __call__(self, img, *args, **kwargs):
        return T.functional.autocontrast(img)

class Equalize(BaseOp):

    def __init__(self, dm):
        super().__init__(dm)

    def __call__(self, img, *args, **kwargs):
        img = (255.*self.unnormalize(img)).type(torch.uint8)
        img = T.functional.equalize(img)
        img = self.normalize(img.type(torch.float32)/255.)
        return img

class Posterize(BaseOp):

    def __init__(self, dm, val):
        super().__init__(dm)
        self.val = self.bits = val

    def __call__(self, img, *args, **kwargs):
        img = (255.*self.unnormalize(img)).type(torch.uint8)
        img = T.functional.posterize(img, self.bits)
        img = self.normalize(img.type(torch.float32)/255.)
        return img

class Solarize(BaseOp):

    def __init__(self, dm, val):
        super().__init__(dm)
        self.val = self.threshold = val

    def __call__(self, img, *args, **kwargs):
        img = self.unnormalize(img)
        img = T.functional.solarize(img, self.threshold)
        img = self.normalize(img)
        return img

class SolarizeAdd(BaseOp):

    def __init__(self, dm, val):
        super().__init__(dm)
        self.val = val
        self.solarize = Solarize(dm ,val)

    def __call__(self, img, *args, **kwargs):
        img = self.solarize(img) + img
        return img

class Contrast(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.adjust_contrast(img, self.val)
        return img

class Sharpness(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.adjust_sharpness(img, self.val)
        return img

class Brightness(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.adjust_brightness(img, self.val)
        return img

class Saturation(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.adjust_saturation(img, self.val)
        return img

class ShearX(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(self.val)), 0.0],
            center=[0, 0],
        )
        return img

class ShearY(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(self.val))],
            center=[0, 0],
        )
        return img

class TranslateX(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.affine(
            img,
            angle=0.0,
            translate=[int(self.val), 0],
            scale=1.0,
            shear=[0.0, 0.0],
            center=[0, 0],
        )
        return img

class TranslateY(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.affine(
            img,
            angle=0.0,
            translate=[0, int(self.val)],
            scale=1.0,
            shear=[0.0, 0.0],
            center=[0, 0],
        )
        return img

class Rotate(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def __call__(self, img, *args, **kwargs):
        img = T.functional.rotate(img, self.val)
        return img

class CropBilinear(BaseOp):

    def __init__(self, val):
        super().__init__()
        self.val = int(val)

    def __call__(self, img, *args, **kwargs):
        H, W = img.shape[-2:]
        return T.functional.resized_crop(
            img, 
            self.val, self.val, H-self.val, W-self.val, 
            size=[H, W], antialias=True)

class PIL_Blur(BaseOp):

    def __init__(self):
        super().__init__()
        self.kernel = torch.FloatTensor([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]).unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)
        self.kernel /= self.kernel.sum()


    def __call__(self, img, *args, **kwargs):
        self.kernel = self.kernel.to(img.device)
        return torch.nn.functional.conv2d(
            img, self.kernel, bias=None, padding='same')

class PIL_Smooth(BaseOp):

    def __init__(self):
        super().__init__()
        self.kernel = torch.FloatTensor([
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1]
        ]).unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)
        self.kernel /= self.kernel.sum()

    def __call__(self, img, *args, **kwargs):
        self.kernel = self.kernel.to(img.device)
        return torch.nn.functional.conv2d(
            img, self.kernel, bias=None, padding='same')


class Cutout(BaseOp):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length=None):
        super().__init__()
        self.n_holes = 1
        self.val = self.length = int(length)

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (N, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h, w = img.shape[-2:]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[..., y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).to(img.device)
        img = img * mask

        return img

class FiveCrops(BaseOp):

    def __init__(self, val, crop_height=224, crop_width=224):
        super().__init__()
        if val == 0:
            self.crop_fn = lambda x: ttach.functional.center_crop(
                x, crop_h=crop_height, crop_w=crop_width)
            self.val = 'center_crop'
        elif val == 1:
            self.crop_fn = lambda x: ttach.functional.crop_lt(
                x, crop_h=crop_height, crop_w=crop_width)
            self.val = 'crop_lt'
        elif val == 2:
            self.crop_fn = lambda x: ttach.functional.crop_lb(
                x, crop_h=crop_height, crop_w=crop_width)
            self.val = 'crop_lb'
        elif val == 3:
            self.crop_fn = lambda x: ttach.functional.crop_rt(
                x, crop_h=crop_height, crop_w=crop_width)
            self.val = 'crop_rt'
        elif val == 4:
            self.crop_fn = lambda x: ttach.functional.crop_rb(
                x, crop_h=crop_height, crop_w=crop_width)
            self.val = 'crop_rb'
        else:
            raise ValueError

    def __call__(self, image):
        return self.crop_fn(image)

class Scale(BaseOp):
    """Scale images
    """
    def __init__(self, val):
        super().__init__()
        self.scale = self.val = val

    def __call__(self, image):
        return ttach.functional.scale(
            image,
            self.scale,
            # interpolation=self.interpolation,
            # align_corners=self.align_corners,
        )

class PoolState(BaseOp):

    def __init__(self, state):
        super().__init__()
        self.val = state

    def __call__(self, image):
        return image

