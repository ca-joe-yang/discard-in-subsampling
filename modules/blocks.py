from .base import BaseModule
from .layers import *
from .register import register

@register
class PoolSearchBasicBlock(BaseModule):

    def __init__(self, block, version='base'):
        super().__init__()
        
        self.block = block
        if version == 'antialias':
            self.block.conv2[0] = PoolSearchBlurPool(self.block.conv2[0])
            self.block.downsample[0] = PoolSearchBlurPool(self.block.downsample[0])
            
            self.child_modules = [
                self.block.conv2, 
                self.block.downsample[0]]
        else:
            self.block.conv1 = PoolSearchConv2d(self.block.conv1)
            self.block.downsample[0] = PoolSearchConv2d(self.block.downsample[0])
            
            self.child_modules = [
                self.block.conv1, 
                self.block.downsample[0]]
        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        return self.block(x)

@register
class PoolSearchBottleNeck(BaseModule):

    def __init__(self, block, version='base'):
        super().__init__()

        self.block = block

        if version == 'antialias':
            self.block.conv3[0] = PoolSearchBlurPool(self.block.conv3[0])
            self.block.downsample[0] = PoolSearchBlurPool(self.block.downsample[0])
                
            self.child_modules = [
                self.block.conv3[0], 
                self.block.downsample[0]]
        else:
            self.block.conv2 = PoolSearchConv2d(self.block.conv2)
            self.block.downsample[0] = PoolSearchConv2d(self.block.downsample[0])
            
            self.child_modules = [
                self.block.conv2, 
                self.block.downsample[0]]
        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        return self.block(x)


@register
class PoolSearchClipBottleNeck(BaseModule):

    def __init__(self, block):
        super().__init__()

        self.block = block
        self.block.avgpool = PoolSearchAvgPool2d(self.block.avgpool)
        self.block.downsample[0] = PoolSearchAvgPool2d(self.block.downsample[0])
        

        self.child_modules = [
            self.block.avgpool,
            self.block.downsample[0]]
        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        return self.block(x)

@register
class PoolSearchRepVggBlock(BaseModule):

    def __init__(self, block, version='cifar'):
        super().__init__()

        self.block = block
        if version == 'cifar':
            self.block.rbr_dense.conv = PoolSearchConv2d(self.block.rbr_dense.conv)
            self.block.rbr_1x1.conv = PoolSearchConv2d(self.block.rbr_1x1.conv)
            self.child_modules = [
                self.block.rbr_dense.conv, 
                self.block.rbr_1x1.conv]
        elif version == 'timm':
            self.block.conv_kxk.conv = PoolSearchConv2d(self.block.conv_kxk.conv)
            self.block.conv_1x1.conv = PoolSearchConv2d(self.block.conv_1x1.conv)
            self.child_modules = [
                self.block.conv_kxk.conv, 
                self.block.conv_1x1.conv]
        else:
            raise ValueError

        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        return self.block(x)

@register
class PoolSearchInceptionB(BaseModule):
    
    def __init__(self, block, **kwargs):
        super().__init__()
    
        self.block = block
        self.block.branch3x3.conv = PoolSearchConv2d(self.block.branch3x3.conv)
        self.block.branch3x3dbl_3.conv = PoolSearchConv2d(self.block.branch3x3dbl_3.conv)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool = PoolSearchMaxPool2d(maxpool)
        self.child_modules = [
            self.block.branch3x3.conv, 
            self.block.branch3x3dbl_3.conv, 
            self.maxpool]
        
        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        branch3x3 = self.block.branch3x3(x)

        branch3x3dbl = self.block.branch3x3dbl_1(x)
        branch3x3dbl = self.block.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.block.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.maxpool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

@register
class PoolSearchInceptionD(BaseModule):
    
    def __init__(self, block, **kwargs):
        super().__init__()
    
        self.block = block
        self.block.branch3x3_2.conv = PoolSearchConv2d(self.block.branch3x3_2.conv)
        self.block.branch7x7x3_4.conv = PoolSearchConv2d(self.block.branch7x7x3_4.conv)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool = PoolSearchMaxPool2d(maxpool)
        self.child_modules = [
            self.block.branch3x3_2.conv, 
            self.block.branch7x7x3_4.conv, 
            self.maxpool]
        
        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        branch3x3 = self.block.branch3x3_1(x)
        branch3x3 = self.block.branch3x3_2(branch3x3)

        branch7x7x3 = self.block.branch7x7x3_1(x)
        branch7x7x3 = self.block.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.block.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.block.branch7x7x3_4(branch7x7x3)

        branch_pool = self.maxpool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

@register
class PoolSearchInvertedResidual(BaseModule):
    def __init__(self, block, **kwargs):
        super().__init__()

        self.block = block
        self.block.branch1[0] = PoolSearchConv2d(self.block.branch1[0])
        self.block.branch2[3] = PoolSearchConv2d(self.block.branch2[3])
        self.child_modules = [
            self.block.branch1[0], 
            self.block.branch2[3]]
        
        self.num_expand = self.child_modules[0].num_expand
        self.R = self.child_modules[0].R

    def forward(self, x):
        return self.block(x)