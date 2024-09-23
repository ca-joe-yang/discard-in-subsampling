import torch
import copy

from modules.layers import PoolSearchConv2d

class ToyModel(torch.nn.Module):
    """
    Define a toy model with a single convolutional layer of subsampling rate = R = stride.
    """
    def __init__(self, in_channel=3, out_channel=1, kernel_size=3, R=2):
        super().__init__()
        self.R = R
        self.conv = torch.nn.Conv2d(
            in_channel, out_channel, 
            kernel_size=kernel_size, stride=R, 
            padding=(kernel_size-1)//2, bias=None)
        
    def forward(self, x):
        x = self.conv(x)
        return x

    def forward_all_states(self, x):
        self.conv.stride = 1
        x = self.conv(x)
        self.conv.stride = self.R
        return x

class ModifiedToyModel(torch.nn.Module):
    """
    Replace the convolutional layer with our modified module ``PoolSearchConv2d''
    """
    def __init__(self, model):
        super().__init__()
        self.R = model.R
        self.modified_conv = PoolSearchConv2d(
            copy.deepcopy(model.conv))

    def forward(self, x):
        x = self.modified_conv(x)
        return x

if __name__ == '__main__':
    R = 2 # subsampling rate
    x = torch.rand(size=[1, 3, 8, 8])
    model = ToyModel(in_channel=x.shape[1], R=R)

    # tensor y is the standard activation
    y_standard = model.forward(x)

    # tensor y_all_states contains the standard activation and the discarded activations
    y_all_states = model.forward_all_states(x)

    modified_model = ModifiedToyModel(model)
    for s in range(R * R):
        dx = s % R
        dy = s // R

        # specify the state we want
        modified_model.modified_conv.set_select_indices([s])
        ys = modified_model(x)

        # assert that each ys we get from our modified module equals to each supposedly discarded activation
        if s == 0:
            print(torch.allclose(y_standard, ys))
        print(torch.allclose(
            y_all_states[..., dy::R, dx::R], ys))
