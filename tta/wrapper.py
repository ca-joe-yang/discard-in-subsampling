from typing import Mapping, Union
import numpy as np

import torch
import torch.nn as nn

class WrapperTTA(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_label_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:

        augmented_logits = []
        for transform in self.transforms:
            augmented_image = transform(image)
            try:
                state = np.array(transform.ops[-1].val)
                logits = self.model.forward_state(
                    augmented_image, state, ret_logits=True)[:, 0]
            except:
                logits = self.model.forward(augmented_image)
            augmented_logits.append(logits)

        return torch.stack(augmented_logits, 1)

    def forward_features(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:

        augmented_feats = []
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            feats = self.model.forward_features(augmented_image, *args)
            augmented_feats.append(feats)

        return torch.stack(augmented_feats, 1)

    def forward_state(
        self, image: torch.Tensor, state
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:

        augmented_logits = []
        for transform in self.transforms:
            augmented_image = transform(image)
            feats, logits = self.model.forward_state(
                augmented_image, state)
            augmented_logits.append(feats[:, 0])

        return torch.stack(augmented_logits, 1)
