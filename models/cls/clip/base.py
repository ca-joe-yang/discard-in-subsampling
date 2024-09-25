import torch
import torch.nn as nn
import torchvision
import clip

from ..register import register_model

class ZeroShotCLIP(nn.Module):

    def __init__(self, variant, dataset, device='cuda'):
        super().__init__()
        if dataset == 'imagenet':
            from .imagenet import templates, classes
            self.templates = templates
            self.classes = classes
            self.num_classes = len(classes)
        else:
            raise ValueError(dataset)

        self.clip_model, clip_preprocess = clip.load(variant)
        self.clip_model.to(device)
        self.logit_scale = 100
        self.device = device
        self.build_classifier_weights()
        self.clip_model = self.clip_model.float()

    def build_classifier_weights(self):
        self.clip_model.eval()
        clip_weights = []
        for i, classname in enumerate(self.classes):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in self.templates]
            with torch.no_grad():
                texts = clip.tokenize(texts).cuda()
                class_embeddings = self.clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                clip_weights.append(class_embedding)

        self.clip_classifier_weights = torch.stack(clip_weights, dim=1).to(self.device).float()



def resnet_base(model):
    member_fns = {}

    def forward_features(self, x):
        visual = self.clip_model.visual
        def stem(x):
            x = visual.relu1(visual.bn1(visual.conv1(x)))
            x = visual.relu2(visual.bn2(visual.conv2(x)))
            x = visual.relu3(visual.bn3(visual.conv3(x)))
            x = visual.avgpool(x)
            return x

        x = stem(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)
        return x
    member_fns[forward_features.__name__] = forward_features

    def forward_head(self, x):
        visual = self.clip_model.visual
        x = visual.attnpool(x)
        x = x / x.norm(dim=-1, keepdim=True)
        logits = 100. * x @ self.clip_classifier_weights
        return logits
    member_fns[forward_head.__name__] = forward_head

    return model, member_fns

@register_model('clip')
def resnet50(dataset):
    model = ZeroShotCLIP('RN50', dataset)
    model.feat_dim = 2048
    model.downsample_layers = [
        ['clip_model/visual/conv1', 'Conv2d'],
        ['clip_model/visual/avgpool', 'AvgPool2d'],
        ['clip_model/visual/layer2/0', 'ClipBottleNeck'],
        ['clip_model/visual/layer3/0', 'ClipBottleNeck'],
        ['clip_model/visual/layer4/0', 'ClipBottleNeck'],
    ]
    return resnet_base(model)

