import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# 준태


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            "efficientnet-b4", num_classes=18)

    def forward(self, x):
        return self.backbone(x)

# 재현


class EfficientNetB5(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            "efficientnet-b5", num_classes=18)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            "efficientnet-b7", num_classes=18)

    def forward(self, x):
        return self.backbone(x)

# 광채2


class EfficientNetB8(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.net = EfficientNet.from_pretrained(
            'efficientnet-b8', num_classes=1024, advprop=True)
        self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.LeakyReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 18))

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


_model_entrypoints = {
    'model1': EfficientNetB8,
    'model2': EfficientNetB7,
    'model3': EfficientNetB5,
    'model4': EfficientNetB4,
    'model5': EfficientNetB4
}


def model_entrypoint(model_name):
    return _model_entrypoints[model_name]


def is_model(model_name):
    return model_name in _model_entrypoints


def create_model(model_name, **kwargs):
    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)
    return model
