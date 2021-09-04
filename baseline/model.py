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


class EfficientNetB6(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            "efficientnet-b6", num_classes=18)

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


class imgInceptionV3(nn.Module):  # input size : (299,299)
    def __init__(self, num_classes):
        super(imgInceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        # self.inception_v3.fc = nn.Linear(2048, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return x


class denseNet(nn.Module):  # input size : (224,224)
    def __init__(self, num_classes):
        super(denseNet, self).__init__()
        self.model = models.densenet161(pretrained=True)
        # self.densenet.classifier = nn.Linear(2208, 18)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


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
