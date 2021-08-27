import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet

class Way3_small(nn.Module):
    def __init__(self, config):
        super().__init__()
#         self.convnet = timm.create_model(config['model'], pretrained=True, num_classes=512)
        self.config = config
        self.convnet = EfficientNet.from_pretrained(self.config['architecture'], num_classes=512)
        self.mask = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        self.gender = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        self.age = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        self.best_f1 = 0
    def forward(self, x):
        features = self.convnet(x)
        mask = self.mask(features)
        gender = self.gender(features)
        age = self.age(features)
        
        return mask, gender, age
    
class Way3_Large(nn.Module):
    def __init__(self, config):
        super().__init__()
#         self.convnet = timm.create_model(config['model'], pretrained=True, num_classes=512)
        self.config = config
        self.convnet = EfficientNet.from_pretrained(self.config['architecture'], num_classes=512)
        self.mask = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        self.gender = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        self.age = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        self.best_f1 = 0
        
    def forward(self, x):
        features = self.convnet(x)
        mask = self.mask(features)
        gender = self.gender(features)
        age = self.age(features)
        
        return mask, gender, age