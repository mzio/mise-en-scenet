"""
VGG16 for gram style and content embeddings and fully-connected encoder models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from collections import namedtuple


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        self.models = models.vgg16(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.vgg_pretrained_features = self.models.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        print('VGG feature extractor loaded!')

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

    def encode(self, X):
        X = self.vgg_pretrained_features(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        return self.models.classifier[:6](X)


class Encoder(nn.Module):
    def __init__(self, num_classes=13, style_dim=256):
        super(Encoder, self).__init__()
        self.fc_content = nn.Linear(4096, 1024)
        self.fc_style = nn.Linear(style_dim * style_dim, 1024)
        self.features = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes)
        )

    def forward(self, content, style):
        x1 = self.fc_content(content)  # to 1024
        style = torch.flatten(style, start_dim=1)
        x2 = self.fc_style(style)  # to 1024
        x = torch.cat((x1, x2), dim=1)
        x = self.features(x)
        return x


class BasicOneNet(nn.Module):
    def __init__(self, num_classes):
        super(BasicOneNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 224 * 224, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 224 * 224)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicNet(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(BasicNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(43264, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 120)
        self.fc4 = nn.Linear(120, num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        # print(x.shape)
        # x = x.view(-1, 64)
        x = self.fc4(x)
        # print(x.shape)
        return x

    def embed(self, x):
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

