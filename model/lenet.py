import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stein_identity import SteinIdentity


class LeNet(SteinIdentity):
    def __init__(self, input_channel=3, num_classes=10, channels=(16, 64), K=200, sigma=1e-4, activation="ReLU",
                 normalization="GN", fd_format="forward"):
        super(LeNet, self).__init__(K=K, sigma=sigma, fd_format=fd_format)
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1",
                                nn.Conv2d(in_channels=input_channel, out_channels=channels[0], kernel_size=5, stride=1,
                                          padding=2,
                                          bias=True))
        if normalization == "GN":
            self.encoder.add_module('norm1', nn.GroupNorm(max(2, int(channels[0] / 16)), channels[0], affine=False))
        elif normalization == "BN":
            self.encoder.add_module('norm1', nn.BatchNorm2d(num_features=channels[0], affine=False))
        elif normalization == "NoNorm":
            pass
        else:
            raise NotImplementedError("Normalization {} Not Implemented".format(normalization))
        if activation == "Hardswish":
            activation = nn.Hardswish(inplace=True)
        elif activation == "GELU":
            activation = nn.GELU()
        elif activation == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation == "SELU":
            activation = nn.SELU(inplace=True)
        else:
            raise NotImplementedError("Activation {} not implemented".format(activation))
        self.encoder.add_module("relu1", activation)
        self.encoder.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder.add_module("conv2",
                                nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=5, stride=1,
                                          padding=2,
                                          bias=True))
        if normalization == "GN":
            self.encoder.add_module('norm2', nn.GroupNorm(max(2, int(channels[1] / 16)), channels[1], affine=False))
        elif normalization == "BN":
            self.encoder.add_module('norm2', nn.BatchNorm2d(num_features=channels[1], affine=False))
        elif normalization == "NoNorm":
            pass
        else:
            raise NotImplementedError("Normalization {} Not Implemented".format(normalization))
        self.encoder.add_module("relu2", activation)
        self.encoder.add_module("globalavg", nn.AdaptiveAvgPool2d((1, 1)))
        classification = nn.Sequential()
        classification.add_module("fc", nn.Linear(channels[1], num_classes))
        self.classification = classification

    def _forward(self, input_image, label):
        batch_size = input_image.size(0)
        features = self.encoder(input_image).view(batch_size, -1)
        cls_result = self.classification(features)
        loss = F.cross_entropy(cls_result, label)
        return cls_result, loss

    def forward(self, input_image, label, estimate_grad=True):
        if estimate_grad:
            with torch.no_grad():
                cls_result, loss = self._forward(input_image, label)
            self._stein_estimation(input_image, label, loss)
        else:
            cls_result, loss = self._forward(input_image, label)
        return cls_result, loss
