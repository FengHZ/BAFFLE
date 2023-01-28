import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from model.stein_identity import SteinIdentity


class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features=16, small_input=True):
        super(_PreProcess, self).__init__()
        if small_input:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=3, stride=1, padding=1,
                                      bias=True))
        else:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                      bias=True))
            self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                  ceil_mode=False))


class _WideResUnit(nn.Module):
    def __init__(self, num_input_features, num_output_features, stride=1, drop_rate=0.3, activation=None,
                 normalization="GN"):
        super(_WideResUnit, self).__init__()
        self.f_block = nn.Sequential()
        if normalization == "GN":
            self.f_block.add_module('norm1', nn.GroupNorm(max(2, int(num_input_features / 16)), num_input_features,
                                                          affine=False))
        elif normalization == "BN":
            self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features, affine=False))
        elif normalization == "NoNorm":
            pass
        else:
            raise NotImplementedError("Normalization {} Not Implemented".format(normalization))
        self.f_block.add_module('relu1', activation)
        self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                   kernel_size=3, stride=stride, padding=1, bias=False))
        self.f_block.add_module('dropout', nn.Dropout(drop_rate))
        if normalization == "GN":
            self.f_block.add_module('norm2',
                                    nn.GroupNorm(max(2, int(num_output_features / 16)), num_output_features,
                                                 affine=False))
        elif normalization == "BN":
            self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features, affine=False))
        elif normalization == "NoNorm":
            pass
        else:
            raise NotImplementedError("Normalization {} Not Implemented".format(normalization))
        self.f_block.add_module('relu2', activation)
        self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                   kernel_size=3, stride=1, padding=1, bias=False))

        if num_input_features != num_output_features or stride != 1:
            self.i_block = nn.Sequential()
            if normalization == "GN":
                self.i_block.add_module('norm', nn.GroupNorm(max(2, int(num_input_features / 16)), num_input_features,
                                                             affine=False))
            elif normalization == "BN":
                self.i_block.add_module('norm', nn.BatchNorm2d(num_input_features, affine=False))
            elif normalization == "NoNorm":
                pass
            else:
                raise NotImplementedError("Normalization {} Not Implemented".format(normalization))
            self.i_block.add_module('relu', activation)
            self.i_block.add_module('conv',
                                    nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=stride,
                                              bias=False))

    def forward(self, x):
        new_features = self.f_block(x)
        if hasattr(self, "i_block"):
            x = self.i_block(x)
        return new_features + x


class _WideBlock(nn.Module):
    def __init__(self, input_channel, channel_width, block_depth, down_sample=False, drop_rate=0.0, activation=None,
                 normalization="GN"):
        super(_WideBlock, self).__init__()
        self.wide_block = nn.Sequential()
        for i in range(block_depth):
            if i == 0:
                unit = _WideResUnit(input_channel, channel_width, stride=int(1 + down_sample),
                                    drop_rate=drop_rate, activation=activation, normalization=normalization)
            else:
                unit = _WideResUnit(channel_width, channel_width, drop_rate=drop_rate, activation=activation,
                                    normalization=normalization)
            self.wide_block.add_module("wideunit%d" % (i + 1), unit)

    def forward(self, x):
        return self.wide_block(x)


class WideResNet(SteinIdentity):
    def __init__(self, num_input_channels=3, num_init_features=16, depth=10, width=12, num_classes=10,
                 data_parallel=False, small_input=True, drop_rate=0.0, K=200, sigma=1e-4, activation="ReLU",
                 normalization="GN", fd_format="forward"):
        super(WideResNet, self).__init__(K=K, sigma=sigma, fd_format=fd_format)
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
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        block_depth = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]
        self._widths = widths
        self.encoder = nn.Sequential()
        self.global_avg = nn.Sequential()
        pre_process = _PreProcess(num_input_channels, num_init_features, small_input=small_input)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        for idx, width in enumerate(widths):
            if idx == 0:
                wide_block = _WideBlock(num_init_features, width, block_depth, drop_rate=drop_rate,
                                        activation=activation, normalization=normalization)
            else:
                wide_block = _WideBlock(widths[idx - 1], width, block_depth, down_sample=True, drop_rate=drop_rate,
                                        activation=activation, normalization=normalization)
            if data_parallel:
                wide_block = nn.DataParallel(wide_block)
            self.encoder.add_module("wideblock%d" % (idx + 1), wide_block)
        global_avg = nn.AdaptiveAvgPool2d((1, 1))
        # we may use norm and relu before the global avg. Standard implementation doesn't use
        if normalization == "GN":
            self.global_avg.add_module("norm", nn.GroupNorm(int(widths[-1] / 16), widths[-1], affine=False))
        elif normalization == "BN":
            self.global_avg.add_module("norm", nn.BatchNorm2d(widths[-1], affine=False))
        elif normalization == "NoNorm":
            pass
        else:
            raise NotImplementedError("Normalization {} Not Implemented".format(normalization))
        self.global_avg.add_module('relu', activation)
        self.global_avg.add_module('avg', global_avg)
        if data_parallel:
            self.global_avg = nn.DataParallel(self.global_avg)
        classification = nn.Sequential()
        classification.add_module("fc", nn.Linear(widths[-1], num_classes))
        if data_parallel:
            classification = nn.DataParallel(classification)
        self.classification = classification

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param.data)
            elif 'conv' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize liner transform
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize the batch norm layer
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
        self.delta = OrderedDict()
        self.grad = OrderedDict()
        for name, p in self.named_parameters():
            self.delta[name] = torch.zeros(p.data.size()).cuda()
            self.grad[name] = 0

    def _forward(self, input_image, label):
        batch_size = input_image.size(0)
        features = self.encoder(input_image)
        avg_features = self.global_avg(features).view(batch_size, -1)
        cls_result = self.classification(avg_features)
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
