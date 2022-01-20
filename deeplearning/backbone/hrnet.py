# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/1/19 9:25 PM
# @Organization: YQN
# @Email: mlshenkai@163.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=(1, 1),
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(residual)
        out += residual
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            out_planes, out_planes * self.expansion, kernel_size=(1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(residual)
        out += residual
        out = self.act(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_in_channels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_in_channels, num_channels
        )

        self.num_in_channels = num_in_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    @staticmethod
    def _check_branches(
        num_branches, blocks, num_blocks, num_in_channels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_in_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_in_channels)
            )
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        down_sample = None
        if (
            stride != 1
            or self.num_in_channels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            down_sample = nn.Sequential(
                nn.Conv2d(
                    self.num_in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = [
            block(
                self.num_in_channels[branch_index],
                num_channels[branch_index],
                stride,
                down_sample,
            )
        ]
        self.num_in_channels[branch_index] = (
            num_channels[branch_index] * block.expansion
        )
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_in_channels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_in_channels[j],
                                num_in_channels[i],
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 0),
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_in_channels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_channels_conv3x3 = num_in_channels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_in_channels[j],
                                        num_out_channels_conv3x3,
                                        kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_out_channels_conv3x3),
                                )
                            )
                        else:
                            num_out_channels_conv3x3 = num_in_channels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_in_channels[j],
                                        num_out_channels_conv3x3,
                                        kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding=(1, 1),
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_out_channels_conv3x3),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_channels(self):
        return self.num_in_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


