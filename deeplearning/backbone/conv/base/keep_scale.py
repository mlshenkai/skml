# -*- coding: utf-8 -*-
# @Author: Kai Shen
# @Created Time: 2022/1/21 11:19 AM
# @Organization: YQN
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import functools
import operator
from torch.nn.common_types import _size_2_t


class Conv2dSamePadding(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2dSamePadding, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride,int):
            stride = (stride, stride)

