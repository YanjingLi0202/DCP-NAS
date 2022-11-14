# coding=utf-8
from __future__ import division
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
#from .utils import _pair
from torch.nn.modules.conv import _ConvNd
from .MFunction import MCF_Function
import pdb
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(_ConvNd):
    '''
        Bi-realNet binary
        '''

    def __init__(self, in_channels, out_channels, kernel_size=3, M=1, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, expand=False, binary=True,
                group_flag=True, ts_flag=False):

        kernel_temp = kernel_size
        kernel_size =_pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(HardBinaryConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')  # ,

        self.expand = expand
        self.need_bias = bias
        self.M = M
        self.binary = binary
        if self.binary:
            self.binfunc = BinaryActivation()
        # self.number_of_weights = in_channels * out_channels * kernel_temp * kernel_temp
        # self.shape = (out_channels, in_channels, kernel_temp, kernel_temp)
        # self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        # self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def _conv_forward(self, input, weight):
        if self.binary:
            input = self.binfunc(input)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight)
