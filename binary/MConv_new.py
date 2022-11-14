# coding=utf-8
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from .utils import _pair
from torch.nn.modules.conv import _ConvNd
from .MFunction_new import MCF_Function
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

class MConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, M=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, binary=False, group_flag=True, ts_flag=False):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

        self.generate_MFilters(kernel_size)
        self.MCF_Function = MCF_Function.apply
        self.out_channels = out_channels
        self.binary = binary
        if self.binary:
            self.binfunc = BinaryActivation()
        
    def generate_MFilters(self, kernel_size):
        self.MFilters = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):
        # inplace op
        if self.binary:
            x = self.binfunc(x)
        #print(self.weight, self.MFilters)
        new_weight = self.MCF_Function(self.weight, self.MFilters)
        #pdb.set_trace()
        return F.conv2d(x, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

