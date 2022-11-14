# coding=utf-8
from __future__ import division
import math
import torch
import torch.nn as nn
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
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        # self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        #real_weights = self.weights.view(self.shape)
        if self.expand:
            x = self.do_expanding(x)
        if self.binary:
            x = self.binfunc(x)

        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        new_bias = self.expand_bias(self.bias) if self.need_bias else self.bias
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, new_bias, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)

        return y

    def do_expanding(self, x):
        index = []
        for i in range(x.size(1)):
            for _ in range(self.M):
                index.append(i)
        index = torch.LongTensor(index).cuda() if x.is_cuda else torch.LongTensor(index)
        return x.index_select(1, index)

    def expand_bias(self, bias):
        index = []
        for i in range(bias.size(0)):
            for _ in range(self.M):
                index.append(i)
        index = torch.LongTensor(index).cuda() if bias.is_cuda else torch.LongTensor(index)
        return bias.index_select(0, index)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', binary={binary}'
        return s.format(**self.__dict__)
