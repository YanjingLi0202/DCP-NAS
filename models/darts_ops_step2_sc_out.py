""" Operations """
import torch
import torch.nn as nn
from utils import genotypes_m as gt
import numpy as np
import torch.nn.functional as F
from binary.MConv_new import MConv
# from binary.BConv import HardBinaryConv as MConv
# M = 2

OPS = {
    'none': lambda C, stride, affine, binary, ts_flag: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, binary, ts_flag: PoolBN('avg', C, 3, stride, 1, affine=affine, binary=binary),
    'max_pool_3x3': lambda C, stride, affine, binary, ts_flag: PoolBN('max', C, 3, stride, 1, affine=affine, binary=binary),
    'skip_connect': lambda C, stride, affine, binary, ts_flag: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, binary=binary),
    'sep_conv_3x3': lambda C, stride, affine, binary, ts_flag: SepConv(C, C, 3, stride, 1, affine=affine, binary=binary, ts_flag=ts_flag),
    'sep_conv_5x5': lambda C, stride, affine, binary, ts_flag: SepConv(C, C, 5, stride, 2, affine=affine, binary=binary, ts_flag=ts_flag),
    'sep_conv_7x7': lambda C, stride, affine, binary, ts_flag: SepConv(C, C, 7, stride, 3, affine=affine, binary=binary, ts_flag=ts_flag),
    'dil_conv_3x3': lambda C, stride, affine, binary, ts_flag: DilConv(C, C, 3, stride, 2, 2, affine=affine, binary=binary, ts_flag=ts_flag), # 5x5
    'dil_conv_5x5': lambda C, stride, affine, binary, ts_flag: DilConv(C, C, 5, stride, 4, 2, affine=affine, binary=binary, ts_flag=ts_flag), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine, binary, ts_flag: FacConv(C, C, 7, stride, 3, affine=affine, binary=binary, ts_flag=ts_flag)
}


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, binary=False, weight_binary=True):
        super(ReLUConvBN, self).__init__()
        self.binary = binary
        #if not self.binary:
        #    self.relu = nn.ReLU()
        
        if weight_binary:
            self.op = nn.Sequential(
                nn.BatchNorm2d(C_in, affine=affine),
                #LearnableBias(C_in),
                MConv(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, binary=False, group_flag=True),
                #nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out,eps=0.001, affine=affine),
                nn.PReLU(C_out)
            )
        else:
            self.op = nn.Sequential(
                nn.BatchNorm2d(C_in, affine=affine),
                MConv(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, binary=False, group_flag=True),#)=binary),
                #nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, eps=0.001, affine=affine),
                nn.PReLU(C_out)
            )
        
        # self.relu = nn.PReLU(C_out)
        

    def forward(self, x):
        
        return self.op(x)


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True, binary=False):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C,eps=0.001, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, binary=False, ts_flag=False):
        super().__init__()
        self.binary = binary
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(C_in, affine=affine),
            #LearnableBias(C_in),
            MConv(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, binary=binary, ts_flag=ts_flag),
            
            # nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out,eps=0.001, affine=affine),
            nn.PReLU(C_out)
        )


        # self.relu = nn.PReLU(C_out)

    def forward(self, x):
        #if not self.binary:
        #    x = self.relu(x)
            # print('2')
        #print("0 std")

        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True, binary=False, ts_flag=False):
        super().__init__()
        self.binary = binary
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(C_in, affine=affine),
            #LearnableBias(C_in),
            MConv(C_in, C_in, (kernel_length, 1), stride=stride, padding=padding, bias=False, binary=binary, ts_flag=ts_flag),
            nn.BatchNorm2d(C_in,eps=0.001, affine=affine),
            nn.PReLU(C_in),
            nn.BatchNorm2d(C_in, affine=affine),
            #LearnableBias(C_in),
            MConv(C_in, C_out, (1, kernel_length), stride=stride, padding=padding, bias=False, binary=binary, ts_flag=ts_flag),
            nn.BatchNorm2d(C_out,eps=0.001, affine=affine),
            nn.PReLU(C_out)
        )

        # self.net = nn.Sequential(
        #     nn.ReLU(),
        #     MConv(C_in, C_in, (kernel_length, 1), M=M, stride=stride, padding=padding, bias=False, binary=binary),
        #     MConv(C_in, C_out, (1, kernel_length), M=M, stride=stride, padding=padding, bias=False, binary=binary),
        #     nn.BatchNorm2d(C_out * M, affine=affine)
        # )
    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, binary=True, ts_flag=False):
        super().__init__()
        self.binary = binary
        #if not self.binary:
        #    self.relu = nn.ReLU()
        #self.net = nn.Sequential(
            #nn.Conv2d(C_in * M, C_in * M, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            #nn.BatchNorm2d(C_in * M, affine=affine),
        #self.move0 = LearnableBias(C_in)
        
        self.bn0 = nn.BatchNorm2d(C_in,eps=0.001, affine=affine)
        self.conv1 = MConv(C_in, C_in, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False, binary=False, group_flag=True, ts_flag=ts_flag)
        self.bn1 = nn.BatchNorm2d(C_in,eps=0.001, affine=affine)
        self.relu0 = nn.PReLU(C_in)
        #self.move1 = LearnableBias(C_in)
        self.conv2 = MConv(C_in, C_out, 1, stride=1, padding=0, bias=False, binary=True, ts_flag=ts_flag)
            # MConv(C_in, C_out, kernel_size, M=M, stride=stride, padding=padding, dilation=dilation, groups=1,
            #         bias=False, binary=binary),

        self.bn2 = nn.BatchNorm2d(C_out,eps=0.001, affine=affine)
        self.relu1 = nn.PReLU(C_out)
        self.stride = stride
        #)
        
        if self.stride == 1: print('okay----sc') 

        # self.net = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(C_in * M, C_in * M, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in,
        #               bias=False),

        #     # MConv(C_in, C_in, kernel_size, M=M, stride=stride, padding=padding, dilation=dilation, groups=C_in,
        #     #         bias=False, binary=binary),
        #     MConv(C_in, C_out, 1, M=M, stride=1, padding=0, bias=False, binary=binary),

        #     # MConv(C_in, C_out, kernel_size, M=M, stride=stride, padding=padding, dilation=dilation, groups=1,
        #     #         bias=False, binary=binary),

        #     nn.BatchNorm2d(C_out * M, affine=affine)
        # )
    def forward(self, x):
        #if not self.binary:
        #    x = self.relu(x)
            # print('4')
        # x = self.net(x)
        # print(x[0,0])
        #print("0 dil")
        res = x
        #x = self.move0(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        # x = res + x if self.stride == 1 else x
        x = self.relu0(x)
        
        #x = self.move1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = res + x if self.stride == 1 else x
        return self.relu1(x) # x # 


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, binary=False, ts_flag=False):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine, binary=binary, ts_flag=ts_flag),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine, binary=binary, ts_flag=ts_flag)
        )

    def forward(self, x):
        #print("0")
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True, binary=False, skip=True):
        super().__init__()
        self.binary = binary
        
        skip = True
        self.bn0 = nn.BatchNorm2d(C_in, affine=affine)
        if skip:
            self.conv1 = MConv(C_in, C_out//2, 1, stride=2, padding=0, bias=False, binary=False, group_flag=True)
            self.conv2 = MConv(C_in, C_out//2, 1, stride=2, padding=0, bias=False, binary=False, group_flag=True)
        else:
            self.conv1 = nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
            self.conv2 = nn.Conv2d(C_in, C_out//2, 1, stride=2, padding=0, bias=False)
            
        self.bn = nn.BatchNorm2d(C_out,eps=0.001, affine=affine)
        self.relu = nn.PReLU(C_out)

    def forward(self, x):
        #if not self.binary:
        #    x = self.relu(x)
            # print('5')i
        x1 = self.bn0(x)
        #print("1 pre")
        out = torch.cat([self.conv1(x1), self.conv2(x1[:, :, 1:, 1:].contiguous())], dim=1)
        out = self.bn(out)
        # out += self.shortcut(x)
        return self.relu(out)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SelectOp_PC(nn.Module):
    """ Mixed PC operation """

    def __init__(self, C, stride):
        super(SelectOp_PC, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2,2)

        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C,eps=0.001, affine=False))
            self._ops.append(op)


    def forward(self, x, weights):
        #channel proportion k=4  
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2, :, :]
        xtemp2 = x[ : ,  dim_2:, :, :]
        # temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        temp1 = self._ops[weights](xtemp)
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
        else:
            ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, 1)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class SelectOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride, binary=False, ts_flag=False):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False, binary=binary, ts_flag=ts_flag)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        op = self._ops[weights]
        return op(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, binary=False):
        super(BasicBlock, self).__init__()
        self.conv1 = MConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, binary=binary)
        self.bn1 = nn.BatchNorm2d(planes,eps=0.001)
        self.conv2 = MConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, binary=binary)
        self.bn2 = nn.BatchNorm2d(planes,eps=0.001)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, binary=binary),
                nn.BatchNorm2d(self.expansion*planes,eps=0.001)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
