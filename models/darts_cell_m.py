import torch
import torch.nn as nn
# from models import darts_ops_step1_sc_out as ops
from models import darts_ops_step2_sc_out as ops
import torch.nn.functional as F
import numpy as np
import itertools

class InitCell(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # down sample and not down sample
        self.ops = nn.ModuleList()
        self.ops.append(nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout)
        ))
        self.ops.append(nn.Sequential(
            nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
            nn.BatchNorm2d(cout)
        ))

    def forward(self, x, sample):
        op = self.ops[int(np.argmax(sample))]
        return F.relu(op(x))


class NormCell(nn.Module):
    def __init__(self, cin, cout, n_blocks=5):
        super().__init__()
        self.ops = nn.ModuleList()
        self.block = ops.BasicBlock
        if cin == cout:
            self.ops.append(ops.Identity())
            for i in range(n_blocks):
                sub_ops = []
                for j in range(2**i):
                    sub_ops.append(self.block(cin, cout))
                self.ops.append(nn.Sequential(*sub_ops))
        else:
            self.ops.append(self.block(cin, cout))
            self.ops.append(self.block(cin, cout, stride=2))

    def forward(self, x, sample):
        op = self.ops[int(np.argmax(sample))]
        return op(x)


# training cell
class Training_Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, binary=False, sum_flag = True):
        super(Training_Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)i
        self.sum_flag = sum_flag
        genotype_reduce = list(itertools.chain.from_iterable(genotype.reduce))
        genotype_normal = list(itertools.chain.from_iterable(genotype.normal))
        if reduction_prev:
            self.preprocess0 = ops.FactorizedReduce(C_prev_prev, C, skip=True, binary=binary)
            self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0, binary=binary, weight_binary=True)
        else:
            self.preprocess0 = ops.ReLUConvBN(C_prev_prev, C, 1, 1, 0, binary=binary, weight_binary=True)
            #print(C,'@@@')
            self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0, binary=binary, weight_binary=True)
        #print(C,'###')
        if reduction:
            op_names, indices = zip(*genotype_reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype_normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, binary=binary)

    def _compile(self, C, op_names, indices, concat, reduction, binary=True, ts_flag=False):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = ops.OPS[name](C, stride, True, binary=binary, ts_flag=ts_flag)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, ops.Identity):
                    h1 = ops.drop_path_(h1, drop_prob, self.training)
                if not isinstance(op2, ops.Identity):
                    h2 = ops.drop_path_(h2, drop_prob, self.training)
            s = h1 + h2
            states += [s]
        #print(self._concat)
        if self.sum_flag:
            states_sum = torch.zeros_like(states[self._concat[0]])
            temp_count = 0
            for i in self._concat:
                temp_count += 1            
                states_sum += states[i]
            states_sum /= temp_count
        #print(states_sum.shape)
        #print(torch.cat([states[i] for i in self._concat],dim=1).shape)
        #print(torch.sum(torch.cat([states[i] for i in self._concat],dim=1),dim=1).shape)
            return states_sum#torch.sum(torch.cat([states[i] for i in self._concat], dim=1), dim=1)
        else:
            return torch.cat([states[i] for i in self._concat], dim=1)
        #return torch.sum([states[i] for i in self._concat], dim=1)
