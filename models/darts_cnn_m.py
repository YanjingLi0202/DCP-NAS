from models.darts_cell_m import *
import utils.genotypes_m as gt
import numpy as np
# from scipy.special import softmax
from models.base_module import MyModule, BaseSearchModule
import itertools
import random
import copy
import torch
import torch.nn.functional as F
# from binary.BConv import HardBinaryConv as MConv
from binary.MConv_new import MConv

class NetworkImageNet(MyModule):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, drop_out=0, drop_path=0, binary=False):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._dropout = drop_out
        self.drop_path_prob = drop_path


        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        self.sum_flag = False
        for i in range(layers):
            if i == layers - 1:
                self.sum_flag = False
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Training_Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, binary=binary, sum_flag = self.sum_flag)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev = C_prev
            C_prev = C_curr if self.sum_flag else cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes, binary=binary)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self._dropout > 0:
            self.dropout = nn.Dropout(p=self._dropout)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        if self._dropout > 0:
            out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
