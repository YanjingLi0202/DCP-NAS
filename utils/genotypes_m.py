""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch
import torch.nn as nn
from models import darts_ops_step1_sc_out as ops


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        # edge_max, primitive_indices = torch.topk(edges, 1)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def parse_numpy(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    # assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        # edge_max, primitive_indices = torch.topk(torch.tensor(edges), 1)
        edge_max, primitive_indices = torch.topk(torch.tensor(edges[:, :-1]), 1) # ignore 'none'
        # edge_max, primitive_indices = torch.topk(edges[:, :-1].clone().detach(), 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


NASNet = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
        [('avg_pool_3x3', 1), ('skip_connect', 0)],
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 0)],
        [('sep_conv_3x3', 1), ('skip_connect', 1)],
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_7x7', 0)],
        [('max_pool_3x3', 1), ('sep_conv_7x7', 0)],
        [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)],
        [('skip_connect', 3), ('avg_pool_3x3', 2)],
        [('sep_conv_3x3', 2), ('max_pool_3x3', 1)],
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)],
        [('sep_conv_3x3', 0), ('avg_pool_3x3', 3)],
        [('sep_conv_3x3', 1), ('skip_connect', 1)],
        [('skip_connect', 0), ('avg_pool_3x3', 1)],
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        [('avg_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('max_pool_3x3', 0), ('sep_conv_7x7', 2)],
        [('sep_conv_7x7', 0), ('avg_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('conv_7x1_1x7', 0), ('sep_conv_3x3', 5)],
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
        [('skip_connect', 0), ('sep_conv_3x3', 1)],
        [('skip_connect', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('skip_connect', 2)],
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('skip_connect', 2), ('max_pool_3x3', 0)],
        [('max_pool_3x3', 0), ('skip_connect', 2)],
        [('skip_connect', 2), ('avg_pool_3x3', 0)]
    ],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 1), ('skip_connect', 0)],
        [('skip_connect', 0), ('dil_conv_3x3', 2)],
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('skip_connect', 2), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('skip_connect', 2)],
        [('skip_connect', 2), ('max_pool_3x3', 1)],
    ],
    reduce_concat=[2, 3, 4, 5])

MDENAS = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('skip_connect', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 3), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 4)],
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('sep_conv_3x3', 3), ('skip_connect', 2)],
        [('dil_conv_3x3', 3), ('sep_conv_5x5', 0)],
    ],
    reduce_concat=range(2, 6))
DDPNAS_1 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 2), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('avg_pool_3x3', 2), ('max_pool_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_3x3', 0), ('avg_pool_3x3', 3)],
        [('max_pool_3x3', 1), ('sep_conv_3x3', 0)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_2 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('skip_connect', 1)],
        [('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 1), ('skip_connect', 2)],
        [('skip_connect', 1), ('sep_conv_3x3', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 2), ('avg_pool_3x3', 0)],
        [('avg_pool_3x3', 0), ('dil_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('max_pool_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_3 = Genotype(
    normal=[
        [('dil_conv_3x3', 1), ('sep_conv_5x5', 0)],
        [('max_pool_3x3', 2), ('skip_connect', 1)],
        [('dil_conv_3x3', 1), ('sep_conv_3x3', 2)],
        [('max_pool_3x3', 1), ('avg_pool_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('avg_pool_3x3', 1), ('dil_conv_3x3', 0)],
        [('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
        [('max_pool_3x3', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 4), ('dil_conv_5x5', 3)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_4 = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('skip_connect', 0)],
        [('avg_pool_3x3', 1), ('skip_connect', 0)],
        [('max_pool_3x3', 3), ('skip_connect', 2)],
        [('sep_conv_3x3', 4), ('max_pool_3x3', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 1), ('skip_connect', 0)],
        [('max_pool_3x3', 3), ('dil_conv_5x5', 1)],
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 3)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_5 = Genotype(
    normal=[
        [('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_3x3', 0), ('skip_connect', 1)],
        [('sep_conv_3x3', 3), ('max_pool_3x3', 1)],
        [('sep_conv_5x5', 3), ('sep_conv_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_5x5', 2), ('dil_conv_3x3', 1)],
        [('sep_conv_3x3', 3), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_PLANT = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('skip_connect', 0)],
        [('dil_conv_3x3', 2), ('skip_connect', 0)],
        [('max_pool_3x3', 3), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 1), ('max_pool_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('skip_connect', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_5x5', 1), ('dil_conv_3x3', 0)],
        [('skip_connect', 1), ('skip_connect', 2)],
        [('max_pool_3x3', 4), ('sep_conv_3x3', 0)]
    ],
    reduce_concat=range(2, 6))

DDP_PC_1 = Genotype( # bs=1024
    normal=[
        [('dil_conv_3x3', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_5x5', 2), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 3)],
        [('sep_conv_5x5', 2), ('dil_conv_5x5', 1)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('avg_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('dil_conv_3x3', 1), ('skip_connect', 3)],
        [('max_pool_3x3', 4), ('sep_conv_5x5', 3)]
    ],
    reduce_concat=range(2, 6))

DDP_PC_2 = Genotype( # bs=512
    normal=[
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('max_pool_3x3', 2), ('dil_conv_3x3', 1)],
        [('dil_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('max_pool_3x3', 2), ('avg_pool_3x3', 1)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('dil_conv_3x3', 0)],
        [('dil_conv_3x3', 0), ('sep_conv_5x5', 1)],
        [('max_pool_3x3', 2), ('dil_conv_5x5', 1)],
        [('sep_conv_5x5', 4), ('sep_conv_5x5', 1)]
    ],
    reduce_concat=range(2, 6))

DDP_PC_3 = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
        [('sep_conv_3x3', 4), ('skip_connect', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 2), ('max_pool_3x3', 0)],
        [('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
        [('skip_connect', 1), ('avg_pool_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

DDP_PATH_1= Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 3)],
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)]],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('dil_conv_3x3', 1)],
        [('sep_conv_5x5', 0), ('skip_connect', 2)],
        [('dil_conv_3x3', 2), ('dil_conv_5x5', 1)],
        [('dil_conv_5x5', 0), ('max_pool_3x3', 2)]],
    reduce_concat=range(2, 6))

DDP_PATH_2 = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_5x5', 1), ('max_pool_3x3', 2)],
        [('max_pool_3x3', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 4), ('dil_conv_5x5', 3)]],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('avg_pool_3x3', 0), ('skip_connect', 1)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 2)],
        [('sep_conv_3x3', 2), ('skip_connect', 0)]],
    reduce_concat=range(2, 6))

DDP_PATH_MEAN_1 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('avg_pool_3x3', 1)],
        [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)],
        [('dil_conv_5x5', 3), ('sep_conv_3x3', 2)],
        [('max_pool_3x3', 1), ('sep_conv_5x5', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 2), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 2), ('dil_conv_3x3', 4)]],
    reduce_concat=range(2, 6))

DDP_PATH_MEAN_2 = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('max_pool_3x3', 0), ('sep_conv_3x3', 2)],
        [('sep_conv_3x3', 3), ('dil_conv_3x3', 0)],
        [('sep_conv_5x5', 4), ('skip_connect', 0)]],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_3x3', 1), ('skip_connect', 0)],
        [('dil_conv_3x3', 3), ('avg_pool_3x3', 2)],
        [('dil_conv_5x5', 4), ('sep_conv_5x5', 1)]
    ], reduce_concat=range(2, 6))

DDP_PATH_MEAN_3 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('dil_conv_3x3', 1)],
        [('max_pool_3x3', 2), ('max_pool_3x3', 0)],
        [('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
        [('avg_pool_3x3', 3), ('dil_conv_3x3', 1)]],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 0), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('avg_pool_3x3', 3), ('avg_pool_3x3', 0)],
        [('skip_connect', 1), ('avg_pool_3x3', 4)]
    ], reduce_concat=range(2, 6))

DDP_PR_1 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 2)],
        [('dil_conv_3x3', 3), ('skip_connect', 1)],
        [('avg_pool_3x3', 1), ('max_pool_3x3', 3)]],
    normal_concat=range(2, 6),
    reduce=[
        [('skip_connect', 1), ('avg_pool_3x3', 0)],
        [('sep_conv_5x5', 2), ('max_pool_3x3', 1)],
        [('sep_conv_5x5', 0), ('skip_connect', 3)],
        [('sep_conv_5x5', 2), ('sep_conv_5x5', 4)]
    ], reduce_concat=range(2, 6))

DDPNAS_PLANT_64 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
        [('sep_conv_3x3', 0), ('max_pool_3x3', 2)],
        [('dil_conv_3x3', 0), ('max_pool_3x3', 2)],
        [('avg_pool_3x3', 3), ('sep_conv_3x3', 2)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('skip_connect', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
        [('max_pool_3x3', 1), ('skip_connect', 2)],
        [('sep_conv_3x3', 3), ('sep_conv_5x5', 1)]],
    reduce_concat=range(2, 6))

DDPNAS_PLANT_224 = Genotype(
    normal=[
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('avg_pool_3x3', 0), ('max_pool_3x3', 2)],
        [('max_pool_3x3', 1), ('sep_conv_5x5', 2)],
        [('dil_conv_3x3', 3), ('sep_conv_5x5', 2)]],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 1), ('sep_conv_3x3', 0)],
        [('max_pool_3x3', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_3x3', 0), ('dil_conv_3x3', 2)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 1)]],
    reduce_concat=range(2, 6))

DDPNAS_PLANT_300M  = Genotype(
    normal=[
        [('dil_conv_3x3', 1), ('max_pool_3x3', 0)],
        [('avg_pool_3x3', 1), ('dil_conv_5x5', 2)],
        [('max_pool_3x3', 0), ('sep_conv_5x5', 2)],
        [('max_pool_3x3', 4), ('dil_conv_3x3', 2)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('max_pool_3x3', 2), ('dil_conv_5x5', 0)],
        [('skip_connect', 3), ('avg_pool_3x3', 1)],
        [('dil_conv_5x5', 4), ('sep_conv_3x3', 0)]],
    reduce_concat=range(2, 6))

DDP_PRA_1= Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('sep_conv_3x3', 2)],
        [('dil_conv_3x3', 2), ('max_pool_3x3', 3)],
        [('dil_conv_3x3', 3), ('dil_conv_5x5', 1)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_5x5', 2), ('avg_pool_3x3', 0)],
        [('sep_conv_5x5', 3), ('skip_connect', 0)],
        [('avg_pool_3x3', 0), ('sep_conv_3x3', 1)]],
    reduce_concat=range(2, 6))

DDP_PATH_LAST_1 = Genotype(
    normal=[
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 1)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 1), ('max_pool_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('avg_pool_3x3', 0), ('dil_conv_5x5', 1)],
        [('dil_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_3x3', 3), ('sep_conv_5x5', 2)],
        [('avg_pool_3x3', 1), ('avg_pool_3x3', 3)]],
    reduce_concat=range(2, 6))

DDPNAS_PATH_4G = Genotype(
    normal=[
        [('skip_connect', 0), ('dil_conv_5x5', 1)],
        [('avg_pool_3x3', 2), ('max_pool_3x3', 1)],
        [('dil_conv_5x5', 1), ('dil_conv_5x5', 3)],
        [('dil_conv_5x5', 1), ('max_pool_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('skip_connect', 0), ('dil_conv_5x5', 1)],
        [('max_pool_3x3', 2), ('avg_pool_3x3', 0)],
        [('skip_connect', 0), ('max_pool_3x3', 3)],
        [('sep_conv_3x3', 2), ('dil_conv_5x5', 3)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_1 = Genotype(
    normal=[
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 1)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 4)]
        ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('skip_connect', 2), ('max_pool_3x3', 3)],
        [('sep_conv_3x3', 3), ('max_pool_3x3', 0)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_2 = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
        [('dil_conv_3x3', 2), ('skip_connect', 0)],
        [('sep_conv_3x3', 3), ('avg_pool_3x3', 1)],
        [('dil_conv_3x3', 4), ('avg_pool_3x3', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('dil_conv_3x3', 1), ('sep_conv_3x3', 0)],
        [('max_pool_3x3', 1), ('dil_conv_5x5', 2)],
        [('dil_conv_5x5', 2), ('sep_conv_5x5', 1)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_3 = Genotype(
    normal=[
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 1)],
        [('avg_pool_3x3', 0), ('dil_conv_5x5', 1)],
        [('dil_conv_3x3', 0), ('skip_connect', 3)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 4)]],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('avg_pool_3x3', 2), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 2), ('dil_conv_3x3', 3)],
        [('sep_conv_5x5', 2), ('skip_connect', 3)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_M2_1 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('dil_conv_5x5', 1)],
        [('dil_conv_5x5', 2), ('sep_conv_5x5', 1)],
        [('skip_connect', 4), ('dil_conv_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('avg_pool_3x3', 1), ('sep_conv_3x3', 0)],
        [('dil_conv_3x3', 1), ('dil_conv_5x5', 0)],
        [('avg_pool_3x3', 2), ('dil_conv_5x5', 3)],
        [('dil_conv_3x3', 4), ('avg_pool_3x3', 2)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_M2_2 = Genotype(
    normal=[
        [('max_pool_3x3', 1), ('dil_conv_3x3', 0)],
        [('sep_conv_3x3', 2), ('sep_conv_3x3', 0)],
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_5x5', 4), ('max_pool_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
        [('avg_pool_3x3', 2), ('max_pool_3x3', 0)],
        [('dil_conv_3x3', 2), ('sep_conv_5x5', 0)],
        [('dil_conv_5x5', 1), ('dil_conv_5x5', 4)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_A_1 = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('dil_conv_5x5', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 2), ('dil_conv_5x5', 1)],
        [('max_pool_3x3', 2), ('max_pool_3x3', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_3x3', 2), ('max_pool_3x3', 1)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 2)],
        [('skip_connect', 3), ('avg_pool_3x3', 4)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_A_2 = Genotype(
    normal=[
        [('dil_conv_5x5', 1), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('max_pool_3x3', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 3), ('sep_conv_5x5', 2)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 0), ('skip_connect', 1)],
        [('dil_conv_5x5', 0), ('skip_connect', 1)],
        [('dil_conv_3x3', 2), ('avg_pool_3x3', 1)],
        [('skip_connect', 1), ('sep_conv_5x5', 4)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_A_3 = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 1), ('dil_conv_3x3', 2)],
        [('dil_conv_5x5', 2), ('sep_conv_5x5', 0)],
        [('max_pool_3x3', 0), ('dil_conv_5x5', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_3x3', 2), ('dil_conv_5x5', 1)],
        [('skip_connect', 1), ('dil_conv_5x5', 3)],
        [('sep_conv_5x5', 2), ('dil_conv_5x5', 4)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_MCN_B_0 = Genotype(
    normal=[
        [('dil_conv_3x3', 1), ('avg_pool_3x3', 0)],
        [('avg_pool_3x3', 2), ('avg_pool_3x3', 0)],
        [('skip_connect', 2), ('sep_conv_3x3', 3)],
        [('skip_connect', 0), ('dil_conv_5x5', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('dil_conv_5x5', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('max_pool_3x3', 3)],
        [('dil_conv_3x3', 4), ('dil_conv_3x3', 3)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_B_1 = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 1)],
        [('max_pool_3x3', 2), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 2), ('avg_pool_3x3', 0)],
        [('dil_conv_5x5', 3), ('sep_conv_3x3', 2)]
    ], normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 0), ('max_pool_3x3', 1)],
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 2)],
        [('dil_conv_3x3', 1), ('max_pool_3x3', 2)],
        [('sep_conv_5x5', 3), ('avg_pool_3x3', 0)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_MCN_B_2 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_3x3', 2), ('avg_pool_3x3', 0)],
        [('avg_pool_3x3', 1), ('max_pool_3x3', 3)],
        [('max_pool_3x3', 3), ('max_pool_3x3', 4)]],
    normal_concat=range(2, 6),
    reduce=[
        [('avg_pool_3x3', 1), ('dil_conv_3x3', 0)],
        [('dil_conv_5x5', 1), ('avg_pool_3x3', 0)],
        [('dil_conv_3x3', 2), ('sep_conv_5x5', 3)],
        [('dil_conv_3x3', 4), ('sep_conv_3x3', 1)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_MCN_B_1_NSep = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 1)],
        [('dil_conv_3x3', 2), ('avg_pool_3x3', 0)],
        [('skip_connect', 0), ('avg_pool_3x3', 2)],
        [('avg_pool_3x3', 1), ('dil_conv_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)],
        [('avg_pool_3x3', 1), ('avg_pool_3x3', 0)],
        [('avg_pool_3x3', 3), ('max_pool_3x3', 2)],
        [('skip_connect', 1), ('dil_conv_3x3', 0)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_MCN_B_1_NSep_PC = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
        [('max_pool_3x3', 0), ('dil_conv_5x5', 2)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('sep_conv_3x3', 2), ('skip_connect', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 1), ('sep_conv_3x3', 0)],
        [('dil_conv_5x5', 2), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 3), ('max_pool_3x3', 2)],
        [('sep_conv_3x3', 3), ('dil_conv_5x5', 4)]
    ], reduce_concat=range(2, 6))

DDPNAS_MCN_B_2_NSep_PC = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('dil_conv_3x3', 0), ('max_pool_3x3', 2)],
        [('sep_conv_5x5', 3), ('max_pool_3x3', 2)],
        [('dil_conv_5x5', 3), ('sep_conv_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('dil_conv_5x5', 2), ('sep_conv_5x5', 0)],
        [('max_pool_3x3', 2), ('dil_conv_5x5', 3)],
        [('dil_conv_5x5', 4), ('sep_conv_5x5', 2)]
    ], reduce_concat=range(2, 6))

DDPNAS_MCN_B_Conv_1 = Genotype(
    normal=[
        [('avg_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('avg_pool_3x3', 2), ('dil_conv_5x5', 0)],
        [('max_pool_3x3', 0), ('dil_conv_3x3', 2)],
        [('dil_conv_3x3', 3), ('dil_conv_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 0), ('dil_conv_5x5', 1)],
        [('avg_pool_3x3', 2), ('max_pool_3x3', 0)],
        [('sep_conv_3x3', 2), ('dil_conv_5x5', 1)],
        [('sep_conv_5x5', 1), ('max_pool_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

BNAS_MCN_1 = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_5x5', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 2)],
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 2)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 0), ('dil_conv_5x5', 1)],
        [('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 3), ('max_pool_3x3', 2)],
        [('max_pool_3x3', 2), ('max_pool_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

BNAS_MCN_2 = Genotype( # 可用
    normal=[
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 3), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 3), ('max_pool_3x3', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_5x5', 1), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
        [('max_pool_3x3', 0), ('max_pool_3x3', 2)],
        [('sep_conv_5x5', 3), ('dil_conv_5x5', 1)]],
    reduce_concat=range(2, 6))

BNAS_MCN_3 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_5x5', 2), ('dil_conv_5x5', 3)],
        [('avg_pool_3x3', 3), ('sep_conv_3x3', 4)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_5x5', 2), ('sep_conv_3x3', 1)],
        [('max_pool_3x3', 3), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 3), ('max_pool_3x3', 4)]
    ],
    reduce_concat=range(2, 6))

BNAS_MCN_4 = Genotype(
    normal=[
        [('dil_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 3)],
        [('sep_conv_5x5', 2), ('max_pool_3x3', 3)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('sep_conv_5x5', 1), ('max_pool_3x3', 2)],
        [('max_pool_3x3', 2), ('sep_conv_5x5', 1)],
        [('skip_connect', 0), ('dil_conv_3x3', 1)]],
    reduce_concat=range(2, 6))

BNAS_MCN_5 = Genotype( # 可用
    normal=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('dil_conv_5x5', 0), ('sep_conv_5x5', 2)],
        [('sep_conv_5x5', 3), ('skip_connect', 0)],
        [('max_pool_3x3', 0), ('dil_conv_5x5', 4)]],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('dil_conv_5x5', 3), ('sep_conv_5x5', 0)],
        [('sep_conv_5x5', 0), ('max_pool_3x3', 3)]],
    reduce_concat=range(2, 6))

BNAS_MCN_6 = Genotype(
    normal=[
        [('dil_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('max_pool_3x3', 0), ('dil_conv_5x5', 2)],
        [('sep_conv_5x5', 1), ('dil_conv_5x5', 0)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 3)]],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
        [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
        [('max_pool_3x3', 1), ('sep_conv_5x5', 0)]],
    reduce_concat=range(2, 6))

BNAS_MCN_7 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 0)],
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_5x5', 3), ('sep_conv_5x5', 0)]],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('max_pool_3x3', 0), ('sep_conv_5x5', 2)],
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)],
        [('dil_conv_5x5', 4), ('sep_conv_3x3', 0)]],
    reduce_concat=range(2, 6))

BNAS_MCN_8 = Genotype(
    normal=[
        [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 3)],
        [('sep_conv_5x5', 3), ('dil_conv_3x3', 0)]],
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)],
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
        [('sep_conv_3x3', 0), ('max_pool_3x3', 1)]],
    reduce_concat=range(2, 6))

PCDARTS_MCN_1 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], # 
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], # sep_conv_5x5
        [('sep_conv_3x3', 3), ('sep_conv_3x3', 0)],
        [('sep_conv_5x5', 2), ('sep_conv_5x5', 0)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)],
        [('sep_conv_5x5', 1), ('dil_conv_3x3', 2)],
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], # sep_conv_5x5
        [('sep_conv_3x3', 3), ('sep_conv_5x5', 1)]],
    reduce_concat=range(2, 6))

DDPNAS_MCN_B_PC_Sep_1 = Genotype(
    normal=[
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 2)],
        [('dil_conv_5x5', 3), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 4), ('dil_conv_5x5', 3)]
    ], normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('max_pool_3x3', 2), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 2), ('dil_conv_5x5', 3)],
        [('dil_conv_5x5', 3), ('dil_conv_5x5', 4)]
    ],
    reduce_concat=range(2, 6))

DDPNAS_MCN_IMAGENET_1 = Genotype(
    normal=[
        [('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
        [('max_pool_3x3', 1), ('dil_conv_5x5', 0)],
        [('avg_pool_3x3', 1), ('dil_conv_5x5', 3)],
        [('sep_conv_5x5', 2), ('avg_pool_3x3', 3)]],
    normal_concat=range(2, 6),
    reduce=[
        [('max_pool_3x3', 0), ('avg_pool_3x3', 1)],
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 2)],
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 1)],
        [('dil_conv_5x5', 2), ('sep_conv_3x3', 0)]],
    reduce_concat=range(2, 6))

DDPNAS_XNOR_A_1 = Genotype(
    normal=[
        [('skip_connect', 1), ('dil_conv_3x3', 0)],
        [('skip_connect', 0), ('dil_conv_5x5', 2)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 4), ('sep_conv_3x3', 2)]
    ],
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 0)],
        [('skip_connect', 1), ('max_pool_3x3', 2)],
        [('dil_conv_5x5', 0), ('avg_pool_3x3', 3)],
        [('skip_connect', 2), ('dil_conv_3x3', 4)]],
    reduce_concat=range(2, 6))

DDPNAS_CHL = Genotype(
    normal=[
        [('sep_conv_5x5', 1), ('dil_conv_5x5', 0)],
        [('dil_conv_3x3', 2), ('dil_conv_5x5', 0)], 
        [('dil_conv_5x5', 0), ('max_pool_3x3', 1)], 
        [('sep_conv_3x3', 2), ('dil_conv_5x5', 4)]], 
    normal_concat=range(2, 6), 
    reduce=[
        [('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], 
        [('sep_conv_3x3', 0), ('dil_conv_5x5', 2)],
        [('dil_conv_5x5', 2), ('skip_connect', 3)], 
        [('skip_connect', 2), ('avg_pool_3x3', 0)]],
    reduce_concat=range(2, 6))

DDPNAS_CHL_B = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('sep_conv_3x3', 1)], 
        [('dil_conv_5x5', 0), ('max_pool_3x3', 2)], 
        [('sep_conv_5x5', 3), ('sep_conv_5x5', 2)], 
        [('max_pool_3x3', 0), ('max_pool_3x3', 2)]],
    normal_concat=range(2, 6), 
    reduce=[
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], 
        [('sep_conv_5x5', 1), ('dil_conv_3x3', 2)], 
        [('dil_conv_5x5', 1), ('avg_pool_3x3', 2)], 
        [('skip_connect', 2), ('avg_pool_3x3', 0)]],
    reduce_concat=range(2, 6))

DDPNAS_CHL_B1 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)], 
        [('dil_conv_3x3', 1), ('max_pool_3x3', 0)], 
        [('max_pool_3x3', 0), ('dil_conv_3x3', 1)], 
        [('max_pool_3x3', 0), ('max_pool_3x3', 4)]], 
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], 
        [('dil_conv_5x5', 2), ('sep_conv_5x5', 1)], 
        [('avg_pool_3x3', 2), ('dil_conv_3x3', 0)], 
        [('avg_pool_3x3', 2), ('sep_conv_5x5', 1)]],
    reduce_concat=range(2, 6))

DDPNAS_CHL_B2 = Genotype(
    normal=[
        [('max_pool_3x3', 1), ('dil_conv_3x3', 0)], 
        [('max_pool_3x3', 1), ('dil_conv_3x3', 0)],
        [('max_pool_3x3', 1), ('dil_conv_5x5', 3)], 
        [('dil_conv_3x3', 0), ('max_pool_3x3', 2)]], 
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], 
        [('sep_conv_5x5', 0), ('skip_connect', 2)], 
        [('skip_connect', 3), ('sep_conv_5x5', 2)], 
        [('sep_conv_3x3', 1), ('max_pool_3x3', 3)]], 
    reduce_concat=range(2, 6))

DDPNAS_CHL_I1 = Genotype(
    normal=[
        [('skip_connect', 0), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 1), ('max_pool_3x3', 2)], 
        [('sep_conv_3x3', 0), ('max_pool_3x3', 3)], 
        [('dil_conv_3x3', 3), ('max_pool_3x3', 4)]], 
    normal_concat=range(2, 6),
    reduce=[
        [('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], 
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], 
        [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 3), ('sep_conv_3x3', 1)]], 
    reduce_concat=range(2, 6))

DDPNAS_CHL_B11 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)], 
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)], 
        [('skip_connect', 3), ('max_pool_3x3', 1)], 
        [('skip_connect', 4), ('dil_conv_5x5', 3)]], 
    normal_concat=range(2, 6), 
    reduce=[
        [('avg_pool_3x3', 0), ('skip_connect', 1)], 
        [('max_pool_3x3', 1), ('dil_conv_3x3', 2)],
        [('avg_pool_3x3', 1), ('sep_conv_3x3', 0)], 
        [('dil_conv_3x3', 4), ('dil_conv_3x3', 2)]], 
    reduce_concat=range(2, 6))

DDPNAS_CHL_I2 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 0), ('dil_conv_3x3', 2)], 
        [('dil_conv_3x3', 0), ('max_pool_3x3', 3)],
        [('sep_conv_3x3', 3), ('skip_connect', 1)]],
    normal_concat=range(2, 6), 
    reduce=[
        [('skip_connect', 0), ('dil_conv_3x3', 1)], 
        [('avg_pool_3x3', 1), ('skip_connect', 2)], 
        [('max_pool_3x3', 2), ('skip_connect', 0)], 
        [('sep_conv_3x3', 4), ('skip_connect', 0)]], 
    reduce_concat=range(2, 6))

DDPNAS_CHL_B15 = Genotype(
    normal=[
        [('avg_pool_3x3', 1), ('max_pool_3x3', 0)],
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)], 
        [('max_pool_3x3', 0), ('max_pool_3x3', 2)], 
        [('sep_conv_3x3', 3), ('skip_connect', 4)]], 
    normal_concat=range(2, 6),
    reduce=[
        [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)], 
        [('skip_connect', 2), ('dil_conv_3x3', 1)], 
        [('max_pool_3x3', 0), ('max_pool_3x3', 3)], 
        [('dil_conv_5x5', 2), ('avg_pool_3x3', 0)]], 
    reduce_concat=range(2, 6))


DDPNAS_CHL_I3 = Genotype(
    normal=[
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)], 
        [('skip_connect', 1), ('sep_conv_3x3', 0)],
        [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
        [('dil_conv_3x3', 2), ('avg_pool_3x3', 1)]], 
    normal_concat=range(2, 6),
    reduce=[
        [('dil_conv_5x5', 1), ('dil_conv_5x5', 0)], 
        [('avg_pool_3x3', 2), ('dil_conv_5x5', 0)], 
        [('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], 
        [('avg_pool_3x3', 1), ('dil_conv_3x3', 2)]],
    reduce_concat=range(2, 6))


DDPNAS_CHL_I4 = Genotype(
    normal=[
        [('max_pool_3x3', 0), ('skip_connect', 1)], 
        [('max_pool_3x3', 2), ('max_pool_3x3', 0)],
        [('dil_conv_5x5', 2), ('dil_conv_3x3', 0)], 
        [('max_pool_3x3', 1), ('dil_conv_3x3', 4)]],
    normal_concat=range(2, 6), 
    reduce=[
        [('skip_connect', 0), ('max_pool_3x3', 1)], 
        [('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
        [('sep_conv_3x3', 3), ('sep_conv_3x3', 2)],
        [('skip_connect', 3), ('max_pool_3x3', 1)]],
    reduce_concat=range(2, 6))
 
DDPNAS_CHL_I6 = Genotype(
    normal=[
        [('avg_pool_3x3', 0), ('max_pool_3x3', 1)], 
        [('max_pool_3x3', 0), ('avg_pool_3x3', 2)], 
        [('max_pool_3x3', 1), ('dil_conv_5x5', 3)], 
        [('max_pool_3x3', 1), ('dil_conv_5x5', 4)]], 
    normal_concat=range(2, 6), 
    reduce=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], 
        [('sep_conv_5x5', 2), ('sep_conv_5x5', 0)],
        [('dil_conv_5x5', 0), ('max_pool_3x3', 1)]], 
    reduce_concat=range(2, 6))

Tangent_1 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
        [('sep_conv_3x3', 0), ('dil_conv_3x3', 1)],
        [('sep_conv_5x5', 3), ('max_pool_3x3', 2)], 
        [('sep_conv_5x5', 4), ('max_pool_3x3', 0)]],
    normal_concat=range(2, 6), 
    reduce=[
        [('max_pool_3x3', 0), ('sep_conv_5x5', 1)], 
        [('dil_conv_3x3', 1), ('dil_conv_5x5', 2)], 
        [('skip_connect', 0), ('dil_conv_5x5', 3)], 
        [('sep_conv_5x5', 2), ('skip_connect', 4)]], 
    reduce_concat=range(2, 6))

Tangent_2 = Genotype(
    normal=[
        [('skip_connect', 0), ('dil_conv_3x3', 1)],
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 2)],
        [('avg_pool_3x3', 2), ('sep_conv_3x3', 3)], 
        [('max_pool_3x3', 0), ('dil_conv_5x5', 4)]], 
    normal_concat=range(2, 6), 
    reduce=[
        [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], 
        [('max_pool_3x3', 1), ('dil_conv_5x5', 2)], 
        [('dil_conv_3x3', 1), ('avg_pool_3x3', 2)], 
        [('sep_conv_3x3', 1), ('dil_conv_5x5', 3)]],
    reduce_concat=range(2, 6))




genotype_array = {
    'NASNet': NASNet,
    'MDENAS': MDENAS,
    'DDPNAS_1': DDPNAS_1,
    'DDPNAS_2': DDPNAS_2,
    'DDPNAS_3': DDPNAS_3,
    'DDPNAS_4': DDPNAS_4,
    'DDPNAS_5': DDPNAS_5,
    'DDPNAS_PLANT': DDPNAS_PLANT,
    'DDPNAS_PLANT_64': DDPNAS_PLANT_64,
    'DDPNAS_PLANT_224': DDPNAS_PLANT_224,
    'DDPNAS_PLANT_300M': DDPNAS_PLANT_300M,
    'DDPNAS_PATH_4G': DDPNAS_PATH_4G,
    'DDP_PC_1': DDP_PC_1,
    'DDP_PC_2': DDP_PC_2,
    'DDP_PATH_1': DDP_PATH_1,
    'DDP_PATH_2': DDP_PATH_2,
    'DDP_PATH_MEAN_1': DDP_PATH_MEAN_1,
    'DDP_PATH_MEAN_2': DDP_PATH_MEAN_2,
    'DDP_PATH_MEAN_3': DDP_PATH_MEAN_3,
    'DDP_PATH_LAST_1': DDP_PATH_LAST_1,
    "PCDARTS_MCN_1": PCDARTS_MCN_1,
    'DDP_PR_1': DDP_PR_1,
    'DDP_PRA_1': DDP_PRA_1,
    'DARTS_V1': DARTS_V1,
    'DARTS_V2': DARTS_V2,
    'DDPNAS_MCN_1': DDPNAS_MCN_1,
    'DDPNAS_MCN_2': DDPNAS_MCN_2,
    'DDPNAS_MCN_3': DDPNAS_MCN_3,
    'DDPNAS_MCN_A_1': DDPNAS_MCN_A_1,
    'DDPNAS_MCN_A_2': DDPNAS_MCN_A_2,
    'DDPNAS_MCN_A_3': DDPNAS_MCN_A_3,
    'DDPNAS_MCN_B_0': DDPNAS_MCN_B_0,
    'DDPNAS_MCN_B_1': DDPNAS_MCN_B_1,
    'DDPNAS_MCN_B_2': DDPNAS_MCN_B_2,
    'DDPNAS_MCN_B_1_NSep': DDPNAS_MCN_B_1_NSep,
    'DDPNAS_MCN_B_1_NSep_PC': DDPNAS_MCN_B_1_NSep_PC,
    'DDPNAS_MCN_B_PC_Sep_1': DDPNAS_MCN_B_PC_Sep_1,
    'DDPNAS_MCN_B_Conv_1': DDPNAS_MCN_B_Conv_1,
    'DDPNAS_XNOR_A_1': DDPNAS_XNOR_A_1,
    'DDPNAS_MCN_M2_1': DDPNAS_MCN_M2_1,
    'DDPNAS_MCN_M2_2': DDPNAS_MCN_M2_2,
    'BNAS_MCN_1': BNAS_MCN_1,
    'BNAS_MCN_2': BNAS_MCN_2,
    'BNAS_MCN_3': BNAS_MCN_3,
    'BNAS_MCN_4': BNAS_MCN_4,
    'BNAS_MCN_5': BNAS_MCN_5,
    'BNAS_MCN_6': BNAS_MCN_6,
    'BNAS_MCN_7': BNAS_MCN_7,
    "DDPNAS_MCN_IMAGENET_1": DDPNAS_MCN_IMAGENET_1,
    'AmoebaNet': AmoebaNet,
    'DDPNAS_CHL': DDPNAS_CHL,
    'DDPNAS_CHL_B': DDPNAS_CHL_B,
    'DDPNAS_CHL_B1': DDPNAS_CHL_B1,
    'DDPNAS_CHL_B2': DDPNAS_CHL_B2,
    'DDPNAS_CHL_I2': DDPNAS_CHL_I2, 
    'DDPNAS_CHL_I4': DDPNAS_CHL_I4,
    'DDPNAS_CHL_I6': DDPNAS_CHL_I6,
    'DDPNAS_CHL_B11': DDPNAS_CHL_B11, 
    'DDPNAS_CHL_B15': DDPNAS_CHL_B15, 
    'Tangent_1':Tangent_1,
    'Tangent_2':Tangent_2,
    }
