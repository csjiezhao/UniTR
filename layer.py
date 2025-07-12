from typing import Optional, Callable
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter_add
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import ChebConv


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        pe = self.pe[:, positions, :].squeeze(0)
        return self.dropout(pe)


class HyperGConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0,
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = nn.Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.position_encoding = PositionalEncoding(hid_dim)
        self.pos_flag = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]

            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        self.De = De

        x = self.lin_n2e(x)
        self.pos_flag = True
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e, size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)
        self.pos_flag = False
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,
                           size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e  # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        if self.pos_flag:
            relative_positions = torch.cat([torch.arange(length, dtype=torch.long) for length in self.De])
            pos_enc = self.position_encoding(relative_positions)
            x_j += pos_enc
        return norm.view(-1, 1) * x_j


class HieraGConv(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, K):
        super(HieraGConv, self).__init__()
        self.cheb_conv = ChebConv(in_dim, hid_dim, K)
        self.hyper_conv = HyperGConv(hid_dim, hid_dim, out_dim, dropout=0.3)

    def forward(self, x, edge_index, hyperedge_index, full_num_nodes, full_num_hyperedges):
        x_low = self.cheb_conv(x, edge_index)
        if hyperedge_index is None:
            return x_low, None
        else:
            x_high, e_high = self.hyper_conv(x_low, hyperedge_index, full_num_nodes, full_num_hyperedges)
            return x_low + x_high, e_high


class HieraGCN(nn.Module):
    def __init__(self, in_dim, node_dim, edge_dim, cheb_k=3, num_layers=1):
        super(HieraGCN, self).__init__()
        self.graph_convs = nn.ModuleList([HieraGConv(in_dim, node_dim, edge_dim, K=cheb_k)])
        self.graph_convs.extend([HieraGConv(node_dim, node_dim, edge_dim, K=cheb_k)
                                 for _ in range(1, num_layers)])

    def forward(self, x, edge_index, hyperedge_index, full_num_nodes, full_num_hyperedges):
        for conv in self.graph_convs:
            x = conv(x, edge_index, hyperedge_index, full_num_nodes, full_num_hyperedges)
        return x
