import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
import torch_geometric as tg
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)


""" Graph U-Net architecture with SAGPooling instead of TopKPooling """

class Ours(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 depth=1, pool_ratios=0.5, act=F.relu, sum_res=True,
                 augmentation=False):
        super(Ours, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = tg.utils.repeat.repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res
        self.augmentation = augmentation

        channels = hidden_channels

        self.pools = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(SAGPooling(channels,
                                         ratio=self.pool_ratios[i],
                                         min_score=None, multiplier=1,
                                         nonlinearity=torch.tanh))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
        # sum or concat mode
        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            if self.augmentation:
                edge_index, edge_weight = self.augment_adj(edge_index,
                                                           edge_weight,
                                                           x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def final_parameters(self):
        yield self.up_convs[-1].weight
        yield self.up_convs[-1].bias

    def reset_final_parameters(self):
        self.up_convs[-1].reset_parameters()






