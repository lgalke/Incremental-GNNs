from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, erdos_renyi_graph)

import torch
from torch_sparse import spspmm

def augment_adj(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                             num_nodes=num_nodes)
    edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                              num_nodes)
    edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                     edge_weight, num_nodes, num_nodes,
                                     num_nodes)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    return edge_index, edge_weight


def test_spspmm_cpu():
    for i in range(2):
        num_nodes = (i+1) * 100
        edge_index = erdos_renyi_graph(num_nodes, 0.5)
        edge_weight = torch.ones(edge_index.size(1))
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)

def test_spspmm_cuda():
    for i in range(2):
        num_nodes = (i+1) * 100
        edge_index = erdos_renyi_graph(num_nodes, 0.5).cuda()
        edge_weight = torch.ones(edge_index.size(1)).cuda()
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
def test_augment_adj_cpu():
    for i in range(2):
        num_nodes = (i+1) * 100
        edge_index = erdos_renyi_graph(num_nodes, 0.5)
        edge_weight = torch.ones(edge_index.size(1))
        edge_index, edge_weight = augment_adj(edge_index, edge_weight, num_nodes)

def test_augment_adj_cuda():
    for i in range(2):
        num_nodes = (i+1) * 100
        edge_index = erdos_renyi_graph(num_nodes, 0.5).cuda()
        edge_weight = torch.ones(edge_index.size(1)).cuda()
        edge_index, edge_weight = augment_adj(edge_index, edge_weight, num_nodes)
