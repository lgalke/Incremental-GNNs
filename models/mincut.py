
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense import dense_mincut_pool, DenseGraphConv
from torch_geometric.nn.conv import GraphConv
from torch_geometric.utils import to_dense_adj

DEBUG = False


def train_mincut(model, optimizer, g, feats, labels, mask=None,
                 epochs=1, state=None, alpha_mc=1.,
                 alpha_o=1.):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, mc_loss, o_loss = model(feats, g)
        if mask is not None:
            clf_loss = F.cross_entropy(logits[mask], labels[mask],
                                       reduction='mean')
        else:
            clf_loss = F.cross_entropy(logits, labels,
                                       reduction='mean')

        loss = clf_loss + alpha_mc * mc_loss + alpha_o * o_loss
        loss.backward()
        optimizer.step()
        print("""Epoch {:d} | Loss: {:.4f} | CLF Loss: {:.4f} | MC Loss: {:.4f} | O Loss: {:.4f}""".format(
            epoch+1, loss.detach().item(),
            clf_loss.detach().item(),
            mc_loss.detach().item(),
            o_loss.detach().item()))

    # Apply encoding once more to get temporal state
    model.eval()
    state = model.encode(feats, g)
    return state


def evaluate_mincut(model, g, feats, labels, mask=None, compute_loss=True,
                    state=None):
    model.eval()
    with torch.no_grad():
        logits, __mc_loss, __o_loss = model(feats, g)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)

    return acc.item(), loss


class MinCUT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 pool_size, n_layers=1, activation=F.relu,
                 dropout=0.5,
                 smoothing=0.99):
        super(MinCUT, self).__init__()
        assert smoothing > 0 and smoothing <= 1.
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.activation = activation
        self.smoothing = smoothing

        channels = hidden_channels

        self.input_layer = GraphConv(in_channels, channels, aggr='add')
        self.mlp = nn.Sequential(nn.Linear(channels, channels),
                                 nn.ReLU(), nn.Linear(channels, pool_size))
        self.output_layer = DenseGraphConv(channels, out_channels, aggr='add')

        self.dropout = nn.Dropout(dropout) if dropout else None

    def encode(self, x, edge_index, mask=None):
        x = self.activation(self.input_layer(x, edge_index))
        if self.dropout:
            x = self.dropout(x)
        s = self.mlp(x)
        # Do pool
        a = to_dense_adj(edge_index)
        if DEBUG:
            print("Pre-pool x:", x.size())
            print("Pre-pool a:", a.size())
        x_pool, a_pool, mc_loss, o_loss = dense_mincut_pool(x, a, s, mask=mask)
        return x_pool, a_pool, s, mc_loss, o_loss

    def decode(self, x_pool, a_pool, s):
        # Unpool
        x_unpool = s @ x_pool
        a_unpool = s @ a_pool @ s.T
        y = self.output_layer(x_unpool, a_unpool)
        if DEBUG:
            print("Pool x:", x_pool.size())
            print("Pool a:", a_pool.size())
            print("Unpool x:", x_unpool.size())
            print("Unpool a:", a_unpool.size())
        return y.squeeze(0)

    def forward(self, x, edge_index, state=None, mask=None):
        x_pool, a_pool, s, mc_loss, o_loss = self.encode(x, edge_index,
                                                         mask=None)
        # Apply temporal smoothing in pooled space
        if state is not None:
            x_pool = state * (1 - self.smoothing) + x_pool * self.smoothing
        y = self.decode(x_pool, a_pool, s)
        return y, mc_loss, o_loss

    def reset_final_parameters(self):
        self.output_layer.reset_parameters()

    def final_parameters(self):
        yield self.output_layer.weight
        yield self.output_layer.lin.weight
        yield self.output_layer.lin.bias

    # def __repr__(self):
    #     return '{}({}, {}, {})'.format(self.__class__.__name__,
    #                                    self.in_channels,
    #                                    self.hidden_channels,
    #                                    self.out_channels)
