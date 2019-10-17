import torch, os, sys

from torch import nn
from torch.nn import functional as F

import layers

from abc import abstractmethod
from math import sqrt

import util


class GCN(nn.Module):
    """
    Graph convolution: node outputs are the average of all neighbors.
    """
    def __init__(self, graph, emb=16, activation=F.relu, bases=None):

        super().__init__()

        rn, n = graph[1]
        r = rn//n

        indices, size = graph

        ih, iw = indices.size()
        vals = torch.ones((ih, ), dtype=torch.float)

        vals = vals / util.sum_sparse(indices, vals, size)

        graph = torch.sparse.FloatTensor(indices=indices.t(), values=vals, size=size) # will this get cuda'd properly?
        self.register_buffer('graph', graph)

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )
            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases).uniform_(-sqrt(bases), sqrt(bases)) )
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )

        self.activation = activation

    def forward(self, x):
        """
        :param x: E by N matrix of node embeddings.

        :return:
        """
        rn, n = self.graph.size()
        r = rn // n

        n, e = x.size()

        # Multiply adjacencies
        h = torch.mm(self.graph, x)
        h = h.view(r, n, e) # new dim for the relations

        if self.bases:
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
        else:
            weights = self.weights

        # Apply weights
        h = torch.bmm(h, weights)

        # sum out relations and apply activation
        return self.activation(h.sum(dim=0))

class GAT(nn.Module):
    """
    Self-attention over the graph

    """

    def __init__(self, graph, emb=16):

        super().__init__(emb=emb)

    def inner(self, x):
        """
        :param x: E by N matrix of node embeddings.

        :return:
        """

