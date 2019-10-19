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
    def __init__(self, edges, n, emb=16, activation=F.relu, bases=None, **kwargs):

        super().__init__()

        indices, size = util.adj(edges, n)

        rn, n = size
        r = rn//n

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

    def __init__(self, edges, n, emb=16, heads=4, norm_method='naive', **kwargs):
        """

        :param graph:
        :param emb:
        :param heads: Number of attention heads per relation
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding size ({emb}) should be divisible by nr of heads ({heads}).'

        self.emb = emb
        self.heads = heads
        self.norm_method = norm_method

        s = emb//heads
        r = self.relations = len(edges.keys())

        self.tokeys    = nn.Parameter(torch.FloatTensor(r, heads, s, s).uniform_(-sqrt(s), sqrt(s)))
        self.toqueries = nn.Parameter(torch.FloatTensor(r, heads, s, s).uniform_(-sqrt(s), sqrt(s)))
        self.tovals    = nn.Parameter(torch.FloatTensor(r, heads, s, s).uniform_(-sqrt(s), sqrt(s)))

        self.unify     = nn.Linear(emb, emb, bias=False)

        # convert the edges dict to a matrix of triples
        s, o, p = [], [], []
        for pred, (sub, obj) in edges.items():
            s.extend(sub)
            o.extend(obj)
            p.extend([pred] * len(sub))

        # graph as triples
        self.register_buffer('indices', torch.tensor([s, p, o], dtype=torch.long).t())

        # graph as block-biagonal sparse matrix
        s, p, o = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]
        s, o = s + (p * n), o + (p * n)
        self.register_buffer('mindices', torch.cat([s[:, None], o[:, None]], dim=1))
        self.msize = (n*r, n*r)

    def forward(self, x):
        """
        :param x: E by N matrix of node embeddings.

        :return:
        """

        n, e = x.size()
        h, r = self.heads, self.relations
        s = e // h
        ed, _ = self.mindices.size() # nr of edges total

        x = x[:, None, :].expand(n, r, e) # expand for relations
        x = x.view(n, r, h, s)            # cut for attention heads

        # multiply so that we have a length s vector for every head in every relation for every node
        keys    = torch.einsum('rhij, nrhj -> rnhi', self.tokeys, x)
        queries = torch.einsum('rhij, nrhj -> rnhi', self.toqueries, x)

        values  = torch.einsum('rhij, nrhj -> hrni', self.tovals, x).contiguous() # note order of indices
        # - h functions as batch dimension here

        # Select from r and n dimensions
        #      Fold h into i, and extract later.

        keys = keys.view(r, n, -1)
        queries = queries.view(r, n, -1)

        # select keys and queries
        skeys    = keys   [self.indices[:, 1], self.indices[:, 0], :]
        squeries = queries[self.indices[:, 1], self.indices[:, 2], :]

        skeys = skeys.view(-1, h, s)
        squeries = squeries.view(-1, h, s)

        # compute raw dot product
        dot = torch.einsum('ehi, ehi -> he', skeys, squeries) # e = nr of edges

        # row normalize dot products
        # print(dot.size(), self.indices.size(), self.mindices.size())

        mindices = self.mindices[None, :, :].expand(h, ed, 2).contiguous()

        assert not util.contains_inf(dot), f'dot contains inf (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        assert not util.contains_nan(dot), f'dot contains nan (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        if self.norm_method == 'softmax':
            dot = util.logsoftmax(mindices, dot, self.msize).exp()
        else:
            dot = util.simple_normalize(mindices, dot, self.msize, method=self.norm_method)

        assert not util.contains_inf(dot), f'dot contains inf (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        assert not util.contains_nan(dot), f'dot contains nan (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        values = values.view(h, r*n, s)
        output = util.batchmm(mindices, dot, self.msize, values)

        assert output.size() == (h, r*n, s)

        output = output.view(h, r, n, s).permute(2, 1, 0, 3).contiguous().view(n, r, h*s)

        # unify the heads
        output = self.unify(output)

        # unify the relations
        output = output.sum(dim=1)

        assert output.size() == (n, e)

        return output

        # print('So far so good.')
        # sys.exit()

