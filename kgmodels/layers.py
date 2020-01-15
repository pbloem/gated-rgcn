import torch, os, sys

from torch import nn
from torch.nn import functional as F
import torch.distributions as ds

import layers

from abc import abstractmethod
from math import sqrt

import util

### Layers for unifying the different embedding vectors produced for each relation

class SumUnify(nn.Module):
    """
    Baseline: just sum them.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, r, e = x.size()
        return F.relu(x.sum(dim=1))

class AttentionUnify(nn.Module):
    """
    Compute an attention value for each relation
    """
    def __init__(self, r, e):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(r, e).uniform_(-sqrt(e), sqrt(e)))
        self.biases = nn.Parameter(torch.zeros(r, dtype=torch.float))

    def forward(self, x):
        n, r, e = x.size()

        att = torch.einsum('ri, nri -> nr', self.weights, x)
        att = torch.einsum('nr, r -> nr', att, self.biases)

        att = F.softmax(att, dim=1)

        return torch.einsum('nr, nri -> ni', att, x)

class MLPUnify(nn.Module):
    """
    Compute an attention value for each relation
    """
    def __init__(self, r, e):
        super().__init__()

        self.lin = nn.Linear(r*e, e)

    def forward(self, x):
        n, r, e = x.size()
        x = x.view(n, r*e)

        return self.lin(x)

### Mixer layers
class GCN(nn.Module):
    """
    Graph convolution: node outputs are the average of all neighbors.
    """
    def __init__(self, edges, n, emb=16, bases=None, unify='sum', **kwargs):

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
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb))
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases))
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb))
            nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))

        if unify == 'sum':
            self.unify = SumUnify()
        elif unify == 'attention':
            self.unify = AttentionUnify(r, emb)
        elif unify == 'mlp':
            self.unify = MLPUnify(r, emb)
        else:
            raise Exception(f'unify {unify} not recognized')

    def forward(self, x, conditional=None):
        """
        :param x: E by N matrix of node embeddings.

        :return:
        """
        rn, n = self.graph.size()
        r = rn // n

        n, e =  x.size()

        if conditional is not None:
            x = x + conditional

        # Multiply adjacencies
        h = torch.mm(self.graph, x) # sparse mm
        h = h.view(r, n, e) # new dim for the relations

        if self.bases is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
        else:
            weights = self.weights

        # Apply weights
        h = torch.einsum('rih, rnh -> nri', weights, h)

        return self.unify(h)

class GCNFirst(nn.Module):
    """
    First graph convolution. No input (one-hot vectors assumed)

    Note that unification is always sum.
    """
    def __init__(self, edges, n, emb=16, bases=None, **kwargs):

        super().__init__()

        self.emb = emb

        # vertical stack to find the normalization
        vindices, vsize = util.adj(edges, n, vertical=False)
        ih, iw = vindices.size()

        vals = torch.ones((ih, ), dtype=torch.float)
        vals = vals / util.sum_sparse(vindices, vals, vsize)

        # horizontal stack for the actual message passing
        indices, size = util.adj(edges, n, vertical=False)

        _, rn = size
        r = rn//n

        graph = torch.sparse.FloatTensor(indices=indices.t(), values=vals, size=size) # will this get cuda'd properly?
        self.register_buffer('graph', graph)

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, n, emb))
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases) )
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            self.bases = nn.Parameter(torch.FloatTensor(bases, n, emb))
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def forward(self, x=None, conditional=None):
        """
        :param x: E by N matrix of node embeddings.

        :return:
        """
        n, rn = self.graph.size()
        r = rn // n
        e = self.emb

        assert x is None and conditional is None

        if self.bases is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
        else:
            weights = self.weights

        assert weights.size() == (r, n, e)

        # Apply weights and sum over relations
        h = torch.mm(self.graph, weights.view(r*n, e))
        assert h.size() == (n, e)

        return h

class GAT(nn.Module):
    """
    We apply one standard self-attention (with multiple heads) to each relation, connecting only
    those nodes that are connected under that relation
    """

    def __init__(self, edges, n, emb=16, heads=4, norm_method='naive', unify='sum', dropin=False, **kwargs):
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

        self.dropin = dropin

        s = emb//heads
        r = self.relations = len(edges.keys())

        self.tokeys    = nn.Parameter(torch.FloatTensor(r, heads, s, s).uniform_(-sqrt(s), sqrt(s)))
        self.toqueries = nn.Parameter(torch.FloatTensor(r, heads, s, s).uniform_(-sqrt(s), sqrt(s)))
        self.tovals    = nn.Parameter(torch.FloatTensor(r, heads, s, s).uniform_(-sqrt(s), sqrt(s)))

        self.unify     = nn.Parameter(torch.FloatTensor(r, emb, emb).uniform_(-sqrt(emb), sqrt(emb)))

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

        if unify == 'sum':
            self.unify_rels = SumUnify()
        elif unify == 'attention':
            self.unify_rels = AttentionUnify(r, emb)
        elif unify == 'mlp':
            self.unify_rels = MLPUnify(r, emb)
        else:
            raise Exception(f'unify {unify} not recognized')

    def forward(self, x, conditional=None):
        """
        :param x: E by N matrix of node embeddings.
        :return:
        """

        n, e = x.size()
        h, r = self.heads, self.relations
        s = e // h
        ed, _ = self.mindices.size() # nr of edges total

        if conditional is not None:
            x = x + conditional

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

        if self.dropin:
            bern = ds.Bernoulli(dot)
            mask = bern.sample()
            dot = dot * mask

        values = values.view(h, r*n, s)
        output = util.batchmm(mindices, dot, self.msize, values)

        assert output.size() == (h, r*n, s)

        output = output.view(h, r, n, s).permute(2, 1, 0, 3).contiguous().view(n, r, h*s)

        # unify the heads
        output = torch.einsum('rij, nrj -> nri', self.unify, output)

        # unify the relations
        return self.unify_rels(output)
