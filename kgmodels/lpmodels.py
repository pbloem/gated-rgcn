import torch, os, sys

from torch import nn
import torch.nn.functional as F
import torch.distributions as ds

from math import sqrt, ceil

import layers, util

import torch as T

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class RGCNLayer(nn.Module):

    def __init__(self, triples, n, r, insize=None, outsize=16, decomp=None, hor=True, numbases=None, numblocks=None):

        super().__init__()

        self.decomp = decomp
        self.insize, self.outsize = insize, outsize
        self.hor = hor

        # horizontally and vertically stacked versions of the adjacency graph
        # (the vertical is always necessary to normalize the adjacencies
        if hor:
            hor_ind, hor_size = util.adj_triples(triples, n, r, vertical=False)

        ver_ind, ver_size = util.adj_triples(triples, n, r, vertical=True)
        rn, _ = ver_size
        r = rn // n

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / util.sum_sparse(ver_ind, vals, ver_size)

        if hor:
            hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
            self.register_buffer('adj', hor_graph)
        else:
            ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
            self.register_buffer('adj', ver_graph)

        h0 = n if insize is None else insize
        h1 = outsize

        # layer 1 weights
        if decomp is None:
            # -- no weight decomposition

            self.weights = nn.Parameter(torch.FloatTensor(r, h0, h1))
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

        elif decomp == 'basis':
            # -- basis decomposition

            assert numbases is not None

            self.comps = nn.Parameter(torch.FloatTensor(r, numbases))
            nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))

            self.bases = nn.Parameter(torch.FloatTensor(numbases, h0, h1))
            nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))

        elif decomp == 'block':
            # -- block decomposition

            assert numblocks is not None
            assert h0 % numblocks == 0 and h1 % numblocks == 0

            self.blocks = nn.Parameter(torch.FloatTensor(r, numblocks, h0 // numblocks, h1 // numblocks))
            nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))

        else:
            raise Exception(f'Decomposition method {decomp} not recognized')

        self.bias = nn.Parameter(torch.FloatTensor(outsize).zero_())

    def forward(self, nodes=None):

        assert (nodes is None) == (self.insize is None)

        ## Layer 1

        if self.hor:
            n, rn = self.adj.size()
        else:
            rn, n = self.adj.size()

        r = rn // n

        h0 = n if self.insize is None else self.insize
        h1 = self.outsize

        if self.decomp is None:
            weights = self.weights

        elif self.decomp == 'basis':
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)

        elif self.decomp == 'block':
            weights = util.block_diag(self.blocks)
            # TODO: multiply in block form (more efficient, but implementation differs per layer type)

        assert weights.size() == (r, h0, h1)

        if self.insize is None:
            # -- input is the identity matrix, just multiply the weights by the adjacencies
            out = torch.mm(self.adj, weights.view(r*h0, h1))

        elif self.hor:
            # -- input is high-dim and output is low dim, multiply h0 x weights first
            nodes = nodes[None, :, :].expand(r, n, h0)
            nw = torch.einsum('rni, rio -> rno', nodes, weights).contiguous()
            out = torch.mm(self.adj, nw.view(r*n, h1))

        else:
            # -- adj x h0 first, then weights
            out = torch.mm(self.adj, nodes)  # sparse mm
            out = out.view(r, n, h0)  # new dim for the relations
            out = torch.einsum('rio, rni -> no', weights, out)

        assert out.size() == (n, h1)

        return out + self.bias


def distmult(triples, nodes, relations):
    """
    Implements the distmult score function.

    :param triples: batch of triples (b, 3) integers
    :param nodes: node embeddings
    :param relations: relation embeddings
    :return:
    """

    b, _ = triples.size()
    n, k = nodes.size()
    r, k = relations.size()

    s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]
    s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

    return (s * p * o).sum(dim=1)


class LinkPrediction(nn.Module):
    """
    Classic RGCN, wired for link prediction.

    Outputs raw (linear) scores for the given triples.
    """

    def __init__(self, triples, n, r, hidden=16, out=16, decomp=None, numbases=None, numblocks=None, decoder='distmult'):

        super().__init__()

        self.layer1 = RGCNLayer(triples, n, r, insize=None, outsize=hidden, hor=True,
                                decomp=decomp, numbases=numbases, numblocks=numblocks)

        self.layer2 = RGCNLayer(triples, n, r, insize=hidden, outsize=out, hor=True,
                                decomp=decomp, numbases=numbases, numblocks=numblocks)

        self.relations = nn.Parameter(torch.FloatTensor(r, out))
        nn.init.xavier_uniform_(self.relations, gain=nn.init.calculate_gain('relu'))

        if decoder == 'distmult':
            self.decoder = distmult
        else:
            self.decoder = decoder

    def forward(self, triples):

        assert triples.size(-1) == 3

        dims = triples.size()[:-1]
        triples = triples.reshape(-1, 3)

        out = F.relu(self.layer1())
        out = self.layer2(out)

        scores = self.decoder(triples, out, self.relations)

        assert scores.size() == (util.prod(dims), )

        return scores.view(*dims)


