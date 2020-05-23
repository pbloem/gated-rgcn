import torch, os, sys

from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as ds

from math import sqrt, ceil, floor
import random

import layers, util
from util import d, tic, toc


import torch as T

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class RGCNLayer(nn.Module):

    def __init__(self, n, r, insize=None, outsize=16, decomp=None, hor=True, numbases=None, numblocks=None):
        """

        :param n:
        :param r:
        :param insize: size of the input. None if the input is one-hot vectors
        :param outsize:
        :param decomp:
        :param hor:
        :param numbases:
        :param numblocks:
        """

        super().__init__()

        self.decomp = decomp
        self.insize, self.outsize = insize, outsize
        self.hor = hor

        rn = r * n

        self.n, self.r = n, r

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
            assert h0 % numblocks == 0 and h1 % numblocks == 0, f'{h0} {h1} '

            self.blocks = nn.Parameter(torch.FloatTensor(r, numblocks, h0 // numblocks, h1 // numblocks))
            nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))

        else:
            raise Exception(f'Decomposition method {decomp} not recognized')

        self.bias = nn.Parameter(torch.FloatTensor(outsize).zero_())

    def forward(self, triples, nodes=None):

        n, r = self.n, self.r
        rn = r * n

        ## Construct the graph

        # horizontally and vertically stacked versions of the adjacency graph
        # (the vertical is always necessary to normalize the adjacencies)

        if self.hor:
            hor_ind, hor_size = util.adj_triples_tensor(triples, n, r, vertical=False)

        ver_ind, ver_size = util.adj_triples_tensor(triples, n, r, vertical=True)

        rn, _ = ver_size

        # compute values of row-normalized adjacency matrices (same for hor and ver)
        vals = torch.ones(ver_ind.size(0), dtype=torch.float, device=d(triples))
        vals = vals / util.sum_sparse(ver_ind, vals, ver_size)

        if self.hor:
            self.adj = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        else:
            self.adj = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)

        if triples.is_cuda:
            self.adj = self.adj.to('cuda')

        ## Perform message passing
        assert (nodes is None) == (self.insize is None)

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


def distmult(triples, nodes, relations, biases=None):
    """
    Implements the distmult score function.

    :param triples: batch of triples, (b, 3) integers
    :param nodes: node embeddings
    :param relations: relation embeddings
    :return:
    """

    b, _ = triples.size()

    with torch.no_grad():
        si, pi, oi = triples[:, 0], triples[:, 1], triples[:, 2]

    # s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

    # faster?
    s = nodes.index_select(dim=0,     index=si)
    p = relations.index_select(dim=0, index=pi)
    o = nodes.index_select(dim=0,     index=oi)

    if biases is None:
        return (s * p * o).sum(dim=1)

    gb, sb, pb, ob = biases

    return (s * p * o).sum(dim=1) + sb[si] + pb[pi] + ob[oi] + gb

def add_inverse_and_self(triples, n, r):
    """
    Adds inverse relations and self loops to a tensor of triples

    :param triples:
    :return:
    """
    b, _ = triples.size()

    inv = torch.cat([triples[:, 2, None], triples[:, 1, None] + r, triples[:, 0, None]], dim=1)

    assert inv.size() == (b, 3)

    all = torch.arange(n, device=d(triples))[:, None]
    id  = torch.empty(size=(n, 1), device=d(triples), dtype=torch.long).fill_(2*r)
    slf = torch.cat([all, id, all], dim=1)

    assert slf.size() == (n, 3)

    return torch.cat([triples, slf, inv], dim=0)


class LinkPrediction(nn.Module):
    """
    Classic RGCN, wired for link prediction.

    Outputs raw (linear) scores for the given triples.
    """

    def __init__(self, triples, n, r, depth=2, hidden=16, out=16, decomp=None, numbases=None, numblocks=None,
                 decoder='distmult', do=None, init=0.85, biases=False, prune=False, dropout=None, **kwargs):

        super().__init__()

        assert triples.dtype == torch.long

        self.layer0 = self.layer1 = None
        self.depth, self.prune = depth, prune
        self.n, self.r = n, r

        self.dropout = dropout

        self.register_buffer('all_triples', triples)

        if self.prune:
            self.lookup = {}
            for node in range(n):
                self.lookup[node] = set()

            for (s, p, o) in triples.tolist():
                for node in [s, o]:

                    self.lookup[node].add((s, p, o))
        else:
            # add inverse relations and self loops
            with torch.no_grad():
                self.register_buffer('all_triples_plus', add_inverse_and_self(triples, n, r))

        if depth == 0:
            self.embeddings = nn.Parameter(torch.FloatTensor(n, hidden).uniform_(-init, init))  # single embedding per node

        elif depth == 1:
            self.layer0 = RGCNLayer(n=n, r=r * 2 + 1, insize=None, outsize=out, hor=True,
                                    decomp=decomp, numbases=numbases, numblocks=numblocks, **kwargs)
        elif depth == 2:
            self.layer0 = RGCNLayer(n=n, r=r * 2 + 1, insize=None, outsize=hidden, hor=True,
                                    decomp=decomp, numbases=numbases, numblocks=numblocks, **kwargs)

            self.layer1 = RGCNLayer(n=n, r=r * 2 + 1, insize=hidden, outsize=out, hor=True,
                                    decomp=decomp, numbases=numbases, numblocks=numblocks, **kwargs)
        else:
            raise Exception('Not yet implemented.')

        self.relations = nn.Parameter(torch.FloatTensor(r, out).uniform_(-init, init))

        if decoder == 'distmult':
            self.decoder = distmult
        else:
            self.decoder = decoder


        self.do = None if do is None else nn.Dropout(do)

        self.biases = biases
        if biases:
            self.gbias = nn.Parameter(torch.zeros((1,)))
            self.sbias = nn.Parameter(torch.zeros((n,)))
            self.pbias = nn.Parameter(torch.zeros((r,)))
            self.obias = nn.Parameter(torch.zeros((n,)))

    def forward(self, batch):

        assert batch.size(-1) == 3

        n, r = self.n, self.r

        dims = batch.size()[:-1]
        batch = batch.reshape(-1, 3)
        batchl = batch.tolist()

        with torch.no_grad():

            if self.prune and self.depth > 0:
                # gather all triples that are relevant to the current batch
                triples = {tuple(t) for t in batchl}

                nds = set()
                for s, _, o in batchl:
                    nds.add(s)
                    nds.add(o)

                for _ in range(self.depth):
                #-- gather all triples that are close enough to the batch triples to be relevant

                    inc_triples = set()
                    for n in nds:
                        inc_triples.update(self.lookup[n])

                    triples.update(inc_triples)

                    nds.update([s for (s, _, _) in inc_triples])
                    nds.update([o for (_, _, o) in inc_triples])

                triples = torch.tensor(list(triples), device=d(self.all_triples), dtype=torch.long)
                with torch.no_grad():
                    triples = add_inverse_and_self(triples, n, r)
            else:
                triples = self.all_triples_plus # just use all triples

            if self.dropout is not None and self.training:
                # We drop out edges by actually removing the triples, to save on memory
                assert len(self.dropout) == 2

                keep, keepid = 1.0 - self.dropout[0], 1.0 - self.dropout[1]

                nt = triples.size(0) - n

                keep_ind = random.sample(range(nt), k=int(floor(keep * nt)) )
                keepid_ind = random.sample(range(nt, nt + n), k=int(floor(keepid * n)))
                ind = keep_ind + keepid_ind

                triples = triples[ind, :]

        nodes = self.embeddings if self.layer0 is None else self.layer0(triples=triples)

        if self.layer1 is not None:
            nodes = self.layer1(triples=triples, nodes=nodes)

        if self.do is not None:
            nodes = self.do(nodes)
            relations = self.do(self.relations)
        else:
            relations = self.relations

        if self.biases:
            biases = (self.gbias, self.sbias, self.pbias, self.obias)
        else:
            biases = None

        scores = self.decoder(batch, nodes, relations, biases=biases)

        assert scores.size() == (util.prod(dims), )

        return scores.view(*dims)


