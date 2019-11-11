from typing import List

import torch

from torch import nn
import torch.distributions as ds
import torch.nn.functional as F

from math import sqrt

import sys

import util
from util import d

"""
TODO:

- Check equivalence to regular RGCN
 -- set sampling boost, max_edges high
 -- make dots equal before normalizing (*= 0 should do it)

- A little more debugging
- Add local feedforwards in blocks.
- Add tokeys, etc parameters
- Optimize out the padding
- Check for further optimizations
- Run on GPU

"""

def pad(elems, max, padelem=0, copy=False):
    ln = len(elems)
    if type(elems) is not list or copy:
        elems = list(elems)

    while len(elems) < max:
        elems.append(padelem)

    return elems, ([1] * ln) + ([0] * (max-ln))

def convert(edges, num_nodes):
    """
    Convert from a relation dictionary to a edgelist representation.
    :param edges:
    :param n:
    :return: A dictionary mapping nodes to outgoing triples.
    """

    res = {n:[] for n in range(num_nodes)}

    for rel, (froms, tos) in edges.items():
        for fr, to in zip(froms, tos):
            res[fr].append((fr, rel, to))

    return res

class Batch():
    """
    Maintains all relevant information about a batch of subgraphs.
    """

    def __init__(self, entities, graph, embeddings, maskid=False):

        n, e = embeddings.size()

        self.entities = entities

        self.nodesets = [set([node]) for node in entities]
        self.edgesets = [set() for _ in entities]

        # mapping from node indices in the batch graph to indices in the original graph
        self.toind = [(bi, ent) for bi, ent in enumerate(entities)]
        self.frind = {(bi, e):i for i, (bi, ent) in enumerate(self.toind)}

        self.graph = graph

        self.orig_embeddings = embeddings

        self.node_emb = torch.zeros((len(entities), e), dtype=torch.float, device=d()) if maskid else embeddings[entities]

        # edges of all subgraphs, with local node indices
        self.edges   = []
        self.weights = None

    def size(self):
        """
        Number of batches
        :return:
        """
        return len(self.edgesets)

    def inc_edges(self, bi, prune=True):

        inc_edges = set()

        for node in self.nodesets[bi]:
            inc_edges.update(self.graph[node])

        if prune:
            return self.prune(inc_edges, bi)
        return inc_edges

    def add_edges(self, edges, bi, weights=None):

        new_nodes = set()

        for (s, p, o) in edges:
            if s not in self.nodesets[bi]:
                new_nodes.add(s)
            if o not in self.nodesets[bi]:
                new_nodes.add(o)

        new_nodes = list(new_nodes)

        self.toind += [(bi, n) for n in new_nodes]
        self.node_emb = torch.cat([self.node_emb, self.orig_embeddings[new_nodes, :]], dim=0)
        self.frind = { (bj, e):i for i, (bj, e) in enumerate(self.toind)}
        self.nodesets[bi].update(new_nodes)

        self.edgesets[bi].update(edges)

        for s, p, o in edges:
            self.edges.append((self.frind[(bi, s)], p, self.frind[(bi, o)]))

        if weights is not None:
            self.weights = torch.cat([self.weights, weights], dim=0) if self.weights is not None else weights

    def prune(self, edges, bi):
        """
        Removes any edges that are already in the batch.
        :param edges:
        :return:
        """

        result = set()
        for edge in edges:
            if edge not in self.edgesets[bi]:
                result.add(edge)

        return result

    def cflat(self):
        """
        Returns an (n, 3) tensor representing the currently selected triples,
        in batch coordinates
        :return:
        """

        return torch.tensor(self.edges, dtype=torch.long, device=d())

    def embeddings(self):
        """
        Returns embeddings for the current node selection.
        :return:
        """

        return self.node_emb

    def num_nodes(self):
        """
        Number of nodes in the batch graph.
        :return:
        """
        return self.node_emb.size(0)

class SamplingClassifier(nn.Module):

    def __init__(self, edges, n, num_cls, depth=2, emb=16, max_edges=37, boost=0, bases=None, maskid=False, dropout=None):
        super().__init__()

        self.r, self.n, self.max_edges = len(edges.keys()), n, max_edges
        self.maskid = maskid

        self.graph = convert(edges, n)

        layers =  [SampleAll(self.graph) for _ in range(depth)]
        layers += [SimpleRGCN(self.graph, self.r, emb, bases=bases, dropout=dropout) for _ in range(depth)]

        self.layers = nn.ModuleList(modules=layers)
        self.embeddings = nn.Parameter(torch.FloatTensor(n, emb).uniform_(-sqrt(emb), sqrt(emb)))

        self.cls = nn.Linear(emb, num_cls)

    def forward(self, batch_nodes : List[int]):

        b = len(batch_nodes)

        # TODO maskid

        batch = Batch(batch_nodes, self.graph, self.embeddings, maskid=self.maskid)

        for i, layer in enumerate(self.layers):
            batch = layer(batch)

        pooled = batch.embeddings()[:b, :]
        c = self.cls(pooled) # (b, num_cls)

        return c

class SamplingRGCN(nn.Module):
    """
    Combines RGCN with a global self-attention. At each layer, the self attention is computed,
    and a samples subgraph is extended by sampling extra edges based on global attention weights. Then, message passing
    (weighted by the re-normalized attention weights) is performed to update the node embeddings.

    At the end, a pooling operation over all nodes in the sampled subgraph determines the class.
    """

    def __init__(self, graph, r, emb, max, norm_method='softplus', boost=0, bases=None, sample=True, convolve=True):
        super().__init__()

        self.max = max
        self.graph = graph # edgelist representation
        self.r = r
        self.boost = boost

        self.sample, self.convolve = sample, convolve

        self.relations = nn.Parameter(torch.randn(r, emb).uniform_(-sqrt(emb), sqrt(emb)))

        self.tokeys    = nn.Parameter(torch.randn(emb, emb).uniform_(-sqrt(emb), sqrt(emb)))
        self.toqueries = nn.Parameter(torch.randn(emb, emb).uniform_(-sqrt(emb), sqrt(emb)))
        # self.tovalues  = nn.Parameter(torch.randn(emb, emb).uniform_(-sqrt(emb), sqrt(emb)))

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )
            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases).uniform_(-sqrt(bases), sqrt(bases)) )
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )

        self.norm_method = norm_method

    def forward(self, batch_nodes, batch_edges, embeddings):
        """
        :param batch_nodes:
        :param batch_edges:
        :param max:
        :param embeddings: Node embeddings from the previous layer. These are updated through message passing for the
        selected nodes, and taken from the embedding layer for the rest
        :return:
        """

        if self.sample:
            with torch.no_grad():
                # - extend the subgraph by sampling. No gradient for this part

                b, n, e = embeddings.size()
                assert b == len(batch_nodes) == len(batch_edges)

                candidates = []
                for i, (nodes, edges)  in enumerate(zip(batch_nodes, batch_edges)):
                    # collect all incident edges
                    inc_edges = set()

                    for node in nodes:
                        inc_edges.update(self.graph[node])

                    # remove already sampled edges
                    inc_edges.difference_update(edges)
                    candidates.append(inc_edges)

                # compute raw scores for all candidate edges

                # -- flatten out the candidates
                batch_idx = []
                candidates_flat = []
                for b, c in enumerate(candidates):
                    batch_idx.extend([b] * len(c))
                    candidates_flat.extend(c)

                cflat = torch.tensor(candidates_flat, device=d())
                bflat = torch.tensor(batch_idx, device=d())

                # -- select the relevant node and relation embeddings
                si, pi, oi = cflat[:, 0], cflat[:, 1], cflat[:, 2]
                semb, pemb, oemb= embeddings[bflat, si, :], self.relations[pi, :], embeddings[bflat, oi, :]

                # -- compute the score (bilinear dot product)
                semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
                oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)
                dots = (semb * pemb * oemb).sum(dim=1) / sqrt(e)

                # -- sort by score (this allows us to cut off the low-scoring edges if we sample too many)
                dots, indices = torch.sort(dots, descending=True)
                cflat = cflat[indices, :]
                bflat = bflat[indices]

                # sample candidates by score
                bern = ds.Bernoulli(torch.sigmoid(dots + self.boost))
                samples = bern.sample().to(torch.bool) # note that this is detached, no gradient here.

                cflat = cflat[samples, :]
                bflat = bflat[samples]

                # add the candidates to the batch
                for i in range(cflat.size(0)):
                    b = bflat[i].item()
                    s, p, o = [c.item() for c in cflat[i, :]]

                    if len(batch_edges[b]) < self.max:

                        batch_nodes[b].add(s)
                        batch_nodes[b].add(o)
                        batch_edges[b].add((s, p, o))

        # convert to batch of sparse graphs
        # - We pad the value and index vectors to be the same length.
        # - We recompute the dot products with the padding.
        # - We add scores as values. The message passing is mediated by these, to a gradient flows back through
        #   the global attention

        if self.convolve:
            b, n, e = embeddings.size()
            r = self.r

            batch_idx = []
            cflat = []

            for bi, edges in enumerate(batch_edges):
                batch_idx.extend([bi] * len(edges))
                cflat.extend(edges)

            cflat = torch.tensor(cflat, device=d())
            bflat = torch.tensor(batch_idx, device=d())

            si, pi, oi = cflat[:, 0], cflat[:, 1], cflat[:, 2]

            semb, pemb, oemb = embeddings[bflat, si, :], self.relations[pi, :], embeddings[bflat, oi, :]

            # -- compute the score (bilinear dot product)
            semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
            oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)
            dots = (semb * pemb * oemb).sum(dim=1) / sqrt(e)

            # convert to sparse matrices
            fr = cflat[:, 0] + n * cflat[:, 1] + (n * r) * bflat
            to = cflat[:, 2] + n * bflat
            indices = torch.cat([fr[:, None], to[:, None]], dim=1)

            # row normalize
            dots = dots * 0.0 # test eq. to RGCN   !!!!!
            dots = util.simple_normalize(indices, dots, (b*r*n, b*n), method=self.norm_method)

            # perform weighted message passing
            output = util.spmm(indices, dots, (b*n*r, b*n), embeddings.reshape(-1, e))
            assert output.size() == (b*n*r, e), f'{output.size()} {(b*n*r, e)}'

            output = output.view(b, r, n, e)

            if self.bases is not None:
                weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
            else:
                weights = self.weights

            # Apply weights
            output = torch.einsum('rij, brnj -> bnri', weights, output)

            # add original embeddings, so the unused nodes don't get zeroed out
            output = torch.cat([output, embeddings[:, :, None, :]], dim=2)

            # unify the relations
            output = output.sum(dim=2)
            output = F.relu(output)
        else:
            output = embeddings

        return batch_nodes, batch_edges, output

class SampleAllOld(nn.Module):
    """
    Extends a list of instance-subgraphs with all incident edges.
    """

    def __init__(self, graph):
        super().__init__()

        self.graph = graph

    def forward(self, batch_nodes, batch_edges, embeddings):

        b, n, e = embeddings.size()
        assert b == len(batch_nodes) == len(batch_edges)

        candidates = []
        for i, (nodes, edges) in enumerate(zip(batch_nodes, batch_edges)):
            # collect all incident edges
            inc_edges = set()

            for node in nodes:
                inc_edges.update(self.graph[node])

            nw_nodes = [s for s, _, _ in inc_edges] + [o for _, _, o in inc_edges]

            edges.update(inc_edges)
            nodes.update(nw_nodes)

        return batch_nodes, batch_edges, embeddings

class SampleAll(nn.Module):
    """
    Extends a list of instance-subgraphs with all incident edges.
    """

    def __init__(self, graph):
        super().__init__()

        self.graph = graph

    def forward(self, batch : Batch):

        b = batch.size()

        for bi in range(b):
            candidates = batch.inc_edges(bi)
            batch.add_edges(candidates, bi)

        return batch

class SampleGA(nn.Module):
    """
    Extends a list of instance-subgraphs with incident edges sampled through global attention
    """

    def __init__(self, graph):
        super().__init__()

        self.graph = graph

    def forward(self, batch_nodes, batch_edges, embeddings, global_attention=None):

        # TODO pass on GA vectors

        with torch.no_grad():
            # - extend the subgraph by sampling. No gradient for this part

            b, n, e = embeddings.size()
            assert b == len(batch_nodes) == len(batch_edges)

            candidates = []
            for i, (nodes, edges) in enumerate(zip(batch_nodes, batch_edges)):
                # collect all incident edges
                inc_edges = set()

                for node in nodes:
                    inc_edges.update(self.graph[node])

                # remove already sampled edges
                inc_edges.difference_update(edges)
                candidates.append(inc_edges)

            # compute raw scores for all candidate edges

            # -- flatten out the candidates
            batch_idx = []
            candidates_flat = []
            for b, c in enumerate(candidates):
                batch_idx.extend([b] * len(c))
                candidates_flat.extend(c)

            cflat = torch.tensor(candidates_flat, device=d())
            bflat = torch.tensor(batch_idx, device=d())

            # -- select the relevant node and relation embeddings
            si, pi, oi = cflat[:, 0], cflat[:, 1], cflat[:, 2]
            semb, pemb, oemb = embeddings[bflat, si, :], self.relations[pi, :], embeddings[bflat, oi, :]

            # -- compute the score (bilinear dot product)
            semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
            oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)
            dots = (semb * pemb * oemb).sum(dim=1) / sqrt(e)

            # -- sort by score (this allows us to cut off the low-scoring edges if we sample too many)
            dots, indices = torch.sort(dots, descending=True)
            cflat = cflat[indices, :]
            bflat = bflat[indices]

            # sample candidates by score
            bern = ds.Bernoulli(torch.sigmoid(dots + self.boost))
            samples = bern.sample().to(torch.bool)  # note that this is detached, no gradient here.

            cflat = cflat[samples, :]
            bflat = bflat[samples]

            # add the candidates to the batch
            for i in range(cflat.size(0)):
                b = bflat[i].item()
                s, p, o = [c.item() for c in cflat[i, :]]

                if len(batch_edges[b]) < self.max:
                    batch_nodes[b].add(s)
                    batch_nodes[b].add(o)
                    batch_edges[b].add((s, p, o))

        return batch_nodes, batch_edges, embeddings

class SimpleRGCN(nn.Module):
    """
    Basic RGCN on subgraphs. Ignores global attention, and performs simple message passing
    """

    def __init__(self, graph, r, emb, bases=None, dropout=None):
        super().__init__()

        self.r, self.emb = r, emb

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )
            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases).uniform_(-sqrt(bases), sqrt(bases)) )
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, batch):

        n, r, e = batch.num_nodes(), self.r, self.emb
        cflat = batch.cflat()

        # convert to sparse matrix
        fr = cflat[:, 0] + n * cflat[:, 1]
        to = cflat[:, 2]
        indices = torch.cat([fr[:, None], to[:, None]], dim=1)

        # row normalize
        values = torch.ones((indices.size(0), ), device=d(), dtype=torch.float)
        values = values / util.sum_sparse(indices, values, (r * n, n))

        # perform message passing
        output = util.spmm(indices, values, (n * r, n), batch.embeddings())

        assert output.size() == (r * n, e), f'{output.size()} {(r * n, e)}'

        output = output.view(r, n, e)

        if self.bases is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
        else:
            weights = self.weights

        # Apply weights
        output = torch.einsum('rij, rnj -> nri', weights, output)

        # unify the relations
        output = output.sum(dim=1)
        output = F.relu(output)

        if self.dropout is not None:
            output = self.dropout(output)

        assert batch.embeddings().size() == output.size(), f'{batch.embeddings().size()} {output.size()}'
        batch.node_emb = output

        return batch

class SimpleRGCNOld(nn.Module):
    """
    Basic RGCN on subgraphs. Ignores global attention, and performs simple message passing
    """

    def __init__(self, graph, r, emb, bases=None, dropout=None):
        super().__init__()

        self.r = r

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )
            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases).uniform_(-sqrt(bases), sqrt(bases)) )
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb).uniform_(-sqrt(emb), sqrt(emb)) )

        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, batch_nodes, batch_edges, embeddings, global_attention=None):

        b, n, e = embeddings.size()
        r = self.r

        util.tic()
        batch_idx = []
        cflat = []

        for bi, edges in enumerate(batch_edges):
            batch_idx.extend([bi] * len(edges))
            cflat.extend(edges)

        cflat = torch.tensor(cflat, device=d())
        bflat = torch.tensor(batch_idx, device=d())

        # convert to sparse matrices
        fr = cflat[:, 0] + n * cflat[:, 1] + (n * r) * bflat
        to = cflat[:, 2] + n * bflat
        indices = torch.cat([fr[:, None], to[:, None]], dim=1)

        # row normalize
        values = torch.ones((indices.size(0), ), device=d(), dtype=torch.float)
        values = values / util.sum_sparse(indices, values, (b * r * n, b * n))

        # perform message passing
        output = util.spmm(indices, values, (b * n * r, b * n), embeddings.reshape(-1, e))

        assert output.size() == (b * n * r, e), f'{output.size()} {(b * n * r, e)}'

        output = output.view(b, r, n, e)

        if self.bases is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
        else:
            weights = self.weights

        # Apply weights
        output = torch.einsum('rij, brnj -> bnri', weights, output)

        # unify the relations
        output = output.sum(dim=2)
        output = F.relu(output)

        if self.dropout is not None:
            output = self.dropout(output)

        return batch_nodes, batch_edges, output