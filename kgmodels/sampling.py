from typing import List

import torch

from torch import nn
import torch.distributions as ds
import torch.nn.functional as F

from math import sqrt

import sys, random

import heapq

import util
from util import d, tic, toc

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

def heapselect(generator, keyfunc, k):
    """
    Selects the k smallest elements from the generator

    :param generator:
    :param key:
    :return:
    """

    heap = []
    heapq.heapify(heap)

    for value in generator:
        key = - keyfunc(value)
        # -- we flip the order so that the _largest_ element gest ejected if the heap is too big.

        if len(heap) < k:
            heapq.heappush(heap, (key, value))
        else:
            heapq.heappushpop(heap, (key, value))

    return [pair[1] for pair in heap]

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

def convert_el(edges):
    """
    Convert from a relation dictionary to single edgelist representation.
    :param edges:
    :param n:
    :return: A dictionary mapping nodes to outgoing triples.
    """

    res = []

    for rel, (froms, tos) in edges.items():
        for fr, to in zip(froms, tos):
            res.append((fr, rel, to))

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

    def gen_inc_edges(self, bi):
        """
        Generates all new incident edges

        :param bi:
        :param globals:
        :return:
        """

        for node in self.nodesets[bi]:
            for edge in self.graph[node]:
                if edge not in self.edgesets[bi]:
                    yield edge

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

    def batch_indices(self, inp : List[int]):
        """
        Map a list of global indices (in the data graph) to indices in the batch graph.

        If the global index i is not in the batch graph, it is mapped to i+ n, where n is the
        number of nodes in the batch graph.

        :param inp:
        :return:
        """
        n = self.node_emb.size(0)

        return [(self.frind[i][1] if i in self.frind else n + i) for i in inp]

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

    def __init__(self, graph, n, num_cls, depth=2, emb=16, max_edges=37, boost=0, bases=None, maskid=False, dropout=None, forward_mp=False, csample=None):
        super().__init__()

        self.r, self.n, self.max_edges = len(graph.keys()), n, max_edges
        self.maskid = maskid

        self.graph = convert(graph, n)
        self.edges = convert_el(graph)

        self.embeddings = nn.Parameter(torch.FloatTensor(n, emb).normal_())

        # global attention params
        self.relations = nn.Parameter(torch.randn(self.r, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)))
        self.tokeys = nn.Parameter(torch.randn(emb, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)))
        self.toqueries = nn.Parameter(torch.randn(emb, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)))

        layers = []

        for d in range(depth):
            if forward_mp and d != 0:
                layers.append(SimpleRGCN(self.graph, self.r, emb, bases=bases, dropout=dropout))

            layers.append(Sample(self.graph, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys, toqueries=self.toqueries, max_edges=max_edges, boost=boost, csample=csample))

        layers += [SimpleRGCN(self.graph, self.r, emb, bases=bases, dropout=dropout) for _ in range(depth)]

        self.layers = nn.ModuleList(modules=layers)

        self.cls = nn.Linear(emb, num_cls)

    def precompute_globals(self):
        """
        Computes global weight for each edge based on the current embeddings. These are periodically updated (i.e.
        once per epoch). These weights are gradient free, and for the batch (after sampling) the current embeddings
        are used to compute the real global weights.

        :return:
        """

        with torch.no_grad():

            n, e = self.embeddings.size()

            si, pi, oi = [s for s, _, _ in self.edges], [p for _, p, _ in self.edges], [o for _, _, o in self.edges]

            semb, pemb, oemb, = self.embeddings[si, :], self.relations[pi, :], self.embeddings[oi, :]

            # -- compute the score (bilinear dot product)
            semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
            oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)
            dots = (semb * pemb * oemb).sum(dim=1) / sqrt(e)

            self.globals = {}
            for i, edge in enumerate(self.edges):
                self.globals[edge] = dots[i].item()

    def forward(self, batch_nodes : List[int]):

        b = len(batch_nodes)

        batch = Batch(batch_nodes, self.graph, self.embeddings, maskid=self.maskid)

        for i, layer in enumerate(self.layers):
            batch = layer(batch, self.globals)

        pooled = batch.embeddings()[:b, :]
        c = self.cls(pooled) # (b, num_cls)

        return c

class SampleAll(nn.Module):
    """
    Extends a subgraph batch with all incident edges.
    """

    def __init__(self, graph, compute_weights=False, nodes=None, relations=None, tokeys=None, toqueries=None):
        super().__init__()

        self.graph = graph
        self.compute_weights = compute_weights

        if self.compute_weights:
            self.nodes = nodes
            self.relations = relations
            self.tokeys    = tokeys
            self.toqueries = toqueries

    def forward(self, batch : Batch):

        b = batch.size()
        n, e = batch.embeddings().size()

        for bi in range(b):

            candidates = batch.inc_edges(bi)
            embeddings = torch.cat([batch.embeddings(), self.nodes], dim=0) # probably expensive

            if self.compute_weights:
                cflat = list(candidates)

                si, pi, oi = [s for s, _, _ in cflat], [p for _, p, _ in cflat], [o for _, _, o in cflat]
                si, oi = batch.batch_indices(si), batch.batch_indices(oi)

                semb, pemb, oemb, = embeddings[si, :], self.relations[pi, :], embeddings[oi, :]

                # -- compute the score (bilinear dot product)
                semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
                oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)
                dots = (semb * pemb * oemb).sum(dim=1) / sqrt(e)

                weights = dots
            else:
                weights = None

            batch.add_edges(candidates, bi, weights=weights)

        return batch

class Sample(nn.Module):
    """
    Extends a subgraph batch by computing global self attention scores and sampling accoridng to their magnitude
    """

    def __init__(self, graph, nodes=None, relations=None, tokeys=None, toqueries=None, max_edges=200, boost=0.0, csample=None):
        super().__init__()

        self.graph = graph

        self.nodes = nodes
        self.relations = relations
        self.tokeys    = tokeys
        self.toqueries = toqueries

        self.max_edges = max_edges
        self.boost = boost

        self.csample = csample

    def forward(self, batch : Batch, globals):
        """

        :param batch:
        :param globals: Estimate of the global attention
        :return:
        """

        b = batch.size()
        n, e = batch.embeddings().size()

        with torch.no_grad():

            for bi in range(b):

                # we can sample this many edges (the total maximum minus the number that have already been sampled)
                max_edges = self.max_edges - len(batch.edgesets[bi])

                if max_edges <= 0:
                    continue

                # candidates = batch.inc_edges(bi)
                # cflat = list(candidates)

                if self.training and self.csample is not None:
                    # cflat.sort(key=lambda edge : - globals[edge])
                    # cflat = cflat[:self.csample]

                    cflat = heapselect(generator=batch.gen_inc_edges(bi), keyfunc=lambda edge : - globals[edge], k=self.csample)
                else:
                    cflat = list(batch.gen_inc_edges(bi))

                embeddings = torch.cat([batch.embeddings(), self.nodes], dim=0)  # probably expensive

                si, pi, oi = [s for s, _, _ in cflat], [p for _, p, _ in cflat], [o for _, _, o in cflat]
                si, oi = batch.batch_indices(si), batch.batch_indices(oi)

                semb, pemb, oemb, = embeddings[si, :], self.relations[pi, :], embeddings[oi, :]

                # compute the score (bilinear dot product)
                semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
                oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)

                dots = (semb * pemb * oemb).sum(dim=1) / e

                if self.training:

                    # sort by score (this allows us to cut off the low-scoring edges if we sample too many)
                    dots, indices = torch.sort(dots, descending=True)

                    # sample candidates by score
                    bern = ds.Bernoulli(torch.sigmoid(dots + self.boost))
                    mask = bern.sample().to(torch.bool)  # note that this is detached, no gradient here.
                    # -- The dots receive a gradient in later layers when they are used for global attention

                    # cut off the low scoring edges
                    if mask.sum() > max_edges:

                        # find the cutoff point
                        cs = mask.cumsum(dim=0) <= max_edges
                        mask = mask * cs

                    # reverse-sort the mask
                    mask = mask[indices.sort()[1]]

                    cand_sampled = []
                    for i, c in enumerate(cflat):
                        if mask[i]:
                            cand_sampled.append(c)

                else:
                    cand_sampled = cflat

                batch.add_edges(cand_sampled, bi)

        # print(f'\ntotal {toc():.4}s     {len(candidates)}     {len(cand_sampled)}')

        return batch

class SimpleRGCN(nn.Module):
    """
    Basic RGCN on subgraphs. Ignores global attention, and performs simple message passing
    """

    def __init__(self, graph, r, emb, bases=None, dropout=None, use_global_weights=False, tokeys=None, toqueries=None, relations=None):
        super().__init__()

        self.r, self.emb = r, emb

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)) )
            self.bases = None
        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases).uniform_(-1/sqrt(bases), 1/sqrt(bases)) )
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)) )

        self.dropout = None if dropout is None else nn.Dropout(dropout)

        self.use_global_weights = use_global_weights
        self.tokeys = tokeys
        self.toqueries = toqueries
        self.relations = relations


    def forward(self, batch, globals):

        n, r, e = batch.num_nodes(), self.r, self.emb
        cflat = batch.cflat()

        # convert to sparse matrix
        fr = cflat[:, 0] + n * cflat[:, 1]
        to = cflat[:, 2]
        indices = torch.cat([fr[:, None], to[:, None]], dim=1)

        # row normalize
        if self.use_global_weights:

            # recompute the global weights
            # -- these ones get gradients, so it's more efficient to recompute the small subset that
            #    requires gradients.
            si, pi, oi = [s for s, _, _ in cflat], [p for _, p, _ in cflat], [o for _, _, o in cflat]
            si, oi = batch.batch_indices(si), batch.batch_indices(oi)

            semb, pemb, oemb, = batch.embeddings[si, :], self.relations[pi, :], batch.embeddings[oi, :]

            # compute the score (bilinear dot product)
            semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
            oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)

            dots = (semb * pemb * oemb).sum(dim=1) / e

            values = F.softplus(dots)
        else:
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
