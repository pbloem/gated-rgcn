from typing import List

import torch

from torch import nn
import torch.distributions as ds
import torch.nn.functional as F

from math import sqrt

import sys, random, math

import heapq

import util
from util import d, tic, toc

from itertools import accumulate

"""
TODO:

Simplified version of the sampling RGCN.

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

ACTIVATION = F.sigmoid
flatten = lambda l: [item for sublist in l for item in sublist]

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
        # -- we flip the order so that the _largest_ element gets ejected if the heap is too big.

        if len(heap) < k:
            heapq.heappush(heap, (key, value))
        else:
            heapq.heappushpop(heap, (key, value))

    return [pair[1] for pair in heap]

def wrs_gen(elem, k, weight_function):
    """
    Weighted reservoir sampling over the given generator

    :param elem: Generator over the elments
    :param k: Number of elements to sample
    :param weight_function: Function that returns a logarithmic weight given an element
    :return:
    """

    keyfunc = lambda x : - (math.log(random.random()) / math.exp(weight_function(x)))
    # -- note the negative key (since heapselect takes the smallest elems)

    return heapselect(elem, k=k, keyfunc=keyfunc)

def pad(elems, max, padelem=0, copy=False):
    ln = len(elems)
    if type(elems) is not list or copy:
        elems = list(elems)

    while len(elems) < max:
        elems.append(padelem)

    return elems, ([1] * ln) + ([0] * (max-ln))

def el2rel(triples, n):
    """
    Converts from a list of triples to a dictionary mapping to outgoing triples
    :param triples:
    :return:
    """

    if isinstance(triples, torch.Tensor):
        triples = triples.tolist()

    assert max([s for s, _, _ in triples]) <= n
    assert max([o for _, _, o in triples]) <= n

    res = {}
    for s in range(n):
        res[s] = []

    for s, p, o in triples:
        res[s].append((s, p, o))

    return res

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

def invert_graph(graph, n):
    """
    Hashes edges by object node instead of subject

    :param edges:
    :param n:
    :return: A dictionary mapping nodes to outgoing triples.
    """

    assert max(graph.keys()) <= n
    assert max(graph.keys()) <= n

    res = {}
    for s in range(n):
        res[s] = []

    for _, edges in graph.items():
        for (s, p, o) in edges:
            res[o].append((s, p, o))

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
    Batch of edges.

    Does not maintain embeddings.

    A batch of I instances represents a subgraph with I disconnected components. We call this the "batch graph". This is
    the graph over which we perform the convolution. "data indices" refer to nodes in the larger data graph, batch
    indices refer to nodes in the batch graph. Note that while the batch graph represents I subgraphs (with I target
    nodes) we think of it as a single graph with I disconnected components.

    Currently, the s and o nodes are in the same subgraph. It may be better to sample separate subgraphs for each.
    """

    def __init__(self, triples, graph, inv_graph=None):
        """

        :param triples: list lists of three integers
        :param graph:
        :param inv_graph:
        """

        if type(triples) is torch.Tensor:
            triples = triples.tolist()

        assert type(triples) is list

        # these triples should be excluded from sampling (if they exist)
        self.triples = [tuple(triple) for triple in triples] # -- convert to set of 3-tuples

        # the nodes currently sampled (a set of nodes for each instance in the batch)
        self.entities = [set([s, o]) for s, _, o in triples]

        # the edges currently sampled (set for each instance)
        self.edgesets = [set() for _ in triples]

        self.graph = graph
        self.inv_graph = invert_graph(graph) if inv_graph is None else inv_graph

    def size(self):
        """
        Number of instances in the batch
        :return:
        """
        return len(self.entities)

    def num_nodes(self):
        """
        Total number of nodes in the batch graph (over all instances)
        :return:
        """
        return sum([len(e) for e in self.entities])

    def gen_inc_edges(self, bi, rm_duplicates=True, rm_seed=True):
        """
        Generates all new incident edges (in data coordinates).

        :param bi:
        :param globals:
        :param rm_duplicates: Remove duplicates (increases memory)
        :param rm_seed: Do not generate the seed triple (the triple we're classifiying).
        :return: a generator over integer 3-tuples representing edges in data coordinates
        """

        seen = set()

        for node in self.entities[bi]:
            for edge in self.graph[node]:
                if edge not in self.edgesets[bi]: # not already sampled
                    if (not rm_duplicates) or edge not in seen: # not already seen
                        if (not rm_seed) or edge != self.triples[bi]:
                            yield edge
                            if rm_duplicates:
                                seen.add(edge)

            for edge in self.inv_graph[node]:
                if edge not in self.edgesets[bi]: # not already sampled
                    if edge[0] != edge[2]: # We've seen these already
                        if (not rm_duplicates) or edge not in seen: # not alrready seen
                            if (not rm_seed) or edge != self.triples[bi]:
                                yield edge
                                if rm_duplicates:
                                    seen.add(edge)

    def add_edges(self, edges, bi):
        """
        Add edges to the batch, for batch index bi
        :param edges:
        :param bi:
        :return:
        """

        for s, _, o in edges:
            self.entities[bi].add(s)
            self.entities[bi].add(o)

        self.edgesets[bi].update(edges)

        # new_nodes = set()
        #
        # for (s, p, o) in edges:
        #     if s not in self.nodesets[bi]:
        #         new_nodes.add(s)
        #         new_nodes.add(o)
        #
        # new_nodes = list(new_nodes)
        #
        # self.toind += [(bi, n) for n in new_nodes]
        #
        # self.frind = { (bj, e):i for i, (bj, e) in enumerate(self.toind)}
        # self.nodesets[bi].update(new_nodes)
        #
        # self.edgesets[bi].update(edges)
        #
        # for s, p, o in edges:
        #     self.edges.append((self.frind[(bi, s)], p, self.frind[(bi, o)]))

    def prune(self, edges, bi):
        """
        Returns a set containing only those edges from the given collection that
        are not already in this batch.

        :param edges:
        :return:
        """

        result = set()
        for edge in edges:
            if edge not in self.edgesets[bi]:
                result.add(edge)

        return result

    def indices(self):
        """
        Returns the data indices of the nodes in the batch graph.

        This can be used to select the node representations to be multiplied by the adjacency
        matrix.

        :return: A list of integers
        """

        return flatten([e for e in self.entities])

    def batch_triples(self):
        """
        Returns the selected triples in batch indices.


        :return:
        """

        n = sum([len(e) for e in self.entities])

        # -- create a map from data indices to batch indices (ignoring instances)
        b2d = self.indices()
        d2b = {di:bi for bi, di in enumerate(b2d)}

        # -- translate triples to batch indices (keep relations as is)
        return [(d2b[s], p, d2b[o]) for s, p, o in self.edges()]

    def edges(self):
        """
        Returns the currently selected edges in data indices.

        :return: A generator of integer triples.
        """

        for bi in range(self.size()):
            for edge in self.edgesets[bi]:
                yield edge


    def target_indices(self):
        """
        Returns the indices of the target nodes in the batch graph.

        :return: A list of pairs (subject, object) of integers. One pair for each
        instance in the batch.
        """
        firsts = [0] + [len(nodes) for _, nodes in enumerate(self.entities)]
        firsts = list(accumulate(firsts[:-1]))

        return firsts, [f+1 for f in firsts]

class SimpleClassifier(nn.Module):
    """
    The simplest version of a sampling RGCN. Two sampling layers, two RGCNs.
    """

    def __init__(self, graph, n, num_cls, emb=128, h=16, ksample=50, boost=0, bases=None, dropout=None, csample=None, **kwargs):
        super().__init__()

        self.r, self.n, self.ksample = len(graph.keys()), n, ksample

        self.graph = convert(graph, n)
        self.inv_graph = invert_graph(self.graph)
        self.edges = convert_el(graph)

        self.embeddings = nn.Parameter(torch.FloatTensor(n, h).normal_())

        self.gbias = nn.Parameter(torch.zeros((1,)))
        self.sbias = nn.Parameter(torch.zeros((self.n,)))
        self.pbias = nn.Parameter(torch.zeros((self.r,)))
        self.obias = nn.Parameter(torch.zeros((self.n,)))

        # global attention params
        self.relations = nn.Parameter(torch.randn(self.r, h))
        nn.init.xavier_uniform_(self.relations, gain=nn.init.calculate_gain('relu'))
        self.tokeys    = nn.Linear(emb, h)
        self.toqueries = nn.Linear(emb, h)

        layers = []
        layers.append(
            Sample(self.graph, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, ksample=ksample, boost=boost, csample=csample, cls=self, **kwargs))
        layers.append(
            Sample(self.graph, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, ksample=ksample, boost=boost, csample=csample, cls=self, **kwargs))
        layers.append(
            SimpleRGCN(self.graph, self.r, emb, h, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, bases=bases, dropout=dropout, cls=self, **kwargs)
        )
        layers.append(
            SimpleRGCN(self.graph, self.r, h, num_cls, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, bases=bases, dropout=dropout, cls=self, **kwargs)
        )

        self.layers = nn.ModuleList(modules=layers)

    def forward(self, batch_nodes : List[int]):

        b = len(batch_nodes)

        batch = Batch(batch_nodes, self.graph, self.embeddings, inv_graph=self.inv_graph)

        for i, layer in enumerate(self.layers):
            batch = layer(batch)

        pooled = batch.embeddings()[:b, :] # !!
        c = self.cls(pooled) # (b, num_cls)
        # -- softmax applied in loss

        return c

def distmult(s, p, o, biases=None):
    """
    Implements the distmult score function.

    :param triples: batch of triples, (b, 3) integers
    :param nodes: node embeddings
    :param relations: relation embeddings
    :return:
    """

    if biases is None:
        return (s * p * o).sum(dim=1)

    pass

    # gb, sb, pb, ob = biases
    #
    # return (s * p * o).sum(dim=1) + sb + pb + ob + gb


class SimpleLP(nn.Module):
    """
    The simplest version of sampling link prediction.
    """

    def __init__(self, triples, n, r, emb=128,
                 decoder='distmult', ksample=50, boost=0,
                 bases=None, dropout=None, csample=None, **kwargs):

        super().__init__()

        self.r, self.n, self.ksample = r, n, ksample

        self.graph = el2rel(triples, n)
        self.inv_graph = invert_graph(self.graph, n)
        self.edges = triples.tolist()

        self.embeddings = nn.Parameter(torch.FloatTensor(n, emb).normal_())

        self.gbias = nn.Parameter(torch.zeros((1,)))
        self.sbias = nn.Parameter(torch.zeros((self.n,)))
        self.pbias = nn.Parameter(torch.zeros((self.r,)))
        self.obias = nn.Parameter(torch.zeros((self.n,)))

        # global attention params
        self.relations = nn.Parameter(torch.randn(self.r, emb))
        nn.init.xavier_uniform_(self.relations, gain=nn.init.calculate_gain('relu'))
        self.tokeys    = nn.Linear(emb, emb)
        self.toqueries = nn.Linear(emb, emb)

        self.globals = {}

        self.sample0 = Sample(self.graph, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, ksample=ksample, boost=boost, csample=csample,
                cls=self, globals=self.globals, **kwargs)

        self.sample1 = Sample(self.graph, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, ksample=ksample, boost=boost, csample=csample,
                cls=self, globals=self.globals,  **kwargs)

        self.rgcn0 = SimpleRGCN(self.graph, self.r, hfr=emb, hto=emb, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, bases=bases, dropout=dropout, cls=self, **kwargs)

        self.rgcn1 = SimpleRGCN(self.graph, self.r, hfr=emb, hto=emb, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                toqueries=self.toqueries, bases=bases, dropout=dropout, cls=self, **kwargs)

        if decoder == 'distmult':
            self.decoder = distmult
        else:
            self.decoder = decoder

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
            # gb, sb, pb, ob = self.gbias, self.sbias[si], self.pbias[pi], self.obias[oi]

            semb = self.tokeys(semb)
            oemb = self.toqueries(oemb)

            dots = (semb * pemb * oemb).sum(dim=1) # + sb + pb + ob + gb
            dots = ACTIVATION(dots)

            self.globals.clear()
            for i, edge in enumerate(self.edges):
                self.globals[tuple(edge)] = dots[i].item()

    def forward(self, triples):

        assert triples.size(-1) == 3

        n, r = self.n, self.r

        dims = triples.size()[:-1]
        triples = triples.reshape(-1, 3)

        b, _ = triples.size()
        batch = Batch(triples=triples, graph=self.graph,  inv_graph=self.inv_graph)

        # Sample
        batch = self.sample0(batch)
        batch = self.sample1(batch)

        # extract batch node embeddings
        nodes = self.embeddings[batch.indices(), :]

        # compute the edge weights
        dtriples = torch.tensor(list(batch.edges()), device=d(), dtype=torch.long)
        btriples = torch.tensor(batch.batch_triples(), device=d(), dtype=torch.long)

        # adjacency matrix indices
        # -- repeans R times, vertically
        bn = batch.num_nodes()

        fr = btriples[:, 0] + bn * btriples[:, 1]
        to = btriples[:, 2]

        indices = torch.cat([fr[:, None], to[:, None]], dim=1)

        si, pi, oi = dtriples[:, 0], dtriples[:, 1], dtriples[:, 2]
        semb, pemb, oemb = self.embeddings[si, :], self.relations[pi, :], self.embeddings[oi, :]

        # compute the score (bilinear dot product)
        semb = self.tokeys(semb)
        oemb = self.toqueries(oemb)

        dots = (semb * pemb * oemb).sum(dim=1)

        values = torch.ones((indices.size(0),), device=d(), dtype=torch.float)
        values = values / util.sum_sparse(indices, values, (r * bn, bn))

        values *= ACTIVATION(dots)  # F.softplus(dots)

        # Message passing

        nodes = self.rgcn0(nodes, indices, values)
        nodes = self.rgcn1(nodes, indices, values)

        subjects, objects = batch.target_indices()

        assert len(subjects) == len(objects) == triples.size(0)

        # extract embeddings for target nodes
        s = nodes[subjects, :]
        o = nodes[objects, :]
        p = self.relations[triples[:, 1], :]

        scores = self.decoder(s, p, o)

        assert scores.size() == (util.prod(dims),)

        return scores.view(*dims)

class Sample(nn.Module):
    """
    Extends a subgraph batch by computing global self attention scores and sampling according to their magnitude.

    Samples a _fixed_ number of extra edges (if enough incident edges are available)
    """

    def __init__(self, graph, nodes=None, relations=None, tokeys=None, toqueries=None,
                 ksample=50, cls=None, csample=None, globals=None, **kwargs):
        super().__init__()

        self.graph = graph

        self.nodes = nodes
        self.relations = relations
        self.tokeys    = tokeys
        self.toqueries = toqueries

        self.ksample = ksample
        self.csample = csample

        self.gbias, self.sbias, self.pbias, self.obias = cls.gbias, cls.sbias, cls.pbias, cls.obias

        self.globals = globals

    def forward(self, batch : Batch):
        """

        :param batch:
        :param globals: Estimate of the global attention
        :return:
        """

        b = batch.size()

        for bi in range(b):

            if self.training and self.csample is not None:
                # Sample a list of candidates using the pre-computed scores
                cflat = wrs_gen(batch.gen_inc_edges(bi),
                                weight_function=lambda edge : self.globals[edge],
                                k=self.csample)
            else:
                cflat = list(batch.gen_inc_edges(bi))


            if len(cflat) == 0:
                continue

            if True: # self.training:
                # TODO: figure out how to behave in inference mode

                cflat = torch.tensor(cflat)

                # Reservoir sampling with the actual weights
                si, pi, oi = \
                    torch.tensor([s for s, _, _ in cflat], dtype=torch.long, device=d()), \
                    torch.tensor([p for _, p, _ in cflat], dtype=torch.long, device=d()), \
                    torch.tensor([o for _, _, o in cflat], dtype=torch.long, device=d())

                semb, pemb, oemb, = self.nodes[si, :], self.relations[pi, :], self.nodes[oi, :]
                # gb, sb, pb, ob = self.gbias, self.sbias[si], self.pbias[pi], self.obias[oi]

                # compute the score (bilinear dot product)
                semb = self.tokeys(semb)
                oemb = self.toqueries(oemb)

                dots = (semb * pemb * oemb).sum(dim=1) # + sb + pb + ob + gb
                dots = ACTIVATION(dots)

                # WRS with a full sort
                # -- could be optimized with a quickselect
                u = torch.rand(*dots.size(), device=d(dots))
                weights = u.log() / dots

                weights, indices = torch.sort(weights, descending=True)
                indices = indices[:self.ksample]

                cand_sampled = cflat[indices, :]
                if random.random() < 0.0:
                    print(cand_sampled.size(), cflat.size())

                cand_sampled = [(s.item(), p.item(), o.item()) for s, p, o in cand_sampled]
            else:
                cand_sampled = cflat

            batch.add_edges(cand_sampled, bi)

        return batch

class SimpleRGCN(nn.Module):
    """
    Perform RGCN over the edges of a batch of a subgraphs
    """

    def __init__(self, graph, r, hfr, hto, nodes, bases=None, dropout=None, use_global_weights=False,
                 tokeys=None, toqueries=None, relations=None, cls = None, **kwargs):
        super().__init__()

        self.r, self.hfr, self.hto = r, hfr, hto
        self.nodes = nodes # base embeddings

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, hfr, hto) )
            self.bases = None
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases))
            self.bases = nn.Parameter(torch.FloatTensor(bases, hfr, hto))

            nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))

        self.dropout = None if dropout is None else nn.Dropout(dropout)

        self.use_global_weights = use_global_weights
        self.tokeys = tokeys
        self.toqueries = toqueries
        self.relations = relations

        self.gbias, self.sbias, self.pbias, self.obias = cls.gbias, cls.sbias, cls.pbias, cls.obias


    def forward(self, nodes, indices, values):
        """
        :param btriples: Batch graph triples as a 3-tensor
        :param embeddings: Batch graph node embeddings
        :return:
        """

        n, r = nodes.size(0), self.r
        
        assert nodes.size(1) == self.hfr

        # perform message passing
        # TODO: if the input is high-dimensional, it's much more efficient to apply the weights first
        output = util.spmm(indices, values, (n * r, n), nodes)

        assert output.size() == (r * n, self.hfr), f'{output.size()} {(r * n, self.hfr)}'

        output = output.view(r, n, self.hfr)

        # compute weights from bases
        if self.bases is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps, self.bases)
        else:
            weights = self.weights

        # Apply weights
        output = torch.einsum('rij, rnj -> nri', weights, output)

        # unify the relations, activation
        output = output.sum(dim=1)
        output = F.relu(output)

        if self.dropout is not None:
            output = self.dropout(output)

        # assert batch.embeddings().size() == output.size(), f'{batch.embeddings().size()} {output.size()}'

        return output
