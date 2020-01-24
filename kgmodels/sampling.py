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

def invert_graph(graph):
    """
    Hashes edges by outgoing node instead of incoming

    :param edges:
    :param n:
    :return: A dictionary mapping nodes to outgoing triples.
    """

    res = {}

    for _, edges in graph.items():
        for (s, p, o) in edges:
            if o not in res:
                res[o] = []
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
    Maintains all relevant information about a batch of subgraphs.
    """

    def __init__(self, entities, graph, embeddings, inv_graph=None):

        n, e = embeddings.size()

        self.entities = entities

        self.nodesets = [set([node]) for node in entities]
        self.edgesets = [set() for _ in entities]

        # mapping from node indices in the batch graph to indices in the original graph
        self.toind = [(bi, ent) for bi, ent in enumerate(entities)]
        self.frind = {(bi, e):i for i, (bi, ent) in enumerate(self.toind)}

        self.graph = graph
        self.inv_graph = invert_graph(graph) if inv_graph is None else inv_graph

        self.orig_embeddings = embeddings

        self.node_emb = embeddings[entities]

        # edges of all subgraphs, with local (!) node indices
        self.edges   = []

    def size(self):
        """
        Number of batches
        :return:
        """
        return len(self.edgesets)

    def gen_inc_edges(self, bi, rm_duplicates=True):
        """
        Generates all new incident edges (in data coordinates).

        :param bi:
        :param globals:
        :param rm_duplicates: Remove duplicates (increases memory)
        :return:
        """

        seen = set()

        for node in self.nodesets[bi]:
            for edge in self.graph[node]:
                if edge not in self.edgesets[bi]:
                    if (not rm_duplicates) or edge not in seen:
                        yield edge
                        if rm_duplicates:
                            seen.add(edge)

        for node in self.nodesets[bi]:
            for edge in self.inv_graph[node]:
                if edge not in self.edgesets[bi]:
                    if edge[0] != edge[2]: # We've seen these already
                        if (not rm_duplicates) or edge not in seen:
                            yield edge
                            if rm_duplicates:
                                seen.add(edge)

    def inc_edges(self, bi, prune=True):

        inc_edges = set()

        for node in self.nodesets[bi]:
            inc_edges.update(self.graph[node])

        if prune:
            return self.prune(inc_edges, bi)
        return inc_edges

    def add_edges(self, edges, bi):

        new_nodes = set()

        for (s, p, o) in edges:
            if s not in self.nodesets[bi]:
                new_nodes.add(s)
            if o not in self.nodesets[bi]:
                new_nodes.add(o)

        new_nodes = list(new_nodes)

        self.toind += [(bi, n) for n in new_nodes]
        #-- for batch node b, self.toind[b] = (bi, n) represents the batch bi and data-node n

        self.node_emb = torch.cat([self.node_emb, self.orig_embeddings[new_nodes, :]], dim=0) # TODO might be faster to re-slice the whole embedding matrix
        self.frind = { (bj, e):i for i, (bj, e) in enumerate(self.toind)}
        self.nodesets[bi].update(new_nodes)

        self.edgesets[bi].update(edges)

        for s, p, o in edges:
            self.edges.append((self.frind[(bi, s)], p, self.frind[(bi, o)]))

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

        --!! note that these are in batch coordinates
        :return:
        """

        return torch.tensor(self.edges, dtype=torch.long, device=d())

    def to_data_node(self, batch_node):

        return self.toind[batch_node][1]

    def to_data_edge(self, edge):
        """
        Map an edge in the batch graph to the corresponding edge in the
        :param edge:
        :return:
        """
        if type(edge) == torch.Tensor:
            assert edge.size() == (3, )
            edge = (edge[0].item(), edge[1].item(), edge[2].item())

        s, p, o = edge
        return self.to_data_node(s), p, self.to_data_node(o)

    def embeddings(self):
        """
        Returns embeddings for the current node selection.

        The first dimension corresponds to the batch coordinates

        !! NOTE: The number of embeddings can be higher than the number of nodes in the original graph, since each batch
        can have its own embeddings.
        :return:
        """

        return self.node_emb

    def num_nodes(self):
        """
        Number of nodes in the batch graph (by local index). Again, can theoretically be higher than the nr of nodes in
        the original graph.

        :return:
        """
        return self.node_emb.size(0)

    def set_embeddings(self, nw_embs):
        """

        :return:
        """

        assert nw_embs.size() == self.node_emb.size()

        self.node_emb = nw_embs

class SamplingClassifier(nn.Module):

    def __init__(self, graph, n, num_cls, depth=2, emb=16, ksample=50, boost=0, bases=None, dropout=None, forward_mp=False, csample=None, indep=False,  **kwargs):
        super().__init__()

        self.r, self.n, self.ksample = len(graph.keys()), n, ksample

        self.graph = convert(graph, n)
        self.inv_graph = invert_graph(self.graph)
        self.edges = convert_el(graph)

        self.embeddings = nn.Parameter(torch.FloatTensor(n, emb).normal_())

        # global attention params
        self.relations = nn.Parameter(torch.randn(self.r, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)))
        self.tokeys = nn.Parameter(torch.randn(emb, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)))
        self.toqueries = nn.Parameter(torch.randn(emb, emb).uniform_(-1/sqrt(emb), 1/sqrt(emb)))

        self.indep = indep
        if indep:
            self.e2i = {edge : i for i, edge in enumerate(self.edges)}

            m = len(self.edges)
            self.edgeweights = nn.Parameter(torch.randn(m))

        layers = []

        for d in range(depth):
            if forward_mp and d != 0:
                layers.append(SimpleRGCN(self.graph, self.r, emb, bases=bases, dropout=dropout, nodes=self.embeddings,
                                         tokeys=self.tokeys, toqueries=self.toqueries, relations=self.relations,  **kwargs))

            layers.append(Sample(self.graph, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                                 toqueries=self.toqueries, ksample=ksample, boost=boost, csample=csample,
                                 indep=(self.edgeweights, self.e2i) if indep else None,
                                **kwargs))

        for _ in range(depth):
            layers += [SimpleRGCN(self.graph, self.r, emb, nodes=self.embeddings, relations=self.relations, tokeys=self.tokeys,
                                 toqueries=self.toqueries, bases=bases, dropout=dropout,
                                 indep=(self.edgeweights, self.e2i) if indep else None,
                                  **kwargs),
                       FF(emb=emb)
                   ]

        self.layers = nn.ModuleList(modules=layers)

        self.cls = nn.Linear(emb, num_cls)

        self.indep = indep

    def set(self, parm, value):

        if parm == 'incdo':

            for layer in self.layers:
                if type(layer) == Sample:
                    layer.incdo = value

        else:
            raise Exception(f'parameter {parm} not recognized.')


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

        batch = Batch(batch_nodes, self.graph, self.embeddings, inv_graph=self.inv_graph)

        for i, layer in enumerate(self.layers):
            batch = layer(batch, self.globals)

        pooled = batch.embeddings()[:b, :] # !!
        c = self.cls(pooled) # (b, num_cls)

        return c
#
# class SampleAll(nn.Module):
#     """
#     Extends a subgraph batch with all incident edges.
#     """
#
#     def __init__(self, graph, compute_weights=False, nodes=None, relations=None, tokeys=None, toqueries=None):
#         super().__init__()
#
#         self.graph = graph
#         self.compute_weights = compute_weights
#
#         if self.compute_weights:
#             self.nodes = nodes
#             self.relations = relations
#             self.tokeys    = tokeys
#             self.toqueries = toqueries
#
#     def forward(self, batch : Batch):
#
#         b = batch.size()
#         n, e = batch.embeddings().size()
#
#         for bi in range(b):
#
#             candidates = batch.inc_edges(bi)
#             embeddings = torch.cat([batch.embeddings(), self.nodes], dim=0) # probably expensive
#
#             if self.compute_weights:
#                 cflat = list(candidates)
#
#                 si, pi, oi = [s for s, _, _ in cflat], [p for _, p, _ in cflat], [o for _, _, o in cflat]
#                 si, oi = batch.batch_indices(si), batch.batch_indices(oi)
#
#                 semb, pemb, oemb, = embeddings[si, :], self.relations[pi, :], embeddings[oi, :]
#
#                 # -- compute the score (bilinear dot product)
#                 semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
#                 oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)
#                 dots = (semb * pemb * oemb).sum(dim=1) / sqrt(e)
#
#                 weights = dots
#             else:
#                 weights = None
#
#             batch.add_edges(candidates, bi, weights=weights)
#
#         return batch

class FF(nn.Module):
    """
    Simple feedwordard module with relu activation (one layer)
    """

    def __init__(self, emb=16):
        super().__init__()

        self.lin = nn.Linear(emb, emb)

    def forward(self, batch : Batch, globals=None):

        batch.node_emb = F.relu(self.lin(batch.node_emb))

        return batch

class Sample(nn.Module):
    """
    Extends a subgraph batch by computing global self attention scores and sampling according to their magnitude.

    Samples a _fixed_ number of extra edges (if enough incident edges are available)
    """

    def __init__(self, graph, nodes=None, relations=None, tokeys=None, toqueries=None, ksample=50, csample=None, indep=None, **kwargs):
        super().__init__()

        self.graph = graph

        self.nodes = nodes
        self.relations = relations
        self.tokeys    = tokeys
        self.toqueries = toqueries

        self.ksample = ksample

        self.csample = csample

        self.indep  = False
        if indep is not None:
            self.attn, self.e2i = indep
            self.indep = True


    def forward(self, batch : Batch, globals):
        """

        :param batch:
        :param globals: Estimate of the global attention
        :return:
        """

        b = batch.size()
        n, e = batch.embeddings().size()

        for bi in range(b):

            # candidates = batch.inc_edges(bi)
            # cflat = list(candidates)

            tic()
            if self.training and self.csample is not None:
                # Sample a list of candidates using the pre-computed scores

                # cflat = heapselect(generator=batch.gen_inc_edges(bi, do=self.incdo), keyfunc=lambda edge : - globals[edge], k=self.csample)
                cflat = wrs_gen(batch.gen_inc_edges(bi),
                                weight_function=lambda edge : globals[edge],
                                k=self.csample)
            else:
                cflat = list(batch.gen_inc_edges(bi))

            if self.training:

                if self.indep:

                    edge_indices = [self.e2i[edge] for edge in cflat]
                    dots = self.attn[edge_indices]

                    cflat = torch.tensor(cflat)

                else:

                    cflat = torch.tensor(cflat)

                    # Reduce the candidates further by reservoir sampling with the actual weights
                    embeddings = self.nodes # raw embeddings (not batch embeddings)

                    si, pi, oi = [s for s, _, _ in cflat], [p for _, p, _ in cflat], [o for _, _, o in cflat]

                    # print(max(si), max(pi), max(oi))
                    # print(embeddings.size())

                    semb, pemb, oemb, = embeddings[si, :], self.relations[pi, :], embeddings[oi, :]

                    # compute the score (bilinear dot product)
                    semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
                    oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)

                    dots = (semb * pemb * oemb).sum(dim=1) / e

                # WRS with a full sort (optimize later)
                u = torch.rand(*dots.size(), device=d(dots))
                weights = u.log() / dots.exp()

                if bi == 0 and random.random() < 0.0:
                    print(batch.entities[bi])
                    print(cflat)
                    print(f'{weights.mean().item():.04}, {weights.std().item():.04}' ,weights[:10])

                weights, indices = torch.sort(weights, descending=True)
                indices = indices[:self.ksample]

                cand_sampled = cflat[indices, :]
                cand_sampled = [(s.item(), p.item(), o.item()) for s, p, o in cand_sampled]

            else:
                cflat = torch.tensor(cflat)
                cand_sampled = cflat

            batch.add_edges(cand_sampled, bi)

        return batch

class SimpleRGCN(nn.Module):
    """
    Perform RGCN over the edges of a batch of a subgraphs

    """

    def __init__(self, graph, r, emb, nodes, bases=None, dropout=None, use_global_weights=False, tokeys=None, toqueries=None, relations=None, indep=None, **kwargs):
        super().__init__()

        self.r, self.emb = r, emb
        self.nodes = nodes # base embeddings

        if bases is None:
            self.weights = nn.Parameter(torch.FloatTensor(r, emb, emb) )
            self.bases = None
            nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

        else:
            self.comps = nn.Parameter(torch.FloatTensor(r, bases))
            self.bases = nn.Parameter(torch.FloatTensor(bases, emb, emb))
            nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))

        self.dropout = None if dropout is None else nn.Dropout(dropout)

        self.use_global_weights = use_global_weights
        self.tokeys = tokeys
        self.toqueries = toqueries
        self.relations = relations

        self.indep  = False
        if indep is not None:
            self.attn, self.e2i = indep
            self.indep = True

    def forward(self, batch, globals):

        n, r, e = batch.num_nodes(), self.r, self.emb
        cflat = batch.cflat()
        #-- NB: these are batch indices

        # convert to sparse matrix
        fr = cflat[:, 0] + n * cflat[:, 1]
        to = cflat[:, 2]
        indices = torch.cat([fr[:, None], to[:, None]], dim=1)

        # row normalize
        if self.use_global_weights:

            if self.indep:
                edge_indices = [self.e2i[batch.to_data_edge(edge)] for edge in cflat]

                dots = self.attn[edge_indices]

            else:
                # use weighted message passing

                # recompute the global weights
                # -- these ones get gradients, so it's more efficient to recompute the small subset that
                #    requires gradients.

                embeddings = batch.embeddings()

                #si, pi, oi = [s for s, _, _ in cflat], [p for _, p, _ in cflat], [o for _, _, o in cflat]

                si, pi, oi = cflat[:, 0], cflat[:, 1], cflat[:, 0]

                try:
                    semb, pemb, oemb = embeddings[si, :], self.relations[pi, :], embeddings[oi, :]
                except Exception as e:
                    print(si.max(), pi.max(), oi.max())
                    print(embeddings.size())

                    raise e

                # compute the score (bilinear dot product)
                semb = torch.einsum('ij, nj -> ni', self.tokeys, semb)
                oemb = torch.einsum('ij, nj -> ni', self.toqueries, oemb)

                dots = (semb * pemb * oemb).sum(dim=1) / e

            values = dots.exp() # F.softplus(dots)
        else:
            values = torch.ones((indices.size(0), ), device=d(), dtype=torch.float)

        values = values / util.sum_sparse(indices, values, (r * n, n))

        # values.retain_grad()
        # self.values = values
        # self.register_backward_hook(lambda gin, gout: print(values.grad))

        # perform message passing
        output = util.spmm(indices, values, (n * r, n), batch.embeddings())

        if random.random() < 0.01:
            # print(semb.mean(dim=1), pemb.mean(dim=1), oemb.mean(dim=1))
            # print((semb * pemb * oemb).sum(dim=1) / e)
            # print(((semb * pemb * oemb).sum(dim=1) / e).exp())
            #

            print(dots)
            print(dots.exp())
            # print(torch.sparse.FloatTensor(indices.t(), values, (n * r, n)).to_dense())

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
        batch.set_embeddings(output)

        return batch
