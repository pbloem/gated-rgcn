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

class RGCNClassic(nn.Module):
    """
    Classic RGCN
    """

    def __init__(self, edges, n, numcls, emb=16, bases=None, softmax=False):

        super().__init__()

        self.emb = emb
        self.bases = bases
        self.numcls = numcls
        self.softmax = softmax

        # horizontally and vertically stacked versions of the adjacency graph
        hor_ind, hor_size = util.adj(edges, n, vertical=False)
        ver_ind, ver_size = util.adj(edges, n, vertical=True)

        _, rn = hor_size
        r = rn//n

        t = len(edges[0][0])

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / util.sum_sparse(ver_ind, vals, ver_size)
        # -- the values are the same for the horizontal and the vertically stacked adjacency matrices
        #    so we can just normalize them by the vertically stacked one and reuse for the horizontal

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

        # res = torch.mm(torch.sparse.FloatTensor(indices=ver_ind.t(), values=torch.ones(ver_ind.size(0), dtype=torch.float), size=ver_size), torch.ones((n, 1), dtype=torch.float))
        # print('.')
        # res = res.data.numpy()
        # n, bins, _ = plt.hist(res, bins =range(25) )
        # print(bins)
        # print(n)
        # plt.yscale('log', nonposy='clip')
        # plt.savefig('hist.png')
        # print('.')
        #
        # sys.exit()

        # layer 1 weights
        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(r, n, emb))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(bases, n, emb))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))


        # layer 2 weights
        if bases is None:

            self.weights2 = nn.Parameter(torch.FloatTensor(r, emb, numcls) )
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, emb, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(emb).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

    def forward(self):

        ## Layer 1

        n, rn = self.hor_graph.size()
        r = rn // n
        e = self.emb
        b, c = self.bases, self.numcls

        if self.bases1 is not None:
            # weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
            weights = torch.mm(self.comps1, self.bases1.view(b, n*e)).view(r, n, e)
        else:
            weights = self.weights1

        assert weights.size() == (r, n, e)

        # Apply weights and sum over relations
        h = torch.mm(self.hor_graph, weights.view(r*n, e))
        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        ## Layer 2

        # Multiply adjacencies by hidden
        h = torch.mm(self.ver_graph, h) # sparse mm
        h = h.view(r, n, e) # new dim for the relations

        if self.bases2 is not None:
            # weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
            weights = torch.mm(self.comps2, self.bases2.view(b, e * c)).view(r, e, c)
        else:
            weights = self.weights2

        # Apply weights, sum over relations
        # h = torch.einsum('rhc, rnh -> nc', weights, h)
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)

        if self.softmax:
            return F.softmax(h + self.bias2, dim=1)

        return h + self.bias2 #-- softmax is applied in the loss

    def penalty(self, p=2):

        assert p==2

        if self.bases is None:
            return self.weights1.pow(2).sum()

        return self.comps1.pow(p).sum() + self.bases1.pow(p).sum()

class LGCN(nn.Module):
    """
    Latent relational graph convolution. Blows up the relational edges, to boost overparametrization, and hopefully
    to create lottery ticket effects.

    Note that even if rp < r there might be such effects, since _all_ latent relations connect nodes that are originally
    connected.

    """

    def __init__(self, triples, n, numcls, emb=16, bases=None, rp=16):

        super().__init__()

        self.emb = emb
        self.bases = bases
        self.numcls = numcls

        r = len({p for _, p, _ in triples})

        # Compute the (non-relational) index pairs of connected edges, and a dense matrix of n-hot encodings of the relations)
        indices = list({(s, o) for (s, _, o) in triples})
        p2i = {(s, o): i for i, (s, o) in enumerate(indices)}
        indices = torch.tensor(indices, dtype=torch.long)
        nt, _ = indices.size()

        # Compute indices for the horizontally and vertically stacked adjacency matrices.
        # -- All edges have all relations, so we just take the indices above and repeat them a bunch of times
        s, o = indices[:, 0][None, :], indices[:, 1][None, :]
        rm = torch.arange(rp)[:, None] # relation multiplier
        se, oe = (s * rm).reshape(-1, 1), (o * rm).reshape(-1, 1)
        # -- indices multiplied by relation
        s, o = s.expand(rp, nt).reshape(-1, 1), o.expand(rp, nt).reshape(-1, 1)

        self.register_buffer('hindices', torch.cat([s, oe], dim=1))
        self.register_buffer('vindices', torch.cat([se, o], dim=1))

        self.register_buffer('nhots', torch.zeros(nt, r))
        for s, p, o in triples:
            self.nhots[p2i[(s, o)], p] = 1
        # -- filling a torch tensor this way is pretty slow. Might be better to start with a python list

        # maps relations to latent relations (one per layer)
        self.to_latent1 = nn.Linear(r, rp)
        self.to_latent2 = nn.Linear(r, rp)

        self.rp , self.r, self.n, self.nt = rp, r, n, nt

        # layer 1 weights
        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(rp, n, emb))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(rp, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(bases, n, emb))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 weights
        if bases is None:
            self.weights2 = nn.Parameter(torch.FloatTensor(rp, emb, numcls) )
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(rp, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, emb, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(emb).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

    def forward(self):
        LACT = torch.relu

        rp, r, n, nt = self.rp, self.r, self.n, self.nt

        latents1 = self.to_latent1(self.nhots)
        assert latents1.size() == (nt, rp)
        latents1 = torch.softmax(latents1, dim=1)
        latents1 = latents1.t().reshape(-1)

        # column normalize
        latents1 = latents1 / util.sum_sparse(self.hindices, latents1, (n, n * rp), row=False)

        assert self.hindices.size(0) == latents1.size(0), f'{self.indices.size()} {latents1.size()}'

        ## Layer 1
        e = self.emb
        b, c = self.bases, self.numcls

        if self.bases1 is not None:
            # weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
            weights = torch.mm(self.comps1, self.bases1.view(b, n*e)).view(rp, n, e)
        else:
            weights = self.weights1

        assert weights.size() == (rp, n, e)

        # Apply weights and sum over relations
        # h = torch.mm(hor_graph, )
        h = util.spmm(indices=self.hindices, values=latents1, size=(n, n * rp), xmatrix=weights.view(rp*n, e))
        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        ## Layer 2

        latents2 = self.to_latent2(self.nhots)
        assert latents2.size() == (nt, rp)
        latents2 = torch.softmax(latents2, dim=1)
        latents2 = latents2.t().reshape(-1)
        # latents2 = LACT(latents2)

        # row normalize
        latents2 = latents2 / util.sum_sparse(self.vindices, latents2, (n * rp, n), row=True)

        # Multiply adjacencies by hidden
        # h = torch.mm(ver_graph, h) # sparse mm
        h = util.spmm(indices=self.vindices, values=latents2, size=(n * rp, n), xmatrix=h)

        h = h.view(rp, n, e) # new dim for the relations

        if self.bases2 is not None:
            # weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
            weights = torch.mm(self.comps2, self.bases2.view(b, e * c)).view(rp, e, c)
        else:
            weights = self.weights2

        # Apply weights, sum over relations
        h = torch.einsum('rhc, rnh -> nc', weights, h)
        # h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)

        return h + self.bias2 #-- softmax is applied in the loss

    def penalty(self, p=2):

        assert p==2

        if self.bases is None:
            return self.weights1.pow(2).sum()

        return self.comps1.pow(p).sum() + self.bases1.pow(p).sum()


class RGCNEmb(nn.Module):
    """
    RGCN with single node embeddings and two GCN layers.

    """

    def __init__(self, edges, n, numcls, emb=128, h=16, bases=None, separate_emb=False):

        super().__init__()

        self.emb = emb
        self.h = h
        self.bases = bases
        self.numcls = numcls
        self.separate_emb = separate_emb

        # horizontally and vertically stacked versions of the adjacency graph
        hor_ind, hor_size = util.adj(edges, n, vertical=False)
        ver_ind, ver_size = util.adj(edges, n, vertical=True)

        rn, _ = ver_size
        r = rn//n

        t = len(edges[0][0])

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / util.sum_sparse(ver_ind, vals, ver_size)
        # -- the values are the same for the horizontal and the vertically stacked adjacency matrices
        #    so we can just normalize them by the vertically stacked one and reuse for the horizontal

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

        if separate_emb:
            self.embeddings = nn.Parameter(torch.FloatTensor(r, n, emb))  # single embedding per node
            nn.init.xavier_uniform_(self.embeddings, gain=nn.init.calculate_gain('relu'))
        else:
            self.embeddings = nn.Parameter(torch.FloatTensor(n, emb)) # single embedding per node
            nn.init.xavier_uniform_(self.embeddings, gain=nn.init.calculate_gain('relu'))

        # layer 1 weights
        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(r, emb, h))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(bases, emb, h))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 weights
        if bases is None:

            self.weights2 = nn.Parameter(torch.FloatTensor(r, h, numcls) )
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, h, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(h).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

    def forward(self):

        ## Layer 1

        rn, n = self.ver_graph.size()
        r = rn // n
        b, c = self.bases, self.numcls

        if self.bases1 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
        else:
            weights = self.weights1

        assert weights.size() == (r, self.emb, self.h)

        # apply weights first
        if self.separate_emb:
            xw = torch.einsum('rne, reh -> rnh', self.embeddings, weights).contiguous()
            # xw = self.embeddings
        else:
            xw = torch.einsum('ne, reh -> rnh', self.embeddings, weights).contiguous()

        hidden1 = torch.mm(self.hor_graph, xw.view(r*n, self.h)) # sparse mm

        assert hidden1.size() == (n, self.h)

        hidden1 = F.relu(hidden1 + self.bias1)

        ## Layer 2

        if self.bases2 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
        else:
            weights = self.weights2

        # Multiply adjacencies by hidden
        hidden2 = torch.mm(self.ver_graph, hidden1) # sparse mm
        hidden2 = hidden2.view(r, n, self.h) # new dim for the relations

        # Apply weights, sum over relations
        hidden2 = torch.einsum('rhc, rnh -> nc', weights, hidden2)

        assert hidden2.size() == (n, c)

        return hidden2 + self.bias2 #-- softmax is applied in the loss

    def penalty(self, p=2):

        assert p==2

        ep = self.embeddings.pow(2).sum()

        if self.bases is None:
            return ep + self.weights1.pow(2).sum()

        return ep + self.comps1.pow(p).sum() + self.bases1.pow(p).sum()

MULT = 100

class RGCNWeighted(nn.Module):
    """
    RGCN with single node embeddings,two GCN layers, and derived weights for each edges

    """

    def __init__(self, edges, n, numcls, emb=128, h=16, bases=None, separate_emb=False, indep=False, normalize=False, sample=False):

        super().__init__()

        self.emb = emb
        self.h = h
        self.bases = bases
        self.numcls = numcls
        self.separate_emb = separate_emb
        self.normalize = normalize
        self.sample = sample

        # horizontally and vertically stacked versions of the adjacency graph
        hor_ind, hor_size = util.adj(edges, n, vertical=False)
        ver_ind, ver_size = util.adj(edges, n, vertical=True)

        rn, _ = ver_size
        r = rn//n

        self.r, self.rn, self.n = r, rn, n

        t = len(edges[0][0])

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / util.sum_sparse(ver_ind, vals, ver_size)
        # -- the values are the same for the horizontal and the vertically stacked adjacency matrices
        #    so we can just normalize them by the vertically stacked one and reuse for the horizontal

        # hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_indices', hor_ind)

        #ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_indices', ver_ind)
        self.register_buffer('values', vals)

        if separate_emb:
            self.embeddings = nn.Parameter(torch.FloatTensor(r, n, emb))  # single embedding per node
            nn.init.xavier_uniform_(self.embeddings, gain=nn.init.calculate_gain('relu'))
        else:
            self.embeddings = nn.Parameter(torch.FloatTensor(n, emb)) # single embedding per node
            nn.init.xavier_uniform_(self.embeddings, gain=nn.init.calculate_gain('relu'))

        # layer 1 weights
        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(r, emb, h))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(bases, emb, h))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 weights
        if bases is None:

            self.weights2 = nn.Parameter(torch.FloatTensor(r, h, numcls) )
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, h, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(h).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

        # convert the edges dict to a matrix of triples
        s, o, p = [], [], []
        for pred, (sub, obj) in edges.items():
            s.extend(sub)
            o.extend(obj)
            p.extend([pred] * len(sub))

        # graph as triples
        self.register_buffer('indices', torch.tensor([s, p, o], dtype=torch.long).t())

        # for computing the attention weights
        self.indep = indep
        if indep:
            self.weights = nn.Parameter(torch.randn(self.indices.size(0)))
        else:
            self.sscore  = nn.Linear(emb, h)
            self.pscore = nn.Parameter(torch.FloatTensor(r, h))
            nn.init.xavier_uniform_(self.pscore, gain=nn.init.calculate_gain('relu'))
            self.oscore = nn.Linear(emb, h)
            # maybe h is too small?

    def edgeweights(self):

        if self.indep:
            scores = self.weights
        else:
            os = self.sscore(self.embeddings[self.indices[:, 0]])
            ps = self.pscore[self.indices[:, 1]]
            ss = self.sscore(self.embeddings[self.indices[:, 2]])

            scores = (os * ps * ss).sum(dim=1) / sqrt(self.h)

        if self.sample:
            res = T.sigmoid(scores)
            brn = ds.Bernoulli(res)

            scores = brn.sample() * res

        return scores

    def forward(self):

        ## Layer 1

        r, rn, n = self.r, self.rn, self.n
        b, c = self.bases, self.numcls

        values = self.edgeweights() # * self.values
        if self.normalize:
            values = values / util.sum_sparse(self.ver_ind, values, (rn, n))

        if self.bases1 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
        else:
            weights = self.weights1

        assert weights.size() == (r, self.emb, self.h)

        # apply weights first
        if self.separate_emb:
            xw = torch.einsum('rne, reh -> rnh', self.embeddings, weights).contiguous()
            # xw = self.embeddings
        else:
            xw = torch.einsum('ne, reh -> rnh', self.embeddings, weights).contiguous()

        # hidden1 = torch.mm(self.hor_graph, xw.view(r*n, self.h)) # sparse mm
        hidden1 = util.spmm(self.hor_indices, values, (n, n* r), xw.view(r*n, self.h))

        assert hidden1.size() == (n, self.h)

        hidden1 = F.relu(hidden1 + self.bias1)

        ## Layer 2

        if self.bases2 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
        else:
            weights = self.weights2

        # Multiply adjacencies by hidden
        # hidden2 = torch.mm(self.ver_graph, hidden1) # sparse mm
        hidden2 = util.spmm(self.ver_indices, values, (n*r, n), hidden1)

        hidden2 = hidden2.view(r, n, self.h) # new dim for the relations

        # Apply weights, sum over relations
        hidden2 = torch.einsum('rhc, rnh -> nc', weights, hidden2)

        assert hidden2.size() == (n, c)

        return hidden2 + self.bias2 #-- softmax is applied in the loss

class NodeClassifier(nn.Module):

    def __init__(self, edges, n, numcls, mixer='gcn', emb=16, depth=2, sep_emb=False, **kwargs):

        super().__init__()

        self.se = sep_emb
        r = len(edges.keys())

        if self.se:
            self.nodes = None
        else:
            self.nodes = nn.Parameter(torch.randn(n, emb))

        gblocks = [GraphBlock(edges, n, mixer, emb, gcn_first=((i==0) if self.se else False), **kwargs) for i in range(depth)]
        self.gblocks = nn.Sequential(*gblocks)

        self.cls = nn.Linear(emb, numcls)

    def forward(self, conditional=None):
        """
        Produces logits over classes for all nodes.
        :param x:
        :return:
        """

        x = self.nodes

        if conditional is not None:
            conditional = x[conditional, :][None, :]

        for block in self.gblocks:
            x = block(x, conditional=conditional)

        return self.cls(x)

class GraphBlock(nn.Module):
    """
    Combines a mixing layer (GCN or GAT) with a local feedforward, residual connection and batchnorm.

    """

    def __init__(self, edges, n, mixer='gcn', emb=16, mult=4, dropout=None, gcn_first=False,
                 res=True, norm=True, ff=True, **kwargs):

        super().__init__()

        self.res = res
        self.norm = norm

        if mixer == 'gcn':
            if gcn_first:
                self.mixer = layers.GCNFirst(edges, n, emb=emb, **kwargs)
            else:
                self.mixer = layers.GCN(edges, n, emb=emb, **kwargs)
        elif mixer == 'gat':
            self.mixer = layers.GAT(edges, n, emb=emb, **kwargs)
        else:
            raise Exception(f'Mixer {mixer} not recognized')

        if self.norm:
            self.bn1 = nn.BatchNorm1d(emb)
            self.bn2 = nn.BatchNorm1d(emb)

        if ff:
            self.ff = nn.Sequential(
                nn.Linear(emb, mult*emb), nn.ReLU(),
                nn.Linear(emb*mult, emb, bias=False)
            )
        else:
            self.bias = nn.Parameter(torch.zeros(emb) )
            self.ff = nn.Sequential(
                util.Lambda(lambda x : x + self.bias),
                nn.ReLU()
            )

        self.do = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x, conditional=None):
        """
        :param x: N by E matrix of node embeddings.

        :return:
        """
        res = x if x is not None else 0.0
        # -- No res connection if the input has separate embeddings per relation
        #    (because the mixer output won't)

        x = self.mixer(x, conditional=conditional)

        if self.res:
            x = x + res
        if self.norm:
            x = self.bn1(x)

        res = x
        x = self.ff(x)

        if self.res:
            x = x + res
        if self.norm:
            x = self.bn2(x)

        if self.do:
            x = self.do(x)

        # x = self.bn1(self.mixer(x))

        return x