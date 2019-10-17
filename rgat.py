import torch
from torch import nn

import torch.nn.functional as F
import util

EPSILON = 0.00000001

class GATLayer(nn.Module):

    def __init__(self, graph, emb_size, normalize=True):
        """

        :param graph:
        :param emb_size:
        :param normalize: Apply batch norm after layer
        """
        super().__init__()

        self.i2n, self.i2r, self.edges = graph

        froms, tos = [], []
        n, r, k = len(self.i2n), len(self.i2r) + 1, emb_size

        for p in self.edges.keys():
            froms.extend(self.edges[p][0])
            tos.extend(self.edges[p][1])

        self.register_buffer('froms', torch.tensor(froms, dtype=torch.long))
        self.register_buffer('tos', torch.tensor(tos, dtype=torch.long))

        self.normalize = normalize
        if normalize:
            self.normparams = nn.Parameter(torch.tensor([1.0, 0.0]))

        self.pool = nn.Sequential(
            nn.Linear(k * r, 4 * k), nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, nodes, rels, sample=None, train=True, dense=False, do=None, dev=torch.cuda.is_available()):

        n, k = nodes.size()
        r, k = rels.size()

        if dense:

            froms = nodes[None, :, :].expand(r, n, k)
            rels = rels.permute(2, 0, 1)

            froms = torch.bmm(froms, rels)

            froms = froms.view(r * n, k)
            adj = torch.mm(froms, nodes.t())  # stacked adjacencies
            adj = F.softmax(adj, dim=0)

            nwnodes = torch.mm(adj, nodes)

        else:
            # create a length m 3-tensor by concatentating the appropriate matrix R for each edge
            rels = [rels[p:p + 1, :].expand(len(self.edges[p][0]), k) for p in range(r)]
            rels = torch.cat(rels, dim=0)

            assert len(self.froms) == rels.size(0)

            froms = nodes[self.froms, :]
            tos = nodes[self.tos, :]

            froms, tos = froms[:, None, :], tos[:, :, None]

            # unnormalized attention weights
            # print(froms.size(), rels.size(), tos.size())
            # sys.exit()
            att = torch.bmm(froms * rels[:, None, :], tos).squeeze()

            if sample is None:

                rel_indices = []
                for p in range(r):
                    rel_indices.extend([p] * len(self.edges[p][0]))
                rel_indices = torch.tensor(rel_indices, device=dev)

                assert len(rel_indices) == rels.size(0)

                fr_indices = self.froms + rel_indices * n
                indices = torch.cat([fr_indices[:, None], self.tos[:, None]], dim=1)
                values = att

            else:

                pass

            self.values = values

            # normalize the values (TODO try sparsemax)

            values = util.logsoftmax(indices, values, (n * r, n), p=10, row=True)
            values = torch.exp(values)

            mm = util.sparsemm(torch.cuda.is_available())
            nwnodes = mm(indices.t(), values, (n * r, n), nodes)

        nwnodes = nwnodes.view(r, n, k)

        # nwnodes = nwnodes.mean(dim=0)
        nwnodes = nwnodes.permute((1, 0, 2)).contiguous().view(n, r * k)
        nwnodes = self.pool(nwnodes)

        if self.normalize:
            gamma, beta = self.normparams

            mean = nwnodes.mean(dim=0, keepdim=True)
            normalized = nwnodes - mean

            var = normalized.var(dim=0, keepdim=True)
            normalized = normalized / torch.sqrt(var + EPSILON)

            nwnodes = normalized * gamma + beta

        if do is not None and train:
            nwnodes = F.dropout(nwnodes, p=do)

        return nwnodes


class Model(nn.Module):
    def __init__(self, k, num_classes, graph, depth=3, normalize=True):
        super().__init__()

        self.i2n, self.i2r, self.edges = graph
        self.num_classes = num_classes

        n = len(self.i2n)

        # relation embeddings
        self.rels = nn.Parameter(torch.randn(len(self.i2r) + 1, k))  # TODO initialize properly (like distmult?)

        # node embeddings (layer 0)
        self.nodes = nn.Parameter(torch.randn(n, k))  # TODO initialize properly (like embedding?)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(GATLayer(graph, k, normalize))

        self.toclass = nn.Sequential(
            nn.Linear(k, num_classes), nn.Softmax(dim=-1)
        )

    def forward(self, sample=None):

        nodes = self.nodes
        for layer in self.layers:
            nodes = layer(nodes, self.rels, sample=sample, train=self.training)

        return self.toclass(nodes)