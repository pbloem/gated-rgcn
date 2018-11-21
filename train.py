import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import util

import data

import rdflib as rdf
import pandas as pd
import numpy as np

import random, sys, tqdm
from tqdm import trange

from argparse import ArgumentParser

EPSILON = 0.000000001

def go(arg):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    edges, (n2i, i2n), (r2i, i2r), train, test = \
        data.load(arg.name, final=arg.final, limit=arg.limit, bidir=not arg.unidir)

    # Convert test and train to tensors
    train_idx = [n2i[name] for name, _ in train.items()]
    train_lbl = [cls for _, cls in train.items()]
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=dev)
    train_lbl = torch.tensor(train_lbl, dtype=torch.long, device=dev)

    test_idx = [n2i[name] for name, _ in test.items()]
    test_lbl = [cls for _, cls in test.items()]
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=dev)
    test_lbl = torch.tensor(test_lbl, dtype=torch.long, device=dev)

    # count nr of classes
    cls = set([int(l) for l in test_lbl] + [int(l) for l in train_lbl])

    """
    Define model
    """
    num_cls = len(cls)

    class GATLayer(nn.Module):

        def __init__(self, graph):
            super().__init__()

            self.i2n, self.i2r, self.edges = graph

            froms, tos = [], []

            for p in edges.keys():
                froms.extend(edges[p][0])
                tos.extend(edges[p][1])

            self.register_buffer('froms', torch.tensor(froms, dtype=torch.long))
            self.register_buffer('tos',  torch.tensor(tos, dtype=torch.long))

            if arg.normalize:
                self.normparams = nn.Parameter(torch.tensor([1.0, 0.0]))

        def forward(self, nodes, rels, sample=None, train=True):

            n, k = nodes.size()
            k, k, r = rels.size()

            if arg.dense:

                froms = nodes[None, :, :].expand(r, n, k)
                rels = rels.permute(2, 0, 1)

                froms = torch.bmm(froms, rels)

                froms = froms.view(r*n, k)
                adj = torch.mm(froms, nodes.t()) # stacked adjacencies
                adj = F.softmax(adj, dim=0)

                nwnodes = torch.mm(adj, nodes)

            else:
                # create a length m 3-tensor by concatentating the appropriate matrix R for each edge
                rels = [rels[None, :, :, p].expand(len(self.edges[p][0]), k, k) for p in range(r)]
                rels = torch.cat(rels, dim=0)

                assert len(self.froms) == rels.size(0)

                rel_indices = []
                for p in range(r):
                    rel_indices.extend([p] * len(self.edges[p][0]))
                rel_indices = torch.tensor(rel_indices, device=dev)

                assert len(rel_indices) == rels.size(0)

                froms = nodes[self.froms, :]
                tos = nodes[self.tos, :]

                froms, tos = froms[:, None, :], tos[:, :, None]

                # unnormalized attention weights
                att = torch.bmm(torch.bmm(froms, rels), tos).squeeze()

                if sample is None:
                    fr_indices = self.froms + rel_indices * n
                    indices = torch.cat([fr_indices[:, None], self.tos[:, None]], dim=1)
                    values = att

                else:

                    pass

                self.values = values
                self.values.retain_grad()

                # normalize the values (TODO try sparsemax)

                values = util.logsoftmax(indices, values, (n*r, n), p=10, row=True)
                values = torch.exp(values)

                mm = util.sparsemm(torch.cuda.is_available())
                nwnodes = mm(indices.t(), values, (n*r, n), nodes)

            nwnodes = nwnodes.view(r, n, k)
            nwnodes = nwnodes.mean(dim=0)

            if arg.normalize:
                gamma, beta = self.normparams

                mean = nwnodes.mean(dim=0, keepdim=True)
                normalized = nwnodes - mean

                var = normalized.var(dim=0, keepdim=True)
                normalized = normalized / torch.sqrt(var + EPSILON)

                nwnodes = normalized * gamma + beta

            if arg.do is not None and train:
                nwnodes = F.dropout(nwnodes, p=arg.do)

            return nwnodes

    class Model(nn.Module):

        def __init__(self, k, num_classes, graph, depth=3):
            super().__init__()

            self.i2n, self.i2r, self.edges = graph
            self.num_classes = num_classes

            n = len(self.i2n)

            # relation embeddings
            self.rels = nn.Parameter(torch.randn(k, k, len(self.i2r) + 1)) # TODO initialize properly (like distmult?)

            # node embeddings (layer 0)
            self.nodes = nn.Parameter(torch.randn(n, k)) # TODO initialize properly (like embedding?)

            self.layers = nn.ModuleList()
            for _ in range(depth):
                self.layers.append(GATLayer(graph))

            self.toclass = nn.Sequential(
                nn.Linear(k, num_classes), nn.Softmax(dim=-1)
            )

        def forward(self, sample=None):

            nodes = self.nodes
            for layer in self.layers:
                nodes = layer(nodes, self.rels, sample=sample, train=self.training)

            return self.toclass(nodes)

    model = Model(k=arg.emb_size, depth=arg.depth, num_classes=num_cls, graph=(i2n, i2r, edges))

    if torch.cuda.is_available():
        model.cuda()
        train_lbl = train_lbl.cuda()
        test_lbl  = test_lbl.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=arg.lr)

    for e in tqdm.trange(arg.epochs):

        model.train(True)

        opt.zero_grad()

        cls = model()[train_idx, :]

        loss = F.cross_entropy(cls, train_lbl)

        loss.backward()
        opt.step()

        print(e, loss.item(), e)

        # Evaluate
        with torch.no_grad():

            model.train(False)
            cls = model()[train_idx, :]
            agreement = cls.argmax(dim=1) == train_lbl
            accuracy = float(agreement.sum()) / agreement.size(0)

            print('   train accuracy ', float(accuracy))

            cls = model()[test_idx, :]
            agreement = cls.argmax(dim=1) == test_lbl
            accuracy = float(agreement.sum()) / agreement.size(0)

            print('   test accuracy ', float(accuracy))

    print('training finished.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Size (nr of dimensions) of the input.",
                        default=150, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Nr of attention layers.",
                        default=4, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="emb_size",
                        help="Size (nr of dimensions) of the node embeddings.",
                        default=16, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("--do",
                        dest="do",
                        help="Dropout",
                        default=None, type=float)

    parser.add_argument("-D", "--dataset-name",
                        dest="name",
                        help="Name of dataset to use [aifb, am]",
                        default='aifb', type=str)

    parser.add_argument("-N", "--normalize",
                        dest="normalize",
                        help="Normalize the embeddings.",
                        action="store_true")

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("-U", "--unidir", dest="unidir",
                        help="Only model relations in one direction.",
                        action="store_true")

    parser.add_argument("--dense", dest="dense",
                        help="Use a dense adjacency matrix with the canonical softmax.",
                        action="store_true")

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit the number of relations.",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

