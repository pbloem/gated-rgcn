import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import util

import data

import rdflib as rdf
import pandas as pd
import numpy as np

import random, sys
from tqdm import trange

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

edges, (n2i, i2n), (r2i, i2r), train, test = data.load('am', final=False)

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
depth = 5
k = 4
num_cls = len(cls)
epochs = 150
lr = 0.001

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

    def forward(self, nodes, rels, sample=None):

        n, k = nodes.size()
        k, k, r = rels.size()

        rels = [rels[None, :, :, p].expand(len(self.edges[p][0]), k, k) for p in range(r)]
        rels = torch.cat(rels, dim=0)

        assert len(self.froms) == rels.size(0)

        froms = nodes[self.froms, :]
        tos = nodes[self.tos, :]

        froms, tos = froms[:, None, :], tos[:, :, None]

        # unnormalized attention weights
        att = torch.bmm(torch.bmm(froms, rels), tos).squeeze()

        if sample is None:

            indices = torch.cat([self.froms[:, None], self.tos[:, None]], dim=1)
            values = att

        else:

            pass

        self.values = values
        self.values.retain_grad()

        # normalize the values (TODO try sparsemax)
        # values = util.absmax(indices, values, (n, n), row=True)
        values = util.logsoftmax(indices, values, (n, n), p=10, row=True)
        values = torch.exp(values)

       #  print(values[:20])

        mm = util.sparsemm(torch.cuda.is_available())

        return mm(indices.t(), values, (n, n), nodes)

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
            nodes = layer(nodes, self.rels, sample=sample)

        return self.toclass(nodes)

model = Model(k=k, depth=depth, num_classes=num_cls, graph=(i2n, i2r, edges))

if torch.cuda.is_available():
    model.cuda()
    train_lbl = train_lbl.cuda()
    test_lbl  = test_lbl.cuda()

opt = torch.optim.Adam(model.parameters(), lr=lr)

for e in range(epochs):

    opt.zero_grad()

    cls = model()[train_idx, :]

    loss = F.cross_entropy(cls, train_lbl)

    loss.backward()
    opt.step()

    print(e, loss.item(), e)

    # Evaluate
    with torch.no_grad():
        cls = model()[train_idx, :]
        agreement = cls.argmax(dim=1) == train_lbl
        accuracy = float(agreement.sum()) / agreement.size(0)

        print('   train accuracy ', float(accuracy))

        cls = model()[test_idx, :]
        agreement = cls.argmax(dim=1) == test_lbl
        accuracy = float(agreement.sum()) / agreement.size(0)

        print('   test accuracy ', float(accuracy))

print('training finished.')


