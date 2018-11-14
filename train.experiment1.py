import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import rdflib as rdf
import pandas as pd
import numpy as np

import gzip, random, sys
from tqdm import trange

def make_batch(i2n, i2r, nbs, train, batch_size=128, length=5, dropout=0.0, start_node=None):
    """
    Create a batch by performing random walks on the given graph
    :param i2n:
    :param i2r:
    :param nbs:
    :param batch_size:
    :param length:
    :return:
    """
    nodes = torch.LongTensor(length, batch_size)
    rels  = torch.FloatTensor(length, batch_size, len(i2r))
    dirs   = torch.FloatTensor(length, batch_size, 2)

    nodes.zero_()
    rels.zero_()
    dirs.zero_()

    target = None

    if train is not None:
        target = torch.LongTensor(batch_size)
        target.zero_()

        keys = [n2i[node] for node in train.keys()]

    for b in range(batch_size):
        node = start_node if start_node is not None else random.choice(keys)

        if train is not None:
            target[b] = train[i2n[node]]

        #backwards random walk from target node
        for i in range(length-1, -1, -1):
            lnode, rel, dr = random.choice(nbs[node])

            nodes[i, b] = node
            rels[i, b, rel] = 1 # one-hot
            dirs[i, b, int(dr)] = 1

            node = lnode

        nodes[length-1, b] = 0 # len(i2n)
    nodes = F.dropout(nodes, p=dropout)

    return nodes, rels, dirs, target

file = './data/aifb/aifb_stripped.nt.gz'

graph = rdf.Graph()

if file.endswith('nt.gz'):
    with gzip.open(file, 'rb') as f:
        graph.parse(file=f, format='nt')
else:
    graph.parse(file, format=rdf.util.guess_format(file))

nodes = set()
relations = set()

for s, p, o in graph:
    nodes.add(str(s))
    nodes.add(str(o))
    relations.add(str(p))

i2n = list(nodes)
n2i = { n:i for i, n in enumerate(i2n) }

i2r = list(relations)
r2i = {r:i for i, r in enumerate(i2r) }

nbs = {}

for s, p, o in graph:
    s, p, o = n2i[str(s)], r2i[str(p)], n2i[str(o)]

    if s not in nbs:
        nbs[s] = []
    if o not in nbs:
        nbs[o] = []

    nbs[s].append((o, p, True))
    nbs[o].append((s, p, False))

print('graph loaded.')

train_file   = './data/aifb/trainingSet.tsv'
test_file    = './data/aifb/testSet.tsv'
label_header = 'label_affiliation'
nodes_header = 'person'

labels_train = pd.read_csv(train_file, sep='\t', encoding='utf8')
labels_test = pd.read_csv(test_file, sep='\t', encoding='utf8')

labels = labels_train[label_header].astype('category').cat.codes

train = {}
for nod, lab in zip(labels_train[nodes_header].values, labels):
   train[nod] = lab

labels = labels_test[label_header].astype('category').cat.codes

test = {}
for nod, lab in zip(labels_test[nodes_header].values, labels):
   test[nod] = lab

print('labels loaded.')

k = 64
num_cls = 5
epochs = 500

class Model(nn.Module):

    def __init__(self, k, num_cls, graph):
        super().__init__()

        self.i2n, self.i2r, self.nbs = graph

        self.emb = nn.Embedding(len(self.i2n)+1, k, padding_idx=0)
        # for param in self.emb.parameters():
        #     param.requires_grad = False

        self.gru = nn.GRU(input_size=(k + len(i2r) + 2), hidden_size=k)
        self.den = nn.Linear(k, num_cls)

    # def parameters(self):
    #     result = []
    #     result.extend(self.gru.parameters())
    #     result.extend(self.den.parameters())
    #     return result

    def forward(self, nodes, per_node=8):

        ns, rs, ds = [], [], []
        for node in nodes:
            n, r, d, _ = make_batch(self.i2n, self.i2r, self.nbs, None, batch_size=per_node, start_node=node)
            ns.append(n)
            rs.append(r)
            ds.append(d)

        nodes = torch.cat(ns, dim=1)
        rels  = torch.cat(rs, dim=1)
        dirs  = torch.cat(ds, dim=1)

        nodes, rels, dirs = Variable(nodes), Variable(rels), Variable(dirs)

        # Model
        emb_nodes = self.emb(nodes + 1) # +1, because idx 0 is for padding

        input = torch.cat([emb_nodes, rels, dirs], dim=2)

        _, hidden = self.gru(input)
        newemb = hidden[-1, :, :]
        tot, cls = newemb.size()

        assert tot/per_node == tot//per_node

        newemb = newemb.unsqueeze(0).view(tot//per_node, per_node, cls).transpose(0,1).mean(dim=0)

        # result = F.sigmoid(output + emb_nodes)
        return self.den(newemb)

    def inference(self, numnodes, nr, nbs, iterations=5):

        embeddings = self.emb.weight.data[1:,:]

        n, k = embeddings.size()

        # Update the embeddings by convolving with the GRU cell
        for it in trange(iterations):
            next_embeddings = torch.FloatTensor(n, k)
            next_embeddings.zero_()

            for node in range(n):
                neighbors = nbs[node]
                nn = len(neighbors)

                nodes = torch.FloatTensor(nn, k + nr + 2) # current node + incoming rel + dir
                hiddens = torch.FloatTensor(nn, k) # incoming node

                nodes.zero_()
                hiddens.zero_()

                nodes[:, :k] = embeddings[node, :].unsqueeze(0).expand(nn, k)

                for i, (innode, rel, dir) in enumerate(neighbors): # TODO get rid of loop
                    hiddens[i, :] = embeddings[innode, :]
                    nodes[i, k+rel] = 1
                    nodes[i, k+nr+int(dir)] = 1

                _, hid = self.gru(nodes.unsqueeze(0), hiddens.unsqueeze(0))
                hid = hid.squeeze().mean(dim=0)

                next_embeddings[node, :] = hid

            embeddings = next_embeddings

        return self.den(embeddings)

model = Model(k, num_cls, graph=(i2n, i2r, nbs))

opt = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train
batch = 64
pn = 8

keys = [n2i[n] for n in train.keys()]
seen = 0
for e in range(epochs):
    for f in range(0, len(keys), batch):
        t = min(len(keys), f + batch)
        nodes = keys[f:t]
        target = torch.tensor(data=[train[i2n[node]] for node in nodes], dtype=torch.long)

        opt.zero_grad()

        cls = model(nodes, per_node=pn)
        loss = F.cross_entropy(cls, target)

        loss.backward()
        opt.step()

        seen += pn * batch

    print(e, loss.item(), seen/128)

print('training finished.')

# Evaluate
clss = model.inference(len(i2n), len(i2r), nbs)
# print('inference finished.')

correct, total = 0, 0
for nod, lab in test.items():
    total += 1
    #
    # nodes, rels, dirs, _ = make_batch(i2n, i2r, nbs, None, dropout=0.0, start_node=n2i[nod])
    # out = model(nodes, rels, dirs).argmax(dim=1)
    #
    # predicted, _ = out.mode()
    # print(predicted, out)

    predicted = clss[n2i[nod]].argmax().item()

    if predicted == lab:
        correct += 1

print('test accuracy ', correct/total)



