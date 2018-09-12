import torch

from torch import nn
import torch.functional as F
from torch.autograd import Variable

import rdflib as rdf
import pandas as pd
import numpy as np

import gzip, random
from tqdm import trange

def make_batch(i2n, i2r, nbs, train, batch_size=128, length=16):
    """
    Create a batch by performing random walks on the given graph
    :param i2n:
    :param i2r:
    :param nbs:
    :param batch_size:
    :param length:
    :return:
    """
    nodes = torch.LongTensor(batch_size, length)
    rels  = torch.FloatTensor(batch_size, length, len(i2r))
    dirs   = torch.FloatTensor(batch_size, length, 2)

    target = torch.LongTensor(batch_size, length)

    rels.zero_()
    dirs.zero_()

    for b in range(batch_size):
        node = random.randint(0, len(i2n))

        for i in range(length):
            node, rel, dr = random.choice(nbs[node])

            nodes[b, i] = node
            rels[b, i, rel] = 1 # one-hot
            dirs[b, i, int(dr)] = 1

    assert torch.max(nodes) <len(i2n)

    return nodes, rels, dirs

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
   print(nod, lab)
   train[nod] = lab

print('labels loaded.')

k = 64
num_cls = 5

emb = nn.Embedding(len(i2n), k)
gru = nn.GRU(input_size=(k + len(i2r) + 2), hidden_size=k)
den = nn.Linear(k, num_cls)

for _ in range(10):
    nodes, rels, dirs, target = make_batch(i2n, i2r, nbs, train)

    nodes, rels, dirs = Variable(rels), Variable(nodes), Variable(dirs)

    # Model
    emb_nodes = emb(nodes)
    input = torch.cat([emb_nodes, rels, dirs], dim=2)

    output = gru(input)

    result = F.sigmoid(output + emb_nodes)
    cls = F.softmax(den(result))

    loss = F.categorical_crossentropy(cls, target)



