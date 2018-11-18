import torch

import rdflib as rdf
import pandas as pd
import gzip, random, sys, os, wget, pickle, tqdm
from collections import Counter
import util

VALPROP = 0.4
REST = '.rest'

def load(name, final=False, limit=None):
    """

    :param name:
    :param final: If true, load the canonical test set, otherwise split a validation set off from the training data.
    :return:
    """
    cachefile = 'data' + os.sep + name + os.sep + 'cache.pkl'
    if os.path.isfile(cachefile) and limit is None:
        print('Using cached data.')
        with open(cachefile, 'rb') as file:
            data = pickle.load(file)
            print('Loaded.')
            return data
    print('No cache (or relation limit set), loading data.')

    if name == 'aifb':
        file = './data/aifb/aifb_stripped.nt.gz'
        train_file = './data/aifb/trainingSet.tsv'
        test_file = './data/aifb/testSet.tsv'
        label_header = 'label_affiliation'
        nodes_header = 'person'

    if name == 'am':
        data_url = 'https://www.dropbox.com/s/1mp9aot4d9j01h9/am_stripped.nt.gz?dl=1'
        file = 'data/am/am_stripped.nt.gz'

        print('dataset file exists: ', os.path.isfile(file))
        if not os.path.isfile(file):
            print('Downloading AM data.')
            wget.download(data_url, file)

        train_file = 'data/am/trainingSet.tsv'
        test_file = 'data/am/testSet.tsv'
        label_header = 'label_cateogory'
        nodes_header = 'proxy'

    graph = rdf.Graph()

    if file.endswith('nt.gz'):
        with gzip.open(file, 'rb') as f:
            graph.parse(file=f, format='nt')
    else:
        graph.parse(file, format=rdf.util.guess_format(file))

    print('RDF loaded.')

    nodes = set()
    relations = Counter()

    for s, p, o in graph:
        nodes.add(str(s))
        nodes.add(str(o))
        relations[str(p)] += 1

    if limit is not None:
        i2r = list(relations.most_common(limit)) + [REST]
    else:
        i2r = list(relations.items())

    r2i = {r: i for i, r in enumerate(i2r)}

    i2n = list(nodes)
    n2i = {n:i for i, n in enumerate(i2n)}

    edges = {}

    for s, p, o in tqdm.tqdm(graph):
        s, p, o = n2i[str(s)], str(p), n2i[str(o)]
        p = r2i[p] if p in r2i else r2i[REST]

        if p not in edges:
            edges[p] = [], []

        edges[p][0].append(s)
        edges[p][1].append(o)

    # Add self connections
    edges[len(i2r)] = [], []

    for i in range(len(i2r)):
        edges[len(i2r)][0].append(i)
        edges[len(i2r)][1].append(i)

    print('graph loaded.')

    labels_train = pd.read_csv(train_file, sep='\t', encoding='utf8')
    if final:
        labels_test = pd.read_csv(test_file, sep='\t', encoding='utf8')
    else:
        ltr = labels_train
        pivot = int(len(ltr)*VALPROP)

        labels_test = ltr[:pivot]
        labels_train = ltr[pivot:]

    labels = labels_train[label_header].astype('category').cat.codes

    train = {}
    for nod, lab in zip(labels_train[nodes_header].values, labels):
       train[nod] = lab
    labels = labels_test[label_header].astype('category').cat.codes

    test = {}
    for nod, lab in zip(labels_test[nodes_header].values, labels):
       test[nod] = lab

    print('labels loaded.')

    with open(cachefile, 'wb') as file:
        pickle.dump([edges, (n2i, i2n), (r2i, i2r), train, test], file)

    return edges, (n2i, i2n), (r2i, i2r), train, test