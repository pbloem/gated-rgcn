import torch

import rdflib as rdf
import pandas as pd
import gzip, random, sys, os, wget, pickle, tqdm
from collections import Counter
import util

VALPROP = 0.4
REST = '.rest'
INV  = 'inv.'

def load(name, final=False, limit=None, bidir=False):
    """
    Loads a knowledge graph dataset. Self connections are automatically added as a special connection

    :param name: Dataset name ('aifb' or 'am' at the moment)
    :param final: If true, load the canonical test set, otherwise split a validation set off from the training data.
    :param limit: If set, the number of unique relations will be limited to this value, plus one for the self-connections,
                  plus one for the remaining connections combined into a single, new relation.
    :param bidir: Whether to include inverse links for each relation
    :return: A tuyple containing the graph data, and the classification test and train sets.
    """
    # -- Check if the data has been cached for quick loading.
    cachefile = 'data' + os.sep + name + os.sep + 'cache.pkl'
    if os.path.isfile(cachefile) and limit is None:
        print('Using cached data.')
        with open(cachefile, 'rb') as file:
            data = pickle.load(file)
            print('Loaded.')
            return data
    print('No cache (or relation limit set), loading data.')

    if name == 'aifb':
        # AIFB data (academics, affiliations, publications, etc. About 8k nodes)
        file = './data/aifb/aifb_stripped.nt.gz'
        train_file = './data/aifb/trainingSet.tsv'
        test_file = './data/aifb/testSet.tsv'
        label_header = 'label_affiliation'
        nodes_header = 'person'

    if name == 'am':
        # Collection of the Amsterdam Museum. Data is downloaded on first load.
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

    elif name == 'bgs':
        file = 'data/bgs/bgs_stripped.nt.gz'
        train_file = 'data/bgs/trainingSet(lith).tsv'
        test_file = 'data/bgs/testSet(lith).tsv'
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'

    # -- Parse the data with RDFLib
    graph = rdf.Graph()

    if file.endswith('nt.gz'):
        with gzip.open(file, 'rb') as f:
            graph.parse(file=f, format='nt')
    else:
        graph.parse(file, format=rdf.util.guess_format(file))

    print('RDF loaded.')

    # -- Collect all node and relation labels
    nodes = set()
    relations = Counter()

    for s, p, o in graph:
        nodes.add(str(s))
        nodes.add(str(o))
        relations[str(p)] += 1

        if bidir:
            relations[INV + str(p)] += 1

    i2n = list(nodes) # maps indices to labels
    n2i = {n:i for i, n in enumerate(i2n)} # maps labels to indices

    # Truncate the list of relations if necessary
    if limit is not None:
        i2r = [r[0] for r in  relations.most_common(limit)] + [REST, INV+REST]
        # the 'limit' most frequent labels are maintained, the rest are combined into label REST to save memory
    else:
        i2r =list(relations.keys())

    r2i = {r: i for i, r in enumerate(i2r)}

    edges = {}

    # -- Collect all edges into a dictionary: relation -> (from, to)
    #    (only storing integer indices)
    for s, p, o in tqdm.tqdm(graph):
        s, p, o = n2i[str(s)], str(p), n2i[str(o)]

        pf = r2i[p] if (p in r2i) else r2i[REST]

        if pf not in edges:
            edges[pf] = [], []

        edges[pf][0].append(s)
        edges[pf][1].append(o)

        if bidir:
            pi = r2i[INV+p] if (INV+p in r2i) else r2i[INV+REST]

            if pi not in edges:
                edges[pi] = [], []

            edges[pi][0].append(o)
            edges[pi][1].append(s)

    # Add self connections explicitly
    edges[len(i2r)] = [], []

    for i in range(len(i2r)):
        edges[len(i2r)][0].append(i)
        edges[len(i2r)][1].append(i)

    print('Graph loaded.')

    # -- Load the classification task
    labels_train = pd.read_csv(train_file, sep='\t', encoding='utf8')
    if final:
        labels_test = pd.read_csv(test_file, sep='\t', encoding='utf8')
    else: # split the training data into train and validation
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

    print('Labels loaded.')

    # -- Cache the results for fast loading next time
    if limit is None:
        with open(cachefile, 'wb') as file:
            pickle.dump([edges, (n2i, i2n), (r2i, i2r), train, test], file)

    return edges, (n2i, i2n), (r2i, i2r), train, test