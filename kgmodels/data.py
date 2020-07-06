import torch

import rdflib as rdf
from rdflib import URIRef
import pandas as pd
import gzip, random, sys, os, wget, pickle, tqdm
from collections import Counter
import util

VALPROP = 0.4
REST = '.rest'
INV  = '.inv'

S = os.sep

def st(node):
    """
    Maps an rdflib node to a unique string. We use str(node) for URIs (so they can be matched to the classes) and
    we use .n3() for everything else, so that different nodes don't become unified.

    :param node:
    :return:
    """
    if type(node) == URIRef:
        return str(node)
    else:
        return node.n3()

def add_neighbors(set, graph, node, depth=2):

    if depth == 0:
        return

    for s, p, o in graph.triples((node, None, None)):
        set.add((s, p, o))
        add_neighbors(set, graph, o, depth=depth-1)

    for s, p, o in graph.triples((None, None, node)):
        set.add((s, p, o))
        add_neighbors(set, graph, s, depth=depth-1)

def load(name, final=False, limit=None, bidir=False, prune=False):
    """
    Loads a knowledge graph dataset. Self connections are automatically added as a special relation

    :param name: Dataset name ('aifb' or 'am' at the moment)
    :param final: If true, load the canonical test set, otherwise split a validation set off from the training data.
    :param limit: If set, the number of unique relations will be limited to this value, plus one for the self-connections,
                  plus one for the remaining connections combined into a single, new relation.
    :param bidir: Whether to include inverse links for each relation
    :param prune: Whether to prune edges that are further than two steps from the target labels
    :return: A tuple containing the graph data, and the classification test and train sets:
              - edges: dictionary of edges (relation -> pair of lists cont. subject and object indices respectively)
    """
    # -- Check if the data has been cached for quick loading.
    cachefile = util.here(f'data{S}{name}{S}cache_{"fin" if final else "val"}_{"pruned" if prune else "unpruned"}.pkl')
    if os.path.isfile(cachefile) and limit is None:
        print('Using cached data.')
        with open(cachefile, 'rb') as file:
            data = pickle.load(file)
            print('Loaded.')
            return data

    print('No cache found (or relation limit is set). Loading data from scratch.')

    if name == 'aifb':
        # AIFB data (academics, affiliations, publications, etc. About 8k nodes)
        file = util.here('data/aifb/aifb_stripped.nt.gz')

        train_file = util.here('data/aifb/trainingSet.tsv')
        test_file = util.here('data/aifb/testSet.tsv')
        label_header = 'label_affiliation'
        nodes_header = 'person'

    elif name == 'am':
        # Collection of the Amsterdam Museum. Data is downloaded on first load.
        data_url = 'https://www.dropbox.com/s/1mp9aot4d9j01h9/am_stripped.nt.gz?dl=1'
        file = util.here('data/am/am_stripped.nt.gz')

        print('dataset file exists: ', os.path.isfile(file))
        if not os.path.isfile(file):
            print('Downloading AM data.')
            wget.download(data_url, file)

        train_file = util.here('data/am/trainingSet.tsv')
        test_file = util.here('data/am/testSet.tsv')
        label_header = 'label_cateogory'
        nodes_header = 'proxy'

    elif name == 'bgs':
        file = util.here('data/bgs/bgs_stripped.nt.gz')
        train_file = util.here('data/bgs/trainingSet(lith).tsv')
        test_file = util.here('data/bgs/testSet(lith).tsv')
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'

    else:
        raise Exception(f'Data {name} not recognized')

    # -- Load the classification task
    labels_train = pd.read_csv(train_file, sep='\t', encoding='utf8')
    if final:
        labels_test = pd.read_csv(test_file, sep='\t', encoding='utf8')
    else:  # split the training data into train and validation
        ltr = labels_train
        pivot = int(len(ltr) * VALPROP)

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

    # -- Parse the data with RDFLib
    graph = rdf.Graph()

    if file.endswith('nt.gz'):
        with gzip.open(file, 'rb') as f:
            graph.parse(file=f, format='nt')
    else:
        graph.parse(file, format=rdf.util.guess_format(file))

    print('RDF loaded.')

    # -- Collect all node and relation labels
    if prune:
        triples = set()
        for node in list(train.keys()) + list(test.keys()):
            add_neighbors(triples, graph, URIRef(node), depth=2)

    else:
        triples = graph

    nodes = set()
    relations = Counter()

    for s, p, o in triples:
        nodes.add(st(s))
        nodes.add(st(o))

        relations[st(p)] += 1

        if bidir:
            relations[INV + str(p)] += 1

    #print(len(nodes))
    # print(len(nodes_uri))
    # print('\n'.join(list(nodes)[:1000]))

    #sys.exit()

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
    for s, p, o in tqdm.tqdm(triples):
        s, p, o = n2i[st(s)], st(p), n2i[st(o)]

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
    edges[len(i2r)] = list(range(len(i2n))), list(range(len(i2n)))

    # for i in range(len(i2n)):
    #     edges[len(i2r)][0].append(i)
    #     edges[len(i2r)][1].append(i)

    print('Graph loaded.')

    # -- Cache the results for fast loading next time
    if limit is None:
        with open(cachefile, 'wb') as file:
            pickle.dump([edges, (n2i, i2n), (r2i, i2r), train, test], file)

    return edges, (n2i, i2n), (r2i, i2r), train, test

def load_strings(file):

    with open(file, 'r') as f:
        return [line.split() for line in f]

def load_lp_random(N=50, ptest=0.2):

    train, test = [], []

    subjects = list(range(N))
    objects  = list(range(N,   2*N))
    thirds   = list(range(2*N, 3*N))

    for s, o, t in zip(subjects, objects, thirds):
        train.append((t, 1, s))
        train.append((t, 2, o))

        dataset = train if random.random() > ptest else test

        dataset.append((s, 0, o))


    i2n = [str(ind) for ind in range(3*N)]
    n2i = {n:i for i, n in enumerate(i2n)}

    i2r = [0, 1, 2]
    r2i = {r: i for i, r in enumerate(i2r)}

    train = torch.tensor(train)
    test = torch.tensor(test)

    return train, test, (n2i, i2n), (r2i, i2r)

def load_lp(name, limit=None, bidir=False, prune=False):
    """
    Loads a knowledge graph dataset. Self connections are NOT automatically added

    :param name: Dataset name ('fb' or 'wn' at the moment)
    :param limit: If set, the number of unique relations will be limited to this value, plus one for the self-connections,
                  plus one for the remaining connections combined into a single, new relation.
    :return: two lists of triples (train, test), two pairs of dicts for nodes and relations
    """

    if name == 'random':
        return load_lp_random()

    if name == 'fb': # Freebase 15k 237
        train_file = util.here('data/fb15k237/train.txt')
        val_file = util.here('data/fb15k237/valid.txt')
        test_file = util.here('data/fb15k237/test.txt')

    elif name == 'wn':
        train_file = util.here('data/wn18rr/train.txt')
        val_file = util.here('data/wn18rr/valid.txt')
        test_file = util.here('data/wn18rr/test.txt')

    elif name == 'toy':
        train_file = util.here('data/toy/train.txt')
        val_file = util.here('data/toy/valid.txt')
        test_file = util.here('data/toy/test.txt')

    else:
        raise Exception(f'Data {name} not recognized')

    train = load_strings(train_file)
    val   = load_strings(val_file)
    test  = load_strings(test_file)

    if limit:
        train = train[:limit]
        val = val[:limit]
        test = test[:limit]

    # mappings for nodes (n) and relations (r)
    nodes, rels = set(), set()
    for triple in train + val + test:
        nodes.add(triple[0])
        rels.add(triple[1])
        nodes.add(triple[2])

    i2n, i2r = list(nodes), list(rels)
    n2i, r2i = {n:i for i, n in enumerate(nodes)}, {r:i for i, r in enumerate(rels)}

    traini, vali, testi = [], [], []

    for s, p, o in train:
        traini.append([n2i[s], r2i[p], n2i[o]])

    for s, p, o in val:
        vali.append([n2i[s], r2i[p], n2i[o]])

    for s, p, o in test:
        testi.append([n2i[s], r2i[p], n2i[o]])

    train, val, test = torch.tensor(traini), torch.tensor(vali), torch.tensor(testi)

    return train, val, test, (n2i, i2n), (r2i, i2r)

def assign(patternnode, num_nodes, entity, cls, mem, ind, sz, unique_constants=True):
    """

    :param patternnode:
    :param num_nodes:
    :param entity:
    :param cls:
    :param mem:
    :param ind: How many-th instance this is
    :param sz: Number of constants in the pattern
    :param unique_constants: Whether to add a new constant node for each instance
    :return:
    """

    if patternnode is None: # class node
        return cls

    if patternnode == 0: # entity node
        return entity

    if patternnode > 0: # new constant

        if unique_constants:
            return num_nodes - 1 + (ind * sz) + patternnode
        else:
            return num_nodes - 1 + patternnode

    if patternnode not in mem:
        inst = random.randint(0, num_nodes - 1) #
        mem[patternnode] = inst

    return mem[patternnode]

def size(bgp):

    nodes = set()
    for s, p, o in bgp:
        if s is not None and s > 0:
            nodes.add(s)
        if o is not None and o > 0:
            nodes.add(o)

    return len(nodes)

def rint(n):
    return random.randint(0, n - 1)

def fan(train=1000, test=1000, others=1000, depth=3, diffusion=5):
    """
    Creates a random classification problem with a "fan" structure

    :param train:
    :param test:
    :param others:
    :param depth:
    :return:
    """

    t = train + test
    r = depth

    edges = { rel : ([], []) for rel in range(r)}

    classes = []
    imax = 2 + test + train + others - 1 # index of the highest node

    for e in range(2, t+2):

        current = e
        cls = random.choice([0, 1])
        classes.append(cls)

        for d in range(depth):
            next = None
            for diff in range(diffusion):

                if diff == 0: # informative connection (unique node per instance)
                    if d == depth - 1:
                        other = cls
                    else:
                        imax += 1
                        next = other = imax
                else: # diffusion connection (from the pool of random nodes)
                    other = 2 + t + rint(others)

                fr, to = edges[d]

                fr.append(current)
                to.append(other)

            current = next

    # add inverse connections
    for d in range(depth):
        fr, to = edges[d]
        edges[d + depth] = to, fr

    # add self connections
    edges[depth * 2] = (list(range(imax+1)), list(range(imax+1)))

    train_idx = torch.arange(2, train+2, dtype=torch.long)
    train_lbl = torch.tensor(classes[:train],  dtype=torch.long)

    test_idx = torch.arange(2+train, 2+train+test, dtype=torch.long)
    test_lbl = torch.tensor(classes[train:], dtype=torch.long)

    return edges, imax+1, (train_idx, train_lbl), (test_idx, test_lbl)


def random_graph(base='aifb', train=1000, test=1000, bgp0=[(0, 1, -2), (-2, 2, -3), (-3, 3, None)], bgp1=[(0, 1, -2), (-2, 2, -3), (-3, 3, None)]):
    """

    Randomly wires basic graph patterns (BGP) into a base graph to generate a binary classification task. Instance nodes
    are chosen at random, and each is extended with one instance of the given BGP.

    For each node variable in the BGP, a random node is chosen from the base graph. For each variable relation, a random
    _relation_ is chosen from the base graph. For each constant node in the BGP, a new node is added to the base graph.
    For each constant _relation_, a new relation is added to the base graph. One node in the BGP represents the class of
    the instance node. This node becomes node 0 or node 1 in the BGP instance, depending on the class chosen (randomly)
    for the instance.

    :param base: Name of the knowledge graph to extend.
    :param train: Number of training instance nodes.
    :param test: Number of test instance nodes.
    :param bgp0, bgp1: Basic graph patterns for class 0 and class 1. Provided as a list of integer triples. Positive integers indicate constants,
        negative integers indicate variables. One node should be 0, which indicates the entity node, and one node should
        be None, which indicates the class node. If string, eval is called to turn into list.
    :return:
    """

    if type(bgp0) == str:
        bgp0 = eval(bgp0)
    if type(bgp1) == str:
        bgp1 = eval(bgp1)

    assert type(bgp0) == list
    assert type(bgp1) == list

    size0, size1 = size(bgp0), size(bgp1)

    if base == 'empty':
        edges = {}
        r = 0

        n = 8000
        t = train + test
        entities = list(range(t))
    else:
        edges, (_, i2n), (_, i2r), _, _= load(name=base)


        n, r, t = len(i2n), len(i2r) + 1, train+test
        entities = random.sample(range(n), t)

    classes = []

    print(f'Loaded {base} as base graph with {n} nodes and {r} relations.')

    # add random BGPs
    for i, e in enumerate(entities):

        cls, bgp, sz = random.choice([(0, bgp0, size0), (1, bgp1, size1)]) # the class of the node
        classes.append(cls)

        memn, memr = {}, {} # remembers which variable is assigned to which graph element
        for s, p, o in bgp:

            assert p != 0
            rl = assign(p, r, None, None, memr, i, sz, unique_constants=False)

            fr = assign(s, n, e, cls, memn, i, sz)
            to = assign(o, n, e, cls, memn, i, sz)

            if rl not in edges.keys():
                edges[rl] = ([], [])

            fs, ts = edges[rl]
            fs.append(fr)
            ts.append(to)

    train_idx = torch.tensor(entities[:train], dtype=torch.long)
    train_lbl = torch.tensor(classes[:train],  dtype=torch.long)

    test_idx = torch.tensor(entities[train:], dtype=torch.long)
    test_lbl = torch.tensor(classes[train:], dtype=torch.long)

    n = max([max(fs+ts) for _, (fs, ts) in edges.items()]) + 1

    return edges, n, (train_idx, train_lbl), (test_idx, test_lbl)


