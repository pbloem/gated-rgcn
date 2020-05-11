from _context import kgmodels

from kgmodels import util
from util import d

import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import rdflib as rdf
import pandas as pd
import numpy as np

import random, sys, tqdm, math, random
from tqdm import trange

import rgat

from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

"""
Full batch RGCN training for link prediction

TODO:
- implement inverse relations, self-loops, edge dropout 

"""

EPSILON = 0.000000001

global repeats

def corrupt(batch, n):
    """
    Corrupts the negatives of a batch of triples (in place). The first copy of the triples is uncorrupted
    TODO: vectorize if this turns out to be a bottleneck

    :param batch_size:
    :param n: nr of nodes in the graph

    :return:
    """
    bs, ns, _ = batch.size()

    for b in range(bs):
        for n in range(1, ns):
            i = random.choice([0, 2]) # index of the part to corrupt
            batch[b, n, i] = random.choice(range(n))


def filter(rawtriples, all, true):
    filtered = []

    for triple in rawtriples:
        if triple == true or not triple in all:
            filtered.append(triple)

    return filtered


def prt(str, end='\n'):
    if repeats == 1:
        print(str, end=end)

def go(arg):

    global repeats
    repeats = arg.repeats

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_accs = []
    test_accs = []

    train, test, (n2i, i2n), (r2i, i2r) = \
        kgmodels.load_lp(arg.name, final=arg.final)

    print(len(i2n), 'nodes')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')

    # set of all triples (for filtering)
    alltriples = set()
    for s, p, o in torch.cat([train, test], dim=0):
        s, p, o = s.item(), p.item(), o.item()

        alltriples.add((s, p, o))

    if arg.decomp == 'block':
        # -- pad the node list to make it divisible by the nr. of blocks

        added = 0
        while len(i2n) % arg.num_blocks != 0:
            label = 'null' + str(added)
            i2n.append(label)
            n2i[label] = len(i2n) - 1

            added += 1

        print(f'nodes padded to {len(i2n)} to make it divisible by {arg.num_blocks} (added {added} null nodes).')

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):

        """
        Define model
        """
        if arg.model == 'classic':
            model = kgmodels.LinkPrediction(
                triples=train, n=len(i2n), r=len(i2r), hidden=arg.emb, out=arg.emb, decomp=arg.decomp,
                numbases=arg.num_bases, numblocks=arg.num_blocks)
        elif arg.model == 'emb':
            pass
        elif arg.model == 'weighted':
            pass
        else:
            raise Exception(f'model not recognized: {arg.model}')

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()

        if arg.opt == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.wd)
        elif arg.opt == 'adamw':
            opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.wd)
        else:
            raise Exception

        # nr of negatives sampled
        ng = arg.negative_rate

        seen = 0
        for e in range(arg.epochs):

            seeni = 0
            for fr in trange(0, train.size(0), arg.batch):

                if arg.limit is not None and seeni > arg.limit:
                    break

                to = min(train.size(0), fr + arg.batch)

                positives = train[fr:to]

                b, _ = positives.size()

                # sample negatives
                triples = positives[:, None, :].expand(b, ng + 1, 3).contiguous()
                corrupt(triples, len(i2n))

                labels = torch.cat([torch.ones(b, 1), torch.zeros(b, ng)], dim=1)

                if torch.cuda.is_available():
                    triples, labels = triples.cuda(), labels.cuda()

                opt.zero_grad()

                out = model(triples)

                loss = F.binary_cross_entropy_with_logits(out, labels)

                if arg.l2weight is not None:
                    l2 = sum([p.pow(2).sum() for p in model.parameters()])
                    loss = loss + arg.l2weight * l2

                if torch.cuda.is_available():
                    print(f'\nPeak gpu memory use is {torch.cuda.max_memory_cached() / 1e9:.2} Gb')

                loss.backward()

                opt.step()

                seen += b; seeni += b

            # Evaluate
            if e % arg.eval_int == 0:
                with torch.no_grad():

                    mrr = hitsat1 = hitsat3 = hitsat10 = 0.0

                    if arg.eval_size is None:
                        testsub = test
                    else:
                        testsub = test[random.sample(range(test.size(0)), k=arg.eval_size)]

                    for s, p, ot in tqdm.tqdm(testsub):
                        s , p, ot = s.item(), p.item(), ot.item()

                        raw_candidates = [(s, p, o) for o in range(len(i2n))]
                        candidates = filter(raw_candidates, alltriples, (s, p, ot))
                        candidates = torch.tensor(candidates)

                        # if len(raw_candidates) != len(candidates):
                        #     print(f'filtered out {len(raw_candidates) - len(candidates)} candidates.')

                        scores = util.batch(model, candidates, batch_size=arg.batch * 2 * (1 + ng))

                        sorted_candidates = [tuple(p[0]) for p in sorted(zip(candidates.tolist(), scores.tolist()), key=lambda p : -p[1])]

                        rank = (sorted_candidates.index((s, p, ot)) + 1)

                        hitsat1 += (rank == 1)
                        hitsat3 += (rank <= 3)
                        hitsat10 += (rank <= 10)
                        mrr += 1.0 / rank

                    mrr = mrr / len(test)
                    hitsat1 = hitsat1 / len(test)
                    hitsat3 = hitsat3 / len(test)
                    hitsat10 = hitsat10 / len(test)

                    print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hitsat1:.4}\t  hits@3 {hitsat3:.4}\t  hits@10 {hitsat10:.4}')


    print('training finished.')

    tracc, teacc = torch.tensor(train_accs), torch.tensor(test_accs)
    print(f'mean training accuracy {tracc.mean():.3} ({tracc.std():.3})  \t{train_accs}')
    print(f'mean test accuracy     {teacc.mean():.3} ({teacc.std():.3})  \t{test_accs}')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Size (nr of dimensions) of the input.",
                        default=150, type=int)

    parser.add_argument("--eval-size",
                        dest="eval_size",
                        help="Subsample size of the test set for intermediate evaluations.",
                        default=None, type=int)

    parser.add_argument("--eval-int",
                        dest="eval_int",
                        help="Nr. of epochs between intermediate evaluations",
                        default=10, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Nr of layers.",
                        default=2, type=int)

    parser.add_argument("-B", "--batch-size",
                        dest="batch",
                        help="Nr of positive triples to consider per batch (negatives are added to this).",
                        default=32, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="emb",
                        help="Size (nr of dimensions) of the hidden layer.",
                        default=16, type=int)

    parser.add_argument("--embedding-init",
                        dest="emb1",
                        help="Size (nr of dimensions) of the _initial_ node embeddings (applies to emb model only).",
                        default=128, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("--loss",
                        dest="loss",
                        help="Loss function (logistic, hinge)",
                        default='hinge', type=str)

    parser.add_argument("--pairwise",
                        dest="loss",
                        help="Apply loss pairwise (if false, loss is applied pointwise)",
                        action='store_true')

    parser.add_argument("-N", "--negative-rate",
                        dest="negative_rate",
                        help="Number of negatives for every positive",
                        default=1, type=int)

    parser.add_argument("--weight-decay",
                        dest="wd",
                        help="Weight decay (using AdamW implementation).",
                        default=0.0, type=float)

    parser.add_argument("--l2-weight", dest="l2weight",
                        help="Weight for explicit L2 loss term.",
                        default=None, type=float)

    parser.add_argument("--do",
                        dest="do",
                        help="Dropout",
                        default=None, type=float)

    parser.add_argument("-D", "--dataset-name",
                        dest="name",
                        help="Name of dataset to use [fb, wn]",
                        default='fb', type=str)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="which model to use",
                        default='classic', type=str)

    parser.add_argument("--indep",
                        dest="indep",
                        help="Learn independent attention weights for each edge instead of ones derived from node embeddings).",
                        action="store_true")

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit the number of relations.",
                        default=None, type=int)

    parser.add_argument("--decomp",
                        dest="decomp",
                        help="decomposition method (basis, block).",
                        default=None, type=str)

    parser.add_argument("--num-bases",
                        dest="num_bases",
                        help="Number of bases (for the basis decomposition).",
                        default=None, type=int)

    parser.add_argument("--num-blocks",
                        dest="num_blocks",
                        help="Number of blocks (for the block diagonal decomposition).",
                        default=None, type=int)

    parser.add_argument("--repeats",
                        dest="repeats",
                        help="Number of times to repeat the experiment.",
                        default=1, type=int)

    parser.add_argument("--unify",
                        dest="unify",
                        help="Method for unifying the relations.",
                        default='sum', type=str)

    parser.add_argument("--opt",
                        dest="opt",
                        help="Optimizer.",
                        default='adamw', type=str)

    parser.add_argument("--conditional", dest="cond",
                        help="Condition on the target node.",
                        action="store_true")

    parser.add_argument("--dropin", dest="dropin",
                        help="Randomly mask out connections by atte tion weight.",
                        action="store_true")

    parser.add_argument("--separate-embeddings", dest="sep_emb",
                        help="Separate embeddings per relation (expensive, but closer to original RGCN).",
                        action="store_true")

    parser.add_argument("--no-res", dest="nores",
                        help="Disable residual connections.",
                        action="store_true")

    parser.add_argument("--no-norm", dest="nonorm",
                        help="Disable batch norm.",
                        action="store_true")

    parser.add_argument("--prune", dest="prune",
                        help="Prune the graph to remove irrelevant links.",
                        action="store_true")

    parser.add_argument("--sample", dest="sample",
                        help="Subsample the graph according to the weights.",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
