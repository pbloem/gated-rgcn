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

import random, sys, tqdm
from tqdm import trange

import rgat

from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import kgbench as kg

"""
Full batch training GAT and RGCN

"""

EPSILON = 0.000000001

global repeats

def prt(str, end='\n'):
    if repeats == 1:
        print(str, end=end)

def go(arg):

    global repeats
    repeats = arg.repeats

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_accs = []
    test_accs = []

    edges = triples = None
    train = test = None

    if arg.name == 'random':
        edges, N, (train_idx, train_lbl), (test_idx, test_lbl) = \
            kgmodels.random_graph(base=arg.base, bgp0=arg.bgp0, bgp1=arg.bgp1)
        num_cls = 2

        print(f'Generated random graph with {N} nodes.')

    if arg.name == 'fan':

        edges, N, (train_idx, train_lbl), (test_idx, test_lbl) = \
            kgmodels.fan(depth=arg.rdepth, diffusion=arg.fdiff, others=1000)
        num_cls = 2

        # print(f'Generated random graph with {N} nodes.')

        # print(list(zip(* (edges[0]))) )
        # sys.exit()
    elif arg.name in kg.names:
        data = kg.load(arg.name, torch=True, prune_dist=arg.depth)

        n2i,i2n = data.e2i, data.i2e
        r2i, i2r = data.r2i, data.i2r

        train_idx, train_lbl = data.training[:, 0], data.training[:, 1]
        test_idx, test_lbl   = data.withheld[:, 0], data.withheld[:, 1]

        triples = util.enrich(data.triples, n=len(i2n), r=len(i2r))
        # -- enrich: add inverse links and self loops.

        N = len(i2n)
        num_cls = data.num_classes

    else:
        edges, (n2i, i2n), (r2i, i2r), train, test = \
            kgmodels.load(arg.name, final=arg.final, limit=arg.limit, bidir=True, prune=arg.prune)

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

        N = len(i2n)

        num_cls = len(cls)

    print('some test set ids and labels', test_idx[:10], test_lbl[:10])

    tnodes = len(i2n)
    totaledges = sum([len(x[0]) for _, x in edges.items()]) if triples is None else triples.size(0)

    print(f'{tnodes} nodes')
    print(f'{len(i2r)} relations')
    print(f'{totaledges} edges (including self loops and inverse)')
    print(f'{(totaledges-tnodes)//2} edges (originally)')
    if train:
        print(f'{len(train.keys())} training labels')
        print(f'{len(test.keys())} test labels')

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):
        """
        Define model
        """
        if arg.mixer == 'classic':
            model = kgmodels.RGCNClassic(edges=edges, n=N, numcls=num_cls, num_rels=len(i2r)*2+1, emb=arg.emb, bases=arg.bases, softmax=arg.softmax, triples=triples)
        elif arg.mixer == 'emb':
            model = kgmodels.RGCNEmb(edges=edges, n=N, numcls=num_cls, emb=arg.emb1, h=arg.emb, bases=arg.bases, separate_emb=arg.sep_emb)
        elif arg.mixer == 'lgcn':
            model = kgmodels.LGCN(
                triples=triples if triples is not None else util.triples(edges),
                n=N, rp=arg.latents, numcls=num_cls, emb=arg.emb1,
                ldepth=arg.latent_depth, lwidth=arg.latent_width, bases=arg.bases)
        elif arg.mixer == 'weighted':
            model = kgmodels.RGCNWeighted(edges=edges, n=N, numcls=num_cls, emb=arg.emb1, h=arg.emb, bases=arg.bases,
                                     separate_emb=arg.sep_emb, indep=arg.indep, sample=arg.sample)
        else:
            model = kgmodels.NodeClassifier(edges=edges, n=N, depth=arg.depth, emb=arg.emb, mixer=arg.mixer, numcls=num_cls,
                                        dropout=arg.do, bases=arg.bases, norm_method=arg.norm_method, heads=arg.heads,
                                        unify=arg.unify, dropin=arg.dropin, sep_emb=arg.sep_emb, res=not arg.nores,
                                        norm=not arg.nonorm, ff=not arg.noff)

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()
            train_lbl = train_lbl.cuda()
            test_lbl  = test_lbl.cuda()

        if arg.opt == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.wd)
        elif arg.opt == 'adamw':
            opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.wd)
        else:
            raise Exception

        plt.figure()

        for e in range(arg.epochs):

            model.train(True)

            opt.zero_grad()

            cls = model()[train_idx, :]
            loss = F.cross_entropy(cls, train_lbl)

            if arg.l2weight is not None:
                loss = loss + arg.l2weight * model.penalty(p=2)

            if arg.l1 > 0.0:
                loss = loss + arg.l1 * model.edgeweights().norm(p=1)

            loss.backward()

            opt.step()

            prt(f'epoch {e},  loss {loss.item():.2}', end='')

            # Evaluate
            with torch.no_grad():

                model.train(False)

                cls = model()[train_idx, :]
                agreement = cls.argmax(dim=1) == train_lbl
                accuracy = float(agreement.sum()) / agreement.size(0)

                prt(f',    train accuracy {float(accuracy):.2}', end='')
                if e == arg.epochs - 1:
                    train_accs.append(float(accuracy))

                cls = model()[test_idx, :]
                agreement = cls.argmax(dim=1) == test_lbl
                accuracy = float(agreement.sum()) / agreement.size(0)

                prt(f',   test accuracy {float(accuracy):.2}')
                if e == arg.epochs - 1:
                    test_accs.append(float(accuracy))

                if arg.mixer == 'weighted':
                    # plot edgeweights

                    weights = model.edgeweights().cpu().numpy()
                    plt.hist(weights, bins=100)
                    plt.yscale('log')
                    plt.savefig(f'edgeweights.{e:03}.png')
                    plt.clf()

            if torch.cuda.is_available():
                del loss # clear memory
                torch.cuda.empty_cache()

            # print(model.gblocks[0].mixer.weights.mean(-1).mean(-1))

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

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Nr of layers.",
                        default=2, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="emb",
                        help="Size (nr of dimensions) of the hidden layer.",
                        default=16, type=int)

    parser.add_argument("--embedding-init",
                        dest="emb1",
                        help="Size (nr of dimensions) of the _initial_ node embeddings.",
                        default=128, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

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
                        help="Name of dataset to use [aifb, am]",
                        default='aifb', type=str)

    parser.add_argument("-m", "--mixer",
                        dest="mixer",
                        help="Which mixing layer to use [gcn, gat]",
                        default='gcn', type=str)

    parser.add_argument("--indep",
                        dest="indep",
                        help="Learn independent attention weights for each edge instead of ones derived from node embeddings).",
                        action="store_true")

    parser.add_argument("--normalize",
                        dest="normalize",
                        help="Normalize the weighted adjacency matrix.",
                        action="store_true")

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit the number of relations.",
                        default=None, type=int)

    parser.add_argument("--bgp0",
                        dest="bgp0",
                        help="BGP for class 0.",
                        default='[(0, 1, 1)]', type=str)

    parser.add_argument("--bgp1",
                        dest="bgp1",
                        help="BGP for class 1.",
                        default='[(0, 2, 2)]', type=str)

    parser.add_argument("--heads",
                        dest="heads",
                        help="Number of attention heads per relation.",
                        default=4, type=int)

    parser.add_argument("--bases",
                        dest="bases",
                        help="Number of bases.",
                        default=None, type=int)

    parser.add_argument("--repeats",
                        dest="repeats",
                        help="Number of times to repeat the experiment.",
                        default=1, type=int)

    parser.add_argument("--random-base",
                        dest="base",
                        help="Base network for the random graph experiment.",
                        default='aifb', type=str)

    parser.add_argument("--random-depth",
                        dest="rdepth",
                        help="Depth of random graph (if applicable).",
                        default=2, type=int)

    parser.add_argument("--fan-diffusion",
                        dest="fdiff",
                        help="Amount of diffusion in the fan graph.",
                        default=5, type=int)

    parser.add_argument("--latents",
                        dest="latents",
                        help="Number of latent relations in the LGCN.",
                        default=None, type=int)

    parser.add_argument("--latent-depth",
                        dest="latent_depth",
                        help="Number of hidden layers in the latent MLP.",
                        default=0, type=int)

    parser.add_argument("--latent-width",
                        dest="latent_width",
                        help="Number of hidden units in the hidden layers of the  MLP.",
                        default=128, type=int)


    parser.add_argument("--l1",
                        dest="l1",
                        help="L1 loss term for the weights.",
                        default=0.0, type=float)

    parser.add_argument("--nm",
                        dest="norm_method",
                        help="Method for row-normalizing the GAT attention weights.",
                        default='abs', type=str)

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

    parser.add_argument("--no-ff", dest="noff",
                        help="Disable local feed-forward (activation only).",
                        action="store_true")

    parser.add_argument("--prune", dest="prune",
                        help="Prune the graph to remove irrelevant links.",
                        action="store_true")

    parser.add_argument("--apply-softmax", dest="softmax",
                        help="Apply the softmax (apparently twice).",
                        action="store_true")

    parser.add_argument("--sample", dest="sample",
                        help="Subsample the graph according to the weights.",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

