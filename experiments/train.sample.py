from _context import kgmodels

from kgmodels import util
from util import d

import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


import rdflib as rdf
import pandas as pd
import numpy as np

import random, sys, tqdm
from tqdm import trange

import rgat

from argparse import ArgumentParser

EPSILON = 0.000000001


global repeats

def prt(str, end='\n'):
    if repeats == 1:
        print(str, end=end)

def go(arg):

    global repeats
    repeats = arg.repeats

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_accs = []
    test_accs = []

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):

        if arg.name == 'random':

            edges, N, (train_idx, train_lbl), (test_idx, test_lbl) = \
                kgmodels.random_graph(base=arg.base, bgp0=arg.bgp0, bgp1=arg.bgp1)
            num_cls = 2

            print(f'Generated random graph with {N} nodes.')

        if arg.name == 'fan':

            edges, N, (train_idx, train_lbl), (test_idx, test_lbl) = \
                kgmodels.fan(depth=arg.rdepth, diffusion=arg.fdiff, others=1000)
            num_cls = 2

        else:
            edges, (n2i, i2n), (r2i, i2r), train, test = \
                kgmodels.load(arg.name, final=arg.final, limit=arg.limit, bidir=True)

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

        """
        Define model
        """

        train_idx, test_idx = [n.item() for n in train_idx], [n.item() for n in test_idx]

        model = kgmodels.SamplingClassifier(graph=edges, n=N, depth=arg.depth, emb=arg.emb, max_edges=arg.max_edges,
                num_cls=num_cls, boost=arg.boost, bases=arg.bases, maskid=arg.maskid, dropout=arg.do, forward_mp=arg.forward_mp,
                csample=arg.csample, incdo=arg.incdo)

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()
            train_lbl = train_lbl.cuda() # move to train loop if memory becomes tight
            test_lbl  = test_lbl.cuda()

        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.wd)

        seen = 0

        for e in range(arg.epochs):

            if arg.inc_anneal is not None:
                if arg.inc_anneal == 'linear':
                    model.set('incdo', arg.incdo * (1.0 - (e/arg.epochs)))
                else:
                    raise Exception(f'{arg.inc_anneal}')

            model.train(True)

            model.precompute_globals()

            correct = 0
            for fr in trange(0, len(train_idx), arg.batch):
                to = min(len(train_idx), fr + arg.batch)

                inputs = train_idx[fr:to] # list, not a tensor
                labels = train_lbl[fr:to]

                opt.zero_grad()

                cls = model(inputs)

                correct += (cls.argmax(dim=1) == labels).sum().item()
                loss = F.cross_entropy(cls, labels)
                loss.backward()

                opt.step()

                tbw.add_scalar('rgat/train-loss', float(loss.item()), seen)
                seen+=1

            # Evaluate
            if e % arg.eval == 0:

                prt(f'epoch {e},  loss {loss.item():.2}', end='')
                prt(f',    train cumulative {float(correct / len(train_idx)):.2} ({correct}/{len(train_idx)})', end='')
                tbw.add_scalar('rgat/train-cumulative', float(correct / len(train_idx)), e)

                with torch.no_grad():

                    if arg.full_eval:
                        model.train(False)

                    correct = 0
                    for fr in range(0, len(train_idx), arg.batch*2):
                        to = min(len(train_idx), fr + arg.batch*2)

                        inputs = train_idx[fr:to]
                        labels = train_lbl[fr:to]

                        cls = model(inputs).argmax(dim=1)
                        correct += int((labels == cls).sum())

                    trn_accuracy = correct/len(train_idx)
                    prt(f',    train accuracy {float(trn_accuracy):.2} ({correct}/{len(train_idx)})', end='')
                    tbw.add_scalar('transformer/train-accuracy', float(trn_accuracy), e)

                    correct = 0
                    for fr in range(0, len(test_idx), arg.batch*2):
                        to = min(len(test_idx), fr + arg.batch*2)

                        inputs = test_idx[fr:to]
                        labels = test_lbl[fr:to]

                        cls = model(inputs).argmax(dim=1)
                        correct += int((labels == cls).sum())

                    tst_accuracy = correct / len(test_idx)
                    prt(f',   test accuracy {float(tst_accuracy):.2} ({correct}/{len(test_idx)})')
                    tbw.add_scalar('rgat/test-accuracy', float(tst_accuracy), e)


            if torch.cuda.is_available():
                del loss # clear memory
                torch.cuda.empty_cache()

        train_accs.append(float(trn_accuracy))
        test_accs.append(float(tst_accuracy))

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

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Size of training batches.",
                        default=16, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="NR of hops into the network.",
                        default=2, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="emb",
                        help="Size (nr of dimensions) of the node embeddings.",
                        default=16, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("--weight-decay",
                        dest="wd",
                        help="Weight decay (using AdamW implementation).",
                        default=0.0, type=float)

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

    parser.add_argument("-N", "--normalize",
                        dest="normalize",
                        help="Normalize the embeddings.",
                        action="store_true")

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

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
                        help="Base network ofr the random graph experiment.",
                        default='aifb', type=str)

    parser.add_argument("--random-depth",
                        dest="rdepth",
                        help="Depth of random graph (if applicable).",
                        default=2, type=int)

    parser.add_argument("--fan-diffusion",
                        dest="fdiff",
                        help="Amount of diffusion in the fan graph.",
                        default=5, type=int)

    parser.add_argument("--max-edges",
                        dest="max_edges",
                        help="Maximum number of edges in sampled graph.",
                        default=250, type=int)

    parser.add_argument("--boost",
                        dest="boost",
                        help="Num added to the global attention scores, before they go into the sigmoid for sampling. Higher boost causes more incident edges to be sampled and fewer deep ones.",
                        default=0, type=float)

    parser.add_argument("--eval",
                        dest="eval",
                        help="Number of epochs between evaluations (if no repeats).",
                        default=5, type=int)

    parser.add_argument("--incdo",
                        dest="incdo",
                        help="Dropout probability on the incident edges. Higher dropouts miss more edges, but speed up processing",
                        default=None, type=float)

    parser.add_argument("--inc-anneal",
                        dest="inc_anneal",
                        help="Annealing schedule for the incdo parameter (linear)",
                        default=None, type=str)

    parser.add_argument("--nm",
                        dest="norm_method",
                        help="Method for row-normalizing the GAT attention weights.",
                        default='abs', type=str)

    parser.add_argument("--unify",
                        dest="unify",
                        help="Method for unifying the relations.",
                        default='sum', type=str)

    parser.add_argument("--conditional", dest="cond",
                        help="Condition on the target node.",
                        action="store_true")

    parser.add_argument("--dropin", dest="dropin",
                        help="Randomly mask out connections by attention weight.",
                        action="store_true")

    parser.add_argument("--mask-id", dest="maskid",
                        help="Mask out the embedding of the node being classified.",
                        action="store_true")

    parser.add_argument("--forward-mp", dest="forward_mp",
                        help="Perform message passing between sampling steps.",
                        action="store_true")

    parser.add_argument("--csample",
                        dest="csample",
                        help="Sub-sample the candidate edges .",
                        default=None, type=int)

    parser.add_argument("--full-eval", dest="full_eval",
                        help="Do a full graph sample for evaluation (may be very memory-intensive).",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

