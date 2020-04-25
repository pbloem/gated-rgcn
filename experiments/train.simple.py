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

EPSILON = 10e-7

global repeats

def prt(str, end='\n'):
    if repeats == 1:
        print(str, end=end)

def go(arg):

    global repeats
    repeats = arg.repeats

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        seed = arg.seed
    torch.manual_seed(seed)
    random.seed(seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_accs = []
    test_accs = []

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):

        # load data
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

        model = kgmodels.SimpleClassifier(graph=edges, n=N, depth=arg.depth, emb=arg.emb, h=arg.hidden, ksample=arg.ksample,
                num_cls=num_cls, boost=arg.boost, bases=arg.bases, dropout=arg.do,
                use_global_weights=not arg.ignore_globals)

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()
            train_lbl = train_lbl.cuda() # move to train loop if memory becomes tight
            test_lbl  = test_lbl.cuda()

        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.wd)

        seen = 0

        for e in range(arg.epochs):

            model.train(True)

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
                seen += arg.batch

            # Evaluate
            if e % arg.eval == 0:

                if torch.cuda.is_available():
                    print(f'\nPeak gpu memory use is {torch.cuda.max_memory_cached() / 1e9:.2} Gb')

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
                    tbw.add_scalar('rgat/train-accuracy', float(trn_accuracy), e)

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
                        default=128, type=int)

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
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

    parser.add_argument("--bases",
                        dest="bases",
                        help="Number of bases.",
                        default=None, type=int)

    parser.add_argument("--repeats",
                        dest="repeats",
                        help="Number of times to repeat the experiment.",
                        default=1, type=int)

    parser.add_argument("--ksample",
                        dest="ksample",
                        help="Number of edges to add per sampling layer.",
                        default=50, type=int)

    parser.add_argument("--boost",
                        dest="boost",
                        help="Num added to the global attention scores, before they go into the sigmoid for sampling. Higher boost causes more incident edges to be sampled and fewer deep ones.",
                        default=0, type=float)

    parser.add_argument("--eval",
                        dest="eval",
                        help="Number of epochs between evaluations (if no repeats).",
                        default=5, type=int)

    parser.add_argument("--seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=-1, type=int)

    parser.add_argument("--ignore-globals", dest="ignore_globals",
                        help="Ignore the global attention weights.",
                        action="store_true")


    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

