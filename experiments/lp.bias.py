from _context import kgmodels

from kgmodels import util
from util import d, tic, toc, get_slug

import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import random, sys, tqdm, math, random, os
from tqdm import trange

import rgat

from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

"""
Experiment to see if bias terms help link prediction
"""

EPSILON = 0.000000001

global repeats

def corrupt(batch, n):
    """
    Corrupts the negatives of a batch of triples (in place). The first copy of the triples is left uncorrupted

    :param batch_size:
    :param n: nr of nodes in the graph

    :return:
    """
    bs, ns, _ = batch.size()

    # new entities to insert
    corruptions = torch.randint(size=(bs * ns,),low=0, high=n, dtype=torch.long, device=d(batch))

    # boolean mask for entries to corrupt
    mask = torch.bernoulli(torch.empty(size=(bs, ns, 1), dtype=torch.float, device=d(batch)).fill_(0.5)).to(torch.bool)
    zeros = torch.zeros(size=(bs, ns, 1), dtype=torch.bool, device=d(batch))
    mask = torch.cat([mask, zeros, ~mask], dim=2)

    batch[mask] = corruptions

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

    tbdir = arg.tb_dir if arg.tb_dir is not None else os.path.join('./runs', get_slug(arg))[:250]
    tbw = SummaryWriter(log_dir=tbdir)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_mrrs = []

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

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):

        """
        Define model
        """
        model = kgmodels.LPShallow(
            triples=train, n=len(i2n), r=len(i2r), embedding=arg.emb, biases=arg.biases,
            edropout = arg.edo, rdropout=arg.rdo, decoder=arg.decoder)

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()

        if arg.opt == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=arg.lr)
        elif arg.opt == 'adamw':
            opt = torch.optim.AdamW(model.parameters(), lr=arg.lr)
        elif arg.opt == 'adagrad':
            opt = torch.optim.Adagrad(model.parameters(), lr=arg.lr)
        elif arg.opt == 'sgd':
            opt = torch.optim.SGD(model.parameters(), lr=arg.lr, nesterov=True, momentum=arg.momentum)
        else:
            raise Exception()

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=arg.patience, optimizer=opt) \
            if arg.sched else None

        # nr of negatives sampled
        ng = arg.negative_rate
        weight = torch.tensor([arg.nweight, 1.0], device=d()) if arg.nweight else None

        seen = 0
        for e in range(arg.epochs):

            seeni, sumloss = 0, 0.0

            for fr in trange(0, train.size(0), arg.batch):

                tic()
                model.train(True)

                # if arg.limit is not None and seeni > arg.limit:
                #     break

                # if torch.cuda.is_available() and random.random() < 0.01:
                #     print(f'\nPeak gpu memory use is {torch.cuda.max_memory_cached() / 1e9:.2} Gb')

                to = min(train.size(0), fr + arg.batch)

                with torch.no_grad():
                    positives = train[fr:to]

                    b, _ = positives.size()

                    # sample negatives
                    if arg.corrupt_global: # global corruption (sample random true triples to corrupt)
                        indices = torch.randint(size=(b*ng,), low=0, high=train.size(0))
                        negatives = train[indices, :].view(b, ng, 3) # -- triples to be corrupted

                    else: # local corruption (directly corrupt the current batch)
                        negatives = positives.clone()[:, None, :].expand(b, ng, 3).contiguous()

                    corrupt(negatives, len(i2n))

                    triples = torch.cat([positives[:, None, :], negatives], dim=1)

                    if torch.cuda.is_available():
                        triples = triples.cuda()

                    if arg.loss == 'bce':
                        labels = torch.cat([torch.ones(b, 1), torch.zeros(b, ng)], dim=1)
                    elif arg.loss == 'ce':
                        labels = torch.zeros(b, dtype=torch.long)
                        # -- CE loss treats the problem as a multiclass classification problem: for a positive triple,
                        #    together with its k corruptions, identify which is the true triple. This is always triple 0,
                        #    but the score function is order equivariant, so i can't see the index of the triple it's
                        #    classifying.

                    if torch.cuda.is_available():
                        labels = labels.cuda()

                opt.zero_grad()

                out = model(triples)

                assert out.size() == (b, ng + 1)

                if arg.loss == 'bce':
                    loss = F.binary_cross_entropy_with_logits(out, labels, weight=weight)
                elif arg.loss == 'ce':
                    loss = F.cross_entropy(out, labels)

                if arg.reg_eweight is not None:
                    loss += model.penalty(which='entities', p=arg.reg_exp, rweight=arg.reg_eweight)

                if arg.reg_rweight is not None:
                    loss += model.penalty(which='relations', p=arg.reg_exp, rweight=arg.reg_rweight)

                loss.backward()

                sumloss += float(loss.item())

                opt.step()

                seen += b; seeni += b
                tbw.add_scalar('biases/train_loss', float(loss.item()), seen)

            # Evaluate
            if (e % arg.eval_int == 0 and e != 0) or e == arg.epochs - 1:
                with torch.no_grad():

                    model.train(False)

                    ranks = []

                    mrr = hitsat1 = hitsat3 = hitsat10 = 0.0

                    if arg.eval_size is None:
                        testsub = test
                    else:
                        testsub = test[random.sample(range(test.size(0)), k=arg.eval_size)]

                    tseen = 0
                    for tail in [True, False]: # head or tail prediction

                        for s, p, o in tqdm.tqdm(testsub):

                            s, p, o = s.item(), p.item(), o.item()

                            if tail:
                                ot = o; del o

                                raw_candidates = [(s, p, o) for o in range(len(i2n))]
                                candidates = filter(raw_candidates, alltriples, (s, p, ot))

                            else:
                                st = s; del s

                                raw_candidates = [(s, p, o) for s in range(len(i2n))]
                                candidates = filter(raw_candidates, alltriples, (st, p, o))

                            candidates = torch.tensor(candidates)
                            scores = util.batch(model, candidates, batch_size=arg.batch * 2)

                            sorted_candidates = [tuple(p[0]) for p in sorted(zip(candidates.tolist(), scores.tolist()), key=lambda p : -p[1])]

                            rank = (sorted_candidates.index((s, p, ot)) + 1) if tail else (sorted_candidates.index((st, p, o)) + 1)
                            ranks.append(rank)

                            hitsat1 += (rank == 1)
                            hitsat3 += (rank <= 3)
                            hitsat10 += (rank <= 10)
                            mrr += 1.0 / rank

                            tseen += 1

                    mrr = mrr / tseen
                    hitsat1 = hitsat1 / tseen
                    hitsat3 = hitsat3 / tseen
                    hitsat10 = hitsat10 / tseen

                    print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hitsat1:.4}\t  hits@3 {hitsat3:.4}\t  hits@10 {hitsat10:.4}')
                    print(f'   ranks : {ranks[:10]}')

                    tbw.add_scalar('biases/mrr', mrr, e)
                    tbw.add_scalar('biases/h@1', hitsat1, e)
                    tbw.add_scalar('biases/h@3', hitsat3, e)
                    tbw.add_scalar('biases/h@10', hitsat10, e)

                    if sched is not None:
                        sched.step(mrr) # reduce lr if mrr stalls

        test_mrrs.append(mrr)

    print('training finished.')

    temrrs = torch.tensor(test_mrrs)
    print(f'mean test MRR    {temrrs.mean():.3} ({temrrs.std():.3})  \t{test_mrrs}')


if __name__ == "__main__":

    mp.set_start_method('spawn')

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

    parser.add_argument("-B", "--batch-size",
                        dest="batch",
                        help="Nr of positive triples to consider per batch (negatives are added to this).",
                        default=32, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="emb",
                        help="Size (nr of dimensions) of the hidden layer.",
                        default=128, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-N", "--negative-rate",
                        dest="negative_rate",
                        help="Number of negatives for every positive",
                        default=1, type=int)

    parser.add_argument("--reg-exp",
                        dest="reg_exp",
                        help="Regularizer exponent (1, 2, 3)",
                        default=2, type=int)

    parser.add_argument("--reg-eweight",
                        dest="reg_eweight",
                        help="Regularizer weight entities",
                        default=0, type=float)

    parser.add_argument("--reg-rweight",
                        dest="reg_rweight",
                        help="Regularizer weight relations",
                        default=0, type=float)

    parser.add_argument("-D", "--dataset-name",
                        dest="name",
                        help="Name of dataset to use [fb, wn, toy]",
                        default='fb', type=str)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="which model to use",
                        default='classic', type=str)

    parser.add_argument("--dec",
                        dest="decoder",
                        help="Which decoding function to use (distmult, transe)",
                        default='distmult', type=str)

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("--repeats",
                        dest="repeats",
                        help="Number of times to repeat the experiment.",
                        default=1, type=int)

    parser.add_argument("--opt",
                        dest="opt",
                        help="Optimizer.",
                        default='adam', type=str)

    parser.add_argument("--momentum",
                        dest="momentum",
                        help="Optimizer momentum (olny for SGD).",
                        default=0.0, type=float)

    parser.add_argument("--loss",
                        dest="loss",
                        help="Which loss function to use (bce, ce).",
                        default='bce', type=str)

    parser.add_argument("--corrupt-global", dest="corrupt_global",
                        help="If not set, corrupts the current batch as negative samples. If set, samples triples globally to corrupt.",
                        action="store_true")

    parser.add_argument("--biases", dest="biases",
                        help="Learn bias parameters.",
                        action="store_true")

    parser.add_argument("--edropout",
                        dest="edo",
                        help="Entity dropout (applied just before encoder).",
                        default=None, type=float)

    parser.add_argument("--rdropout",
                        dest="rdo",
                        help="Relation dropout (applied just before encoder).",
                        default=None, type=float)

    parser.add_argument("--sched", dest="sched",
                        help="Enable scheduler.",
                        action="store_true")

    parser.add_argument("--reciprocal", dest="reciprocal",
                        help="Learn reciprocal relations.",
                        action="store_true")

    parser.add_argument("--patience",
                        dest="patience",
                        help="Plateau scheduler patience.",
                        default=1, type=float)

    parser.add_argument("--nweight",
                        dest="nweight",
                        help="Weight of negative samples (BCE loss only).",
                        default=None, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)



    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

