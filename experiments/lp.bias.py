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
    Corrupts the negatives of a batch of triples (in place).

    Randomly corrupts either heads or tails

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

def corrupt_one(batch, candidates, target):
    """
    Corrupts the negatives of a batch of triples (in place).

    Corrupts either only head or only tails

    :param batch_size:
    :param n: nr of nodes in the graph
    :param target: 0 for head, 1 for predicate, 2 for tail


    :return:
    """
    bs, ns, _ = batch.size()

    # new entities to insert
    #corruptions = torch.randint(size=(bs * ns,),low=0, high=n, dtype=torch.long, device=d(batch))
    corruptions = torch.tensor(random.choices(candidates, k=bs*ns),  dtype=torch.long, device=d(batch)).view(bs, ns)

    batch[:, :, target] = corruptions

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

    train, val, test, (n2i, i2n), (r2i, i2r) = \
        kgmodels.load_lp(arg.name)

    # set of all triples (for filtering)
    alltriples = set()
    for s, p, o in torch.cat([train, val, test], dim=0):
        s, p, o = s.item(), p.item(), o.item()

        alltriples.add((s, p, o))

    truedicts = util.truedicts(alltriples)

    if arg.final:
        train, test = torch.cat([train, val], dim=0), test
    else:
        train, test = train, val

    subjects   = list({s for s, _, _ in train})
    predicates = list({p for _, p, _ in train})
    objects    = list({o for _, _, o in train})
    ccandidates = (subjects, predicates, objects)

    print(len(i2n), 'nodes')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')

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

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=arg.patience, optimizer=opt, mode='max', factor=0.95, threshold=0.0001) \
            if arg.sched else None
        #-- defaults taken from libkge

        # nr of negatives sampled
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

                    # # sample negatives
                    # if arg.corrupt_global: # global corruption (sample random true triples to corrupt)
                    #     indices = torch.randint(size=(b*ng,), low=0, high=train.size(0))
                    #     negatives = train[indices, :].view(b, ng, 3) # -- triples to be corrupted
                    #
                    # else: # local corruption (directly corrupt the current batch)
                    #     negatives = positives.clone()[:, None, :].expand(b, ng, 3).contiguous()


                    ttriples = []
                    for target, ng in zip([0, 1, 2], arg.negative_rate):
                        if ng > 0:

                            negatives = positives.clone()[:, None, :].expand(b, ng, 3).contiguous()
                            corrupt_one(negatives, ccandidates[target] if arg.limit_negatives else range(len(i2n)), target)

                            ttriples.append(torch.cat([positives[:, None, :], negatives], dim=1))

                    triples = torch.cat(ttriples, dim=0)

                    b, _, _ = triples.size()

                    if arg.loss == 'bce':
                        labels = torch.cat([torch.ones(b, 1), torch.zeros(b, ng)], dim=1)
                    elif arg.loss == 'ce':
                        labels = torch.zeros(b, dtype=torch.long)
                        # -- CE loss treats the problem as a multiclass classification problem: for a positive triple,
                        #    together with its k corruptions, identify which is the true triple. This is always triple 0.
                        #    (It may seem like the model could easily cheat by always choosing triple 0, but the score
                        #    function is order equivariant, so it can't choose by ordering.)

                    if torch.cuda.is_available():
                        triples = triples.cuda()
                        labels = labels.cuda()

                opt.zero_grad()

                out = model(triples)

                if arg.loss == 'bce':
                    loss = F.binary_cross_entropy_with_logits(out, labels, weight=weight, reduction=arg.lred)
                elif arg.loss == 'ce':
                    loss = F.cross_entropy(out, labels, reduction=arg.lred)

                if arg.reg_eweight is not None:
                    loss = loss + model.penalty(which='entities', p=arg.reg_exp, rweight=arg.reg_eweight)

                if arg.reg_rweight is not None:
                    loss = loss + model.penalty(which='relations', p=arg.reg_exp, rweight=arg.reg_rweight)

                assert not torch.isnan(loss), 'Loss has become NaN'

                loss.backward()

                sumloss += float(loss.item())

                #print('emean: ', model.relations.grad.mean().item())

                opt.step()

                seen += b; seeni += b
                tbw.add_scalar('biases/train_loss', float(loss.item()), seen)

            # Evaluate
            if ((e+1) % arg.eval_int == 0) or e == arg.epochs - 1:

                with torch.no_grad():

                    model.train(False)

                    if arg.eval_size is None:
                        testsub = test
                    else:
                        testsub = test[random.sample(range(test.size(0)), k=arg.eval_size)]

                    mrr, hits, ranks = util.eval_batch(
                        model=model, valset=testsub, truedicts=truedicts, n=len(i2n),
                        batch_size=arg.test_batch, verbose=True)

                    if arg.check_simple:
                        mrrs, hitss, rankss = util.eval_simple(
                            model=model, valset=testsub, alltriples=alltriples, n=len(i2n), verbose=True)

                        assert ranks == rankss
                        assert mrr == mrrs

                    print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hits[0]:.4}\t  hits@3 {hits[1]:.4}\t  hits@10 {hits[2]:.4}')
                    print(f'   ranks : {ranks[:10]}')

                    print('len check', len(ranks), len(testsub))

                    tbw.add_scalar('biases/mrr', mrr, e)
                    tbw.add_scalar('biases/h@1', hits[0], e)
                    tbw.add_scalar('biases/h@3', hits[1], e)
                    tbw.add_scalar('biases/h@10', hits[2], e)

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


    parser.add_argument("--test-batch",
                        dest="test_batch",
                        help="Number of triples per batch (including all candidates).",
                        default=10, type=int)

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
                        help="Number of negatives for every positive (for s, p and o respectively)",
                        nargs=3,
                        default=[10, 0, 10], type=int)

    parser.add_argument("--reg-exp",
                        dest="reg_exp",
                        help="Regularizer exponent (1, 2, 3)",
                        default=2, type=int)

    parser.add_argument("--reg-eweight",
                        dest="reg_eweight",
                        help="Regularizer weight entities",
                        default=None, type=float)

    parser.add_argument("--reg-rweight",
                        dest="reg_rweight",
                        help="Regularizer weight relations",
                        default=None, type=float)

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

    parser.add_argument("--limit-negatives", dest="limit_negatives",
                        help="Sample oly negative heads that have appeared in the head position (and likewise for tails).",
                        action="store_true")

    parser.add_argument("--reciprocal", dest="reciprocal",
                        help="Learn reciprocal relations.",
                        action="store_true")

    parser.add_argument("--check-simple", dest="check_simple",
                        help="Double check with the simple (and slow) computation of the MRR.",
                        action="store_true")

    parser.add_argument("--patience",
                        dest="patience",
                        help="Plateau scheduler patience.",
                        default=1, type=float)

    parser.add_argument("--loss-reduction",
                        dest="lred",
                        help="How to reduce the loss along the batch dimension (sum, mean).",
                        default='sum', type=str)

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

