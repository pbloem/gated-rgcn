import _context

from kgmodels.util import eval

import unittest
import torch
from torch import nn

import sys


class TestUtil(unittest.TestCase):


    def test_eval(self):

        N = 500

        class Scorer(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, triples):
                scores = triples[:, 0] if triples[0, 2] == triples[1, 2] else triples[:, 2]

                return scores.to(torch.float).pow(-1.0)

        model = Scorer()

        testset = [(i, 0, i) for i in range(N)]
        testset = torch.tensor(testset, dtype=torch.long)

        mrr, hits, ranks = eval(model=model, valset=testset, alltriples=[], n=N, verbose=False)

        self.assertEqual(ranks, list(range(1, N+1)) * 2)
        self.assertEqual(mrr, sum([1/r for r in ranks])/len(ranks))
        self.assertEqual(hits[0], 2/(N * 2)) # one correct ans for head corrupt, one for tail
        self.assertEqual(hits[1], 6/(N * 2))
        self.assertEqual(hits[2], 20/(N * 2))
