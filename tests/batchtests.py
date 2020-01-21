import _context

from kgmodels import Batch, convert

import unittest
import torch
from torch import nn
#import sparse.layers.densities

import layers

def edges():
    """
    Small graph for testing

    :return:
    """

    return {
        0: ([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]),
        1: (list(range(5)), list(range(5)))
    }

def data():
    return convert(edges(), 5)

class TestBatch(unittest.TestCase):

    def test_init(self):

        embeddings = nn.Parameter(torch.rand(5, 16))
        entities = [0, 2, 4]
        batch = Batch(entities, data(), embeddings)

    def test_size(self):

        embeddings = nn.Parameter(torch.rand(5, 16))
        entities = [0, 2, 4]
        batch = Batch(entities, data(), embeddings)

        self.assertEqual(batch.size(), 3)

    def test_gen_inc_edges(self):

        embeddings = nn.Parameter(torch.rand(5, 16))
        entities = [0, 2, 4]
        batch = Batch(entities, data(), embeddings)

        returned = set()
        expected = set([(0, 0, 1), (0, 1, 0), (4, 0, 0)])
        num = 0
        for edge in batch.gen_inc_edges(0):
            returned.add(edge)
            num += 1

        self.assertEqual(num, 3)
        self.assertEqual(returned, expected)

        returned = set()
        expected = set([(2, 0, 3), (2, 1, 2), (1, 0, 2)])
        num = 0
        for edge in batch.gen_inc_edges(1): # note index of the entity, not entity itself
            returned.add(edge)
            num += 1

        self.assertEqual(num, 3)
        self.assertEqual(returned, expected)

    def test_add_edges(self):

        embeddings = nn.Parameter(torch.rand(5, 16))
        entities = [0, 2, 4]
        batch = Batch(entities, data(), embeddings)

        self.assertEquals(batch.node_emb.size(0), len(batch.toind))

        batch.add_edges(list(batch.gen_inc_edges(0)), 0)

        expected = set([(1, 1, 1), (1, 0, 2), (4, 1, 4), (3, 0, 4)])
        returned = set()
        for edge in batch.gen_inc_edges(0):
            returned.add(edge)

        self.assertEqual(returned, expected)

        self.assertEquals(batch.node_emb.size(0), len(batch.toind))

    def test_add_edges2(self):

        embeddings = nn.Parameter(torch.rand(5, 16))
        entities = [0, 2, 4]
        batch = Batch(entities, data(), embeddings)

        self.assertEquals(batch.node_emb.size(0), len(batch.toind))

        batch.add_edges(list(batch.gen_inc_edges(1)), 1)

        expected = set([(3, 1, 3), (3, 0, 4), (1, 1, 1), (0, 0, 1)])
        returned = set()
        for edge in batch.gen_inc_edges(1):
            returned.add(edge)

        self.assertEqual(returned, expected)

        self.assertEquals(batch.node_emb.size(0), len(batch.toind))

    def test_cflat(self):

        embeddings = nn.Parameter(torch.rand(5, 16))
        entities = [0, 2, 4]
        batch = Batch(entities, data(), embeddings)

        batch.add_edges(list(batch.gen_inc_edges(0)), 0)


        cfnodes = set()
        for s, p, o in batch.cflat():
            cfnodes.add(s.item())
            cfnodes.add(o.item())

        self.assertEquals(len(cfnodes), 3) # unifque nodes in edges
        self.assertEquals(batch.node_emb.size(0), 5) # 3 original entity nodes, and two nodes added to instance 0 of the batch
        # -- These may overlap in the graph, but they are kept distinct in the batch
