import torch, os, sys

from torch import nn

import layers

class NodeClassifier(nn.Module):

    def __init__(self, edges, n, numcls, mixer='gcn', emb=16, depth=2, **kwargs):

        super().__init__()

        self.nodes = nn.Parameter(torch.randn(n, emb))

        gblocks = [GraphBlock(edges, n, mixer, emb, **kwargs) for _ in range(depth)]
        self.gblocks = nn.Sequential(*gblocks)

        self.cls = nn.Linear(emb, numcls)

    def forward(self):
        """
        Produces logits over classes for all nodes.
        :param x:
        :return:
        """

        x = self.gblocks(self.nodes)
        return self.cls(x)

class GraphBlock(nn.Module):
    """
    Combines a mixing layer (GCN or GAT) with a local feedforward, residual connection and batchnorm.

    """

    def __init__(self, edges, n, mixer='gcn', emb=16, mult=4, dropout=None, **kwargs):

        super().__init__()

        if mixer == 'gcn':
            self.mixer = layers.GCN(edges, n, emb=emb, **kwargs)
        elif mixer == 'gat':
            self.mixer = layers.GAT(edges, n, emb=emb, **kwargs)
        else:
            raise Exception(f'Mixer {mixer} not recognized')

        self.bn1 = nn.BatchNorm1d(emb)
        self.bn2 = nn.BatchNorm1d(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, mult*emb), nn.ReLU(),
            nn.Linear(emb*mult, emb, bias=False)
        )

        self.do = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: N by E matrix of node embeddings.

        :return:
        """

        x = self.bn1(self.mixer(x) + x)

        x = self.bn2(self.ff(x) + x)

        if self.do:
            x = self.do(x)

        return x