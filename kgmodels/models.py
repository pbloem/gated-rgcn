import torch, os, sys

from torch import nn

import layers

class NodeClassifier(nn.Module):

    def __init__(self, edges, n, numcls, mixer='gcn', emb=16, depth=2, sep_emb=False, **kwargs):

        super().__init__()

        self.se = sep_emb
        r = len(edges.keys())

        if self.se:
            self.nodes = None
        else:
            self.nodes = nn.Parameter(torch.randn(n, emb))

        gblocks = [GraphBlock(edges, n, mixer, emb, gcn_first=((i==0) if self.se else False), **kwargs) for i in range(depth)]
        self.gblocks = nn.Sequential(*gblocks)

        self.cls = nn.Linear(emb, numcls)

    def forward(self, conditional=None):
        """
        Produces logits over classes for all nodes.
        :param x:
        :return:
        """

        x = self.nodes

        if conditional is not None:
            conditional = x[conditional, :][None, :]

        for block in self.gblocks:
            x = block(x, conditional=conditional)

        return self.cls(x)

class GraphBlock(nn.Module):
    """
    Combines a mixing layer (GCN or GAT) with a local feedforward, residual connection and batchnorm.

    """

    def __init__(self, edges, n, mixer='gcn', emb=16, mult=4, dropout=None, gcn_first=False,
                 res=True, norm=True, ff=True, **kwargs):

        super().__init__()

        self.res = res
        self.norm = norm

        if mixer == 'gcn':
            if gcn_first:
                self.mixer = layers.GCNFirst(edges, n, emb=emb, **kwargs)
            else:
                self.mixer = layers.GCN(edges, n, emb=emb, **kwargs)
        elif mixer == 'gat':
            self.mixer = layers.GAT(edges, n, emb=emb, **kwargs)
        else:
            raise Exception(f'Mixer {mixer} not recognized')

        if self.norm:
            self.bn1 = nn.BatchNorm1d(emb)
            self.bn2 = nn.BatchNorm1d(emb)

        if ff:
            self.ff = nn.Sequential(
                nn.Linear(emb, mult*emb), nn.ReLU(),
                nn.Linear(emb*mult, emb, bias=False)
            )
        else:
            self.ff = nn.ReLU()

        self.do = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x, conditional=None):
        """
        :param x: N by E matrix of node embeddings.

        :return:
        """
        res = x if x is not None else 0.0
        # -- No res connection if the input has separate embeddings per relation
        #    (because the mixer output won't)

        x = self.mixer(x, conditional=conditional)

        if self.res:
            x = x + res
        if self.norm:
            x = self.bn1(x)

        res = x
        x = self.ff(x)

        if self.res:
            x = x + res
        if self.norm:
            x = self.bn2(x)

        if self.do:
            x = self.do(x)

        # x = self.bn1(self.mixer(x))

        return x