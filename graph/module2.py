from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import SGConv
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))

        self.gat_layers.append(torch.nn.Linear(num_hidden * heads[-2], num_classes))

    def forward(self, inputs, target, lamb, mixup_hidden = False):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        
        h2 = target
        for l in range(self.num_layers):
            h2 = self.gat_layers[l](self.g, h2).flatten(1)
        
        if mixup_hidden:
            h = lamb * h + (1 - lamb) * h2
        # output projection
        logits = self.gat_layers[-1](h)
        # logits = self.gat_layers[-1](self.g, h).mean(1)
        return h, logits

    def head_forward(self, h):
        logits = self.gat_layers[-1](h)
        return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        # self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)
        self.layers.append(torch.nn.Linear(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        
        logits = self.layers[-1](h)
        return h, logits

    def head_forward(self, h):
        logits = self.layers[-1](h)
        return logits

class SGC(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_hidden,
                 k=2,
                 cached=True):
        super(SGC, self).__init__()

        self.g = g
        # input layer
        self.layers = SGConv(in_feats, n_hidden, k, cached)

        self.head = torch.nn.Linear(n_hidden, n_classes)
    
    def head_forward(self, h):
        logits = self.head(h)
        return logits

    def forward(self, features):
        h = self.layers(self.g, features)
        logits = self.head(h)
        return h, logits
    