import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.nn import SAGEConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        """
        Construct GCNII layer

        :param in_features: input dimension
        :param out_features: output dimension
        :param residual: ratio of residual connection
        :param variant: variant version introduced in GCNII
        """
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters
        
        :return: none
        """
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        """
        Compute a GCNII layer
        
        :param input: input feature
        :param adj: adjacency matrix
        :param lamda: ratio of lamda
        :param alpha: alpha
        :param l: l^{th} layer in an overall model
        """
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII_student(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, thidden, nclass, dropout, lamda, alpha, variant):
        """
        Constructor of GCNII student model

        :param nfeat: input dimension
        :param nlayers: number of layers
        :param nhidden: student's hidden feature dimension
        :param thidden: teacher's hidden feature dimension
        :param nclass: number of output class
        :param dropout: ratio of dropout
        :param dropout: ratio of lamda
        :param alpha: alpha
        :param variant: variant version introduced in GCNII
        """
        super(GCNII_student, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.tlayers = nlayers
        self.nhidden = nhidden
        self.thidden = thidden
        if self.nhidden != self.thidden:
            self.match_dim = nn.Linear(nhidden, thidden)

    def forward(self, x, adj):
        """
        Forward x into class

        :param x: input node features
        :param adj: adjacency matrix
        :return: task prediction, last hidden embedding
        """
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i in range(self.tlayers):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.convs[0](layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        hidden_emb = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](hidden_emb)
        if self.nhidden != self.thidden:
            hidden_emb = self.match_dim(hidden_emb)

        return layer_inner, hidden_emb

class GraphSAGE_student(nn.Module):
    def __init__(self, in_feats, n_hidden, t_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        super(GraphSAGE_student, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None
        
        self.t_layers = n_layers
        self.n_hidden = n_hidden
        self.t_hidden = t_hidden
        if self.n_hidden != self.t_hidden:
            self.match_dim = nn.Linear(n_hidden, t_hidden)

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        h = self.layers[0](graph, h)
        for i in range(self.t_layers):
            h = self.layers[1](graph, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        prior = h
        h = self.layers[2](graph, h)
        return h, prior

class GAT_student(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, t_hidden, num_classes, heads, activation,
                 feat_drop, attn_drop, negative_slope, residual):
        super(GAT_student, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[1], 
            feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        
        self.t_layers = num_layers
        self.n_hidden = num_hidden
        self.t_hidden = t_hidden
        if self.n_hidden != self.t_hidden:
            self.match_dim = nn.Linear(num_hidden, t_hidden)

    def forward(self, inputs):
        h = inputs
        h, att = self.gat_layers[0](self.g, h)
        for i in range(self.t_layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h, att = self.gat_layers[1](self.g, h)
            h = h.flatten(1)
        prior = h.mean(1)
        # output projection
        logits, att = self.gat_layers[-1](self.g, h)
        logits = logits.mean(1)
        return logits, prior, att

class GCN_student(nn.Module):
    def __init__(self, g, in_feats, n_hidden, t_hidden, n_classes, n_layers, activation, dropout):
        super(GCN_student, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        
        self.t_layers = n_layers
        self.n_hidden = n_hidden
        self.t_hidden = t_hidden
        if self.n_hidden != self.t_hidden:
            self.match_dim = nn.Linear(n_hidden, t_hidden)


    def forward(self, features):
        h = features
        h = self.layers[0](self.g, h)
        for i in range(self.t_layers):
            h = self.dropout(h)
            h = self.layers[1](self.g, h)
        prior = h
        h = self.dropout(h)
        h = self.layers[2](self.g, h)
        return h, prior