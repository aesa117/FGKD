from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model_MustaD import *

# adapted from MustaD

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
# model 종류 지정
parser.add_argument('--model', default='gcn', help='model type')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data - should to set data name
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
# cuda device assignment
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
# features, adj connect to device
features = features.to(device)
adj = adj.to(device)
# set checkpoint file name
checkpt_file = "./teacher/teacher_"+str(args.data)+str(args.layer)+".pth"

# Define model
if str(args.model) == 'gcn':
    model = GCNII(in_feats=features.shape[1],
                n_hidden=args.hidden, 
                n_layers=args.layer, 
                n_classes=int(labels.max()) + 1, 
                dropout=args.dropout, 
                lamda=args.lamda, 
                alpha=args.alpha, 
                variant=args.variant)


