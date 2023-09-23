import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
parser.add_argument('--student', type=str, default='PLP', help='Student Model')
parser.add_argument('--distill', action='store_false', default=True, help='Distill or not')
parser.add_argument('--device', type=int, default=3, help='CUDA Device')
parser.add_argument('--ptype', type=str, default='ind', help='plp type: ind(inductive); tra(transductive/onehot)')
parser.add_argument('--labelrate', type=int, default=20, help='label rate')
parser.add_argument('--mlp_layers', type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')
parser.add_argument('--grad', type=int, default=1, help='output grad or not')

parser.add_argument('--automl', action='store_true', default=False, help='Automl or not')
parser.add_argument('--ntrials', type=int, default=10, help='Number of trials')
parser.add_argument('--njobs', type=int, default=10, help='Number of jobs')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

