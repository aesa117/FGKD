import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *

import scipy.sparse as sp
import copy
from pathlib import Path

import dgl
from models.model_KD import *
from models.model_utils import *

from data.utils import load_tensor_data, check_writable
from data.get_dataset import get_experiment_config

from utils.logger import get_logger
from utils.metrics import accuracy

# adapted from MustaD & CPF

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    # add model type argument (ex. GCNII, GAT, GraphSAGE)
    parser.add_argument('--model', default='GCNII', help='model type')
    parser.add_argument('--dataset', default='cora', help='dateset')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='Label rate')
    return parser.parse_args()

def choose_path(conf):
    output_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                     'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    check_writable(output_dir)
    cascade_dir = output_dir.joinpath('cascade')
    check_writable(cascade_dir)
    return output_dir, cascade_dir

def choose_model(conf):
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=conf['num_layers'],
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] =='GAT':
        num_heads = 8
        num_layers = conf['num_layers']
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=8,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=2,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'citeseer':
            conf['layer'] = 32
            conf['hidden'] = 256
            conf['lamda'] = 0.6
            conf['dropout'] = 0.7
        elif conf['dataset'] == 'pubmed':
            conf['hidden'] = 256
            conf['lamda'] = 0.4
            conf['dropout'] = 0.5
        model = GCNII(nfeat=features.shape[1],
                      nlayers=conf['layer'],
                      nhidden=conf['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=conf['dropout'],
                      lamda=conf['lamda'],
                      alpha=conf['alpha'],
                      variant=False).to(conf['device'])
    return model

def train():
    model.train()
    optimizer.zero_grad()
    if conf['model_name'] == 'GCN':
        output, _ = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        output, _, _ = model(G.ndata['feat'])
    elif conf['model_name'] == 'GraphSAGE':
        output, _ = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'GCNII':
        output, _ = model(G.ndata['feat'])
    else:
        raise ValueError(f'Undefined Model')
    output = F.log_softmax(output, dim=1)
    
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    
    return loss_train.item(),acc_train.item()

def validate():
    """
    Validate the model
    make sure teacher, student, optimizer, node features, adjacency, validation index is defined aforehead
    :return: validation loss, validation accuracy
    """
    model.eval()

    with torch.no_grad():
        # loss_CE
        if conf['model_name'] == 'GCN':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            output, _ = model(G.ndata['feat'])[0:1]
        elif conf['model_name'] == 'GraphSAGE':
            output, _ = model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            output, _ = model(G.ndata['feat'])

        output = F.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))

        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        if conf['model_name'] == 'GCN':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GraphSAGE':
            output, _ = model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            output, _ = model(G.ndata['feat'])
        else:
            raise ValueError(f'Undefined Model')
        output = F.log_softmax(output, dim=1)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        print("Test set results: loss= {:.4f} acc_test= {:.4f}".format(
            loss_test.item(), acc_test.item()))
    return loss_test.item(),acc_test.item()

if __name__ == '__main__':
    # argument parse
    args = arg_parse(argparse.ArgumentParser())
    
    # model-specific configuration
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    conf = get_training_config(config_path, model_name=args.teacher)
    
    # dataset-specific configuration
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    
    # device
    if args.device > 0:
        conf['device'] = torch.device("cuda:" + str(args.device))
    else:
        conf['device'] = torch.device("cpu")
    
    # print configuration dict
    conf = dict(conf, **args.__dict__)
    print(conf)
    
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load data
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['model_name'], conf['dataset'], args.labelrate, conf['device'])
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    
    cudaid = "cuda:"+str(args.dev)
    device = torch.device(cudaid)
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])
    checkpt_file = "./teacher/teacher_"+str(args.data)+str(args.layer)+".pth"
    
    model = choose_model(conf)
    if conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'pubmed':
            conf['wd1'] = 0.0005
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': conf['wd1']},
            {'params': model.params2, 'weight_decay': conf['wd2']},
        ], lr=conf['learning_rate'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])

    
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(500):
        loss_train, acc_train = train()
        loss_val, acc_val = validate()
        if (epoch + 1) % 10 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_train),
            'acc:{:.2f}'.format(acc_train*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 50: # modify patience 200 -> 50
            break
    
    if args.test:
        acc = test()
    
    print('The number of parameters in the student: {:04d}'.format(count_params(model)))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.2f}".format(acc*100))