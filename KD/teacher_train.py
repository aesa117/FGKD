import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import dgl
from models.model_KD import *
from models.model_utils import *

from data.utils import load_tensor_data
from data.get_dataset import get_experiment_config

from utils.metrics import accuracy
from sklearn.metrics import f1_score

# adapted from MustaD & CPF

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora', help='dateset')
    parser.add_argument('--teacher', default='GCNII', help='model type')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    return parser.parse_args()

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
        num_heads = conf['num_heads']
        num_layers = conf['num_layers']
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=conf['hidden'],
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=conf['dropout'],
                    attn_drop=conf['att_dropout'],
                    negative_slope=conf['alpha'],     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=conf['num_layers'],
                          activation=F.relu,
                          dropout=conf['dropout'],
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'citeseer':
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
    else:
        raise ValueError(f'Undefined Model.')
    return model

def train():
    model.train()
    optimizer.zero_grad()
    if conf['model_name'] == 'GCN':
        output, _ = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        output, _ = model(G.ndata['feat'])[0:2]
    elif conf['model_name'] == 'GraphSAGE':
        output, _ = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'GCNII':
        output, _ = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    
    output = F.log_softmax(output, dim=1)
    
    acc_train = accuracy(output[idx_train], labels[idx_train].to(conf['device']))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(conf['device']))
    loss_train.backward()
    optimizer.step()
    
    return loss_train.item(), acc_train.item()

def validate():
    model.eval()

    with torch.no_grad():
        # loss_CE
        if conf['model_name'] == 'GCN':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            output, _ = model(G.ndata['feat'])[0:2]
        elif conf['model_name'] == 'GraphSAGE':
            output, _ = model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            output, _ = model(features, adj)
        else:
            raise ValueError(f'Undefined Model')

        output = F.log_softmax(output, dim=1)
        
        acc_val = accuracy(output[idx_val], labels[idx_val].to(conf['device']))
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(conf['device']))

    return loss_val.item(), acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        if conf['model_name'] == 'GCN':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            output, _ = model(G.ndata['feat'])[0:2]
        elif conf['model_name'] == 'GraphSAGE':
            output, _ = model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            output, _ = model(features, adj)
        else:
            raise ValueError(f'Undefined Model')
        
        output = F.log_softmax(output, dim=1)
        
        acc_test = accuracy(output[idx_test], labels[idx_test].to(conf['device']))
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(conf['device']))
        
    return loss_test.item(), acc_test.item()

if __name__ == '__main__':
    # argument parse
    args = arg_parse(argparse.ArgumentParser())
    
    # model-specific configuration
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    conf = get_training_config(config_path, model_name=args.teacher)
    
    # dataset-specific configuration
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']

    conf['device'] = torch.device("cuda:" + str(args.device))
    
    # print configuration dict
    conf = dict(conf, **args.__dict__)
    print(conf)
    
    # check point file path
    checkpt_file = "./teacher/Teacher_"+str(conf['model_name'])+"dataset_"+str(conf['dataset'])+".pth"
    
    # tensorboard name
    board_name = "Teacher_"+str(conf['model_name'])+"dataset_"+str(conf['dataset'])
    writer = SummaryWriter("./Log/Log_teacher/"+board_name)
    
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load data
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['dataset'], conf['device'], config_data_path)
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])
    
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

    
    start = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(500):
        loss_train, acc_train = train()
        loss_val, acc_val = validate()
        if (epoch + 1) % 10 == 0:
            print('Epoch:{:04d}'.format(epoch+1),'train:','loss:{:.3f}'.format(loss_train), 'acc:{:.2f}'.format(acc_train*100),
            '| val','loss:{:.3f}'.format(loss_val), 'acc:{:.2f}'.format(acc_val*100))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 200: # modify patience 200 -> 50
            break
        
        # write
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Acc/train', acc_train, epoch)
        
        writer.add_scalar('Loss/val', loss_val, epoch)
        writer.add_scalar('Acc/val', acc_val, epoch)
    writer.close()
    
    end = time.time()
    result_time = str(datetime.timedelta(seconds=(end-start))).split(".")
    
    loss_test, acc_test = test()
    
    print('The number of parameters in the teacher: {:04d}'.format(count_params(model)))
    print('Load {}th epoch'.format(best_epoch))
    print('Test loss:{:.2f}'.format(loss_test), 'acc:{:.2f}'.format(acc_test*100))
    print('Training Time: ', result_time[0])