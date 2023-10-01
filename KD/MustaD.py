import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import copy
import time
import itertools
from pathlib import Path
from models.model_KD import *
from models.model_KD_student import *
from models.model_utils import *

from data.get_dataset import get_experiment_config
from data.utils import load_tensor_data, load_ogb_data, check_writable, initialize_label
from utils.logger import output_results, get_logger
from collections import defaultdict, namedtuple
from utils.metrics import accuracy

from mask import *
from models.selector import *

def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--lbd_pred', type=float, default=0, help='lambda for prediction loss')
    parser.add_argument('--lbd_embd', type=float, default=0, help='lambda for embedding loss')
    parser.add_argument('--distill', action='store_false', default=True, help='Distill or not')
    parser.add_argument('--device', type=int, default=3, help='CUDA Device')
    parser.add_argument('--ptype', type=str, default='ind', help='plp type: ind(inductive); tra(transductive/onehot)')
    parser.add_argument('--labelrate', type=int, default=20, help='label rate')
    parser.add_argument('--mlp_layers', type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')
    parser.add_argument('--grad', type=int, default=1, help='output grad or not')

    parser.add_argument('--automl', action='store_true', default=False, help='Automl or not')
    parser.add_argument('--ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--njobs', type=int, default=10, help='Number of jobs')
    return parser.parse_args()

def choose_model(conf):
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            t_hidden=teacher_conf['hidden'],
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
                    t_hidden=teacher_conf['hidden'],
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
                          t_hidden=teacher_conf['hidden'],
                          n_classes=labels.max().item() + 1,
                          n_layers=conf['layer'],
                          activation=F.relu,
                          dropout=conf['dropout'],
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'citeseer':
            conf['layer'] = 32
            conf['hidden'] = 128
            teacher_conf['hidden'] = 256
            conf['lamda'] = 0.6
            conf['dropout'] = 0.7
        elif conf['dataset'] == 'pubmed':
            conf['hidden'] = 128
            teacher_conf['hidden'] = 256
            conf['lamda'] = 0.4
            conf['dropout'] = 0.5
        model = GCNII_student(nfeat=features.shape[1],
                      nlayers=conf['layer'],
                      nhidden=conf['hidden'],
                      t_hidden=teacher_conf['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=conf['dropout'],
                      lamda=conf['lamda'],
                      alpha=conf['alpha'],
                      variant=False).to(conf['device'])
    else:
        raise ValueError(f'Undefined Model.')
    return model

def selector_model(conf):
    if conf['model_name'] in ['GCN', 'GCNII', 'GAT']:
        hidden_embedding = 64
    else:
        hidden_embedding = 128
    selector_model = MLP(num_layers=3,
                         input_dim=hidden_embedding,
                         hidden_dim=hidden_embedding, 
                         output_dim=hidden_embedding,
                         dropout=0.5)
    return selector_model

def train():
    """
    Start training with a stored hyperparameters on the dataset
    :make sure teacher, student, optimizer, node features, adjacency, train index is defined aforehead
    :return: train loss, train accuracy
    """
    teacher.eval()
    model.train()
    optimizer.zero_grad()

    # loss_CE
    if conf['model_name'] == 'GCN':
        t_output, t_hidden = teacher(G.ndata['feat'])
        s_output, s_hidden = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        t_output, t_hidden = teacher(G.ndata['feat'])[0:1]
        s_output, s_hidden = model(G.ndata['feat'])[0:1]
    elif conf['model_name'] == 'GraphSAGE':
        t_output, t_hidden = teacher(G, G.ndata['feat'])
        s_output, s_hidden = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'GCNII':
        t_output, t_hidden = teacher(G.ndata['feat'])[0:1]
        s_output, s_hidden = model(G.ndata['feat'])[0:1]

    s_out = F.log_softmax(s_output, dim=1)
    loss_CE = F.nll_loss(s_out[idx_train], labels[idx_train].to(device))
    acc_train = accuracy(s_out[idx_train], labels[idx_train].to(device))

    # loss_task
    t_output = t_output/temperature
    t_y = t_output[idx_train]
    s_y = s_output[idx_train]
    loss_task = kl_kernel(t_y, s_y)

    # loss_hidden
    t_x = t_hidden[idx_train]
    s_x = s_hidden[idx_train]
    loss_hidden = kl_kernel(t_x, s_x)

    # loss_final
    loss_train = loss_CE + args.lbd_pred*loss_task + args.lbd_embd*loss_hidden
    loss_train.backward()
    optimizer.step()

    return loss_train.item(),acc_train.item()

def kl_kernel(t_x, s_x):
    kl_loss_op = torch.nn.KLDivLoss(reduction='none')
    t_x = F.softmax(t_x, dim=1)
    s_x = F.log_softmax(s_x, dim=1)
    return torch.mean(torch.sum(kl_loss_op(s_x, t_x), dim=1))

def validate():
    """
    Validate the model
    make sure teacher, student, optimizer, node features, adjacency, validation index is defined aforehead
    :return: validation loss, validation accuracy
    """
    teacher.eval()
    model.eval()

    with torch.no_grad():
        # loss_CE
        if conf['model_name'] == 'GCN':
            t_output, t_hidden = teacher(G.ndata['feat'])
            s_output, s_hidden = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            t_output, t_hidden = teacher(G.ndata['feat'])[0:1]
            s_output, s_hidden = model(G.ndata['feat'])[0:1]
        elif conf['model_name'] == 'GraphSAGE':
            t_output, t_hidden = teacher(G, G.ndata['feat'])
            s_output, s_hidden = model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            t_output, t_hidden = teacher(G.ndata['feat'])[0:1]
            s_output, s_hidden = model(G.ndata['feat'])[0:1]

        s_out = F.log_softmax(s_output, dim=1)
        loss_CE = F.nll_loss(s_out[idx_val], labels[idx_val].to(device))

        # loss_task
        t_y = t_output[idx_val]
        s_y = s_output[idx_val]
        loss_task = kl_kernel(t_y, s_y)

        # loss_hidden
        t_x = t_hidden[idx_val]
        s_x = s_hidden[idx_val]
        loss_hidden = kl_kernel(t_x, s_x)

        # loss_final
        loss_val = loss_CE + args.lbd_pred*loss_task + args.lbd_embd*loss_hidden
        acc_val = accuracy(s_output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    """
    Test the model
    make sure student, node features, adjacency, test index is defined aforehead
    :return: test accuracy
    """
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        if conf['model_name'] == 'GCN':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            output, _ = model(G.ndata['feat'])[0:1]
        elif conf['model_name'] == 'GraphSAGE':
            output, _= model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            output, _ = model(G.ndata['feat'])
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return acc_test.item()

if __name__ == '__main__':
    temperature = 2
    args = arg_parse(argparse.ArgumentParser())
    
    # teacher model-specific configuration
    teacher_config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    teacher_conf = get_training_config(teacher_config_path, model_name=args.teacher)
    t_PATH = "./teacher/teacher_"+str(args.teacher)+str(args.data)+".pth"
    teacher = choose_model(teacher_conf)
    
    # student model-specific configuration
    config_path = Path.cwd().joinpath('models', 'mustad.conf.yaml')
    conf = get_training_config(config_path, model_name=args.student)
    checkpt_file = "./KD_student/student_"+str(args.student)+str(args.data)+str(args.layer)+".pth"
    model = choose_model(conf)
    
    # dataset-specific configuration
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
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
    labels_init = initialize_label(idx_train, labels_one_hot).to(conf['device'])
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    
    cudaid = "cuda:"+str(args.dev)
    device = torch.device(cudaid)
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])

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
    
    acc = test()
    
    print('The number of parameters in the student: {:04d}'.format(count_params(model)))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.2f}".format(acc*100))
