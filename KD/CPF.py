import argparse
import numpy as np
import torch
import torch.optim as optim
import time
import datetime
from pathlib import Path
from models.model_KD import *
from models.PLP import *
from models.model_utils import *

from data.get_dataset import get_experiment_config
from data.utils import load_tensor_data, initialize_label
from utils.metrics import *
from utils.metrics import accuracy

from mask import *
from models.selector import *

from pytorch_metric_learning import losses, miners
from sklearn.metrics.cluster import normalized_mutual_info_score

def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--lbd_embd', type=int, default=1, help='selected embedding loss rate')
    parser.add_argument('--distill', action='store_false', default=True, help='Distill or not')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--ptype', type=str, default='ind', help='plp type: ind(inductive); tra(transductive/onehot)')
    parser.add_argument('--mlp_layers', type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')
    parser.add_argument('--grad', type=int, default=1, help='output grad or not')

    parser.add_argument('--automl', action='store_true', default=False, help='Automl or not')
    parser.add_argument('--ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--njobs', type=int, default=10, help='Number of jobs')
    return parser.parse_args()

def choose_model(conf, G, features, labels, byte_idx_train, labels_one_hot):
    if conf['model_name'] == 'PLP':
        model = PLP(g=G,
                    num_layers=conf['num_layers'],
                    in_dim=G.ndata['feat'].shape[1],
                    emb_dim=conf['emb_dim'],
                    num_classes=labels.max().item() + 1,
                    activation=F.relu,
                    feat_drop=conf['feat_drop'],
                    attn_drop=conf['attn_drop'],
                    residual=False,
                    byte_idx_train=byte_idx_train,
                    labels_one_hot=labels_one_hot,
                    ptype=conf['ptype'],
                    mlp_layers=conf['mlp_layers']).to(conf['device'])
    else:
        raise ValueError(f'Undefined Model.')
    return model

def selector_model_init(conf):
    if conf['model_name'] in ['GCN', 'GCNII', 'GAT']:
        embedding_size = 64
        hidden_embedding = 128
    else:
        embedding_size = 128
        hidden_embedding = 256
    selector_model = MLP(num_layers=conf['nlayer'],
                         input_dim=conf['feaetures'],
                         hidden_dim=hidden_embedding, 
                         output_dim=embedding_size,
                         dropout=0.5)
    
    # add new input layer after delete first layer
    selector_model.layers = selector_model.layers[1:]
    input = nn.Linear(embedding_size, hidden_embedding)
    selector_model.layers = nn.Sequential(input, *selector_model.layers)
    
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
    if teacher_conf['model_name'] == 'GCN':
        t_output, t_hidden = teacher(G.ndata['feat'])
        s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
    elif teacher_conf['model_name'] == 'GAT':
        t_output, t_hidden = teacher(G.ndata['feat'])[0:2]
        s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
    elif teacher_conf['model_name'] == 'GraphSAGE':
        t_output, t_hidden = teacher(G, G.ndata['feat'])
        s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
    elif teacher_conf['model_name'] == 'GCNII':
        t_output, t_hidden = teacher(features, adj)
        s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
    else:
        raise ValueError(f'Undefined Model')
    
    s_out = F.log_softmax(s_output, dim=1)
    t_out = F.log_softmax(t_output, dim=1)
    loss_dist = my_loss(s_out[idx_no_train], t_out[idx_no_train], 2).to(conf['device'])
    acc_train = accuracy(s_out[idx_train], labels[idx_train].to(conf['device']))

    # mask selection - training and extract masks for clustering score
    updated_masks = selection(selector_model, t_hidden, labels, normalized_mutual_info_score, selector_loss, selector_optimizer, masks, num_masks, data_size)
    best_mask = updated_masks[0]
    masked_t_hidden = best_mask * t_hidden
    
    # loss_hidden
    t_x = masked_t_hidden[idx_train]
    s_x = s_hidden[idx_train]
    loss_hidden = kl_kernel(t_x, s_x)

    # loss_final
    loss_train = loss_dist + args.lbd_embd*loss_hidden
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
        if teacher_conf['model_name'] == 'GCN':
            t_output, t_hidden = teacher(G.ndata['feat'])
            s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
        elif teacher_conf['model_name'] == 'GAT':
            t_output, t_hidden = teacher(G.ndata['feat'])[0:2]
            s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
        elif teacher_conf['model_name'] == 'GraphSAGE':
            t_output, t_hidden = teacher(G, G.ndata['feat'])
            s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
        elif teacher_conf['model_name'] == 'GCNII':
            t_output, t_hidden = teacher(features, adj)
            s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
        else:
            raise ValueError(f'Undefined Model')
        
        s_out = F.log_softmax(s_output, dim=1)
        t_out = F.log_softmax(t_output, dim=1)
        loss_dist = my_loss(s_out[idx_no_train], t_out[idx_no_train], 2).to(conf['device'])
        acc_val = accuracy(s_out[idx_train], labels[idx_train].to(conf['device']))

        # mask selection - training and extract masks for clustering score
        updated_masks = selection(selector_model, t_hidden, labels, normalized_mutual_info_score, selector_loss, selector_optimizer, masks, num_masks, data_size)
        best_mask = updated_masks[0]
        masked_t_hidden = best_mask * t_hidden
        
        # loss_hidden
        t_x = masked_t_hidden[idx_train]
        s_x = s_hidden[idx_train]
        loss_hidden = kl_kernel(t_x, s_x)

        # loss_final
        loss_val = loss_dist + args.lbd_embd*loss_hidden
        acc_val = accuracy(s_output[idx_val], labels[idx_val].to(conf['device']))
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
        output = model(G.ndata['feat'], labels_init)[0]
        acc_test = accuracy(output[idx_test], labels[idx_test].to(conf['device']))
        return acc_test.item()

if __name__ == '__main__':
    temperature = 2
    args = arg_parse(argparse.ArgumentParser())
    
    # teacher model-specific configuration
    teacher_config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    teacher_conf = get_training_config(teacher_config_path, model_name=args.teacher)
    t_PATH = "./teacher/teacher_"+str(args.teacher)+"_"+str(args.dataset)+".pth"
    
    # student model-specific configuration
    config_path = Path.cwd().joinpath('models', 'distill.conf.yaml')
    conf = get_training_config(config_path, model_name=args.student)
    checkpt_file = "./KD_student/student_"+str(args.student)+"_"+str(args.dataset)+".pth"
    
    # dataset-specific configuration
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    conf['device'] = torch.device("cuda:" + str(args.device))
    teacher_conf['device'] = torch.device("cuda:" + str(args.device))
    
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
            load_tensor_data(conf['dataset'], conf['device'], config_data_path)
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    
    idx_no_train = torch.LongTensor(
        np.setdiff1d(np.array(range(len(labels))), idx_train.cpu())).to(conf['device'])
    byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(conf['device'])
    byte_idx_train[idx_train] = True
    labels_init = initialize_label(idx_train, labels_one_hot).to(conf['device'])
    
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])

    # choose model type & setting optimizer for model
    teacher = choose_model(teacher_conf)
    teacher.load_state_dict(torch.load(t_PATH))
    teacher.eval()
    model = choose_model(conf, G, G.ndata['feat'], labels, byte_idx_train, labels_one_hot)
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
    
    # selector model generate
    selector_model = selector_model_init(conf)
    selector_loss = losses.TripletMarginLoss(margin=0.3, 
                                            swap=False,
                                            smooth_loss=False,
                                            triplets_per_anchor="all")
    mining_func = miners.TripletMarginMiner(margin=0.3, 
                                                     type_of_triplets="semihard")
    selector_optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector_model.parameters()), lr=0.01, weight_decay=0.001)
    
    # initialize mask
    num_masks = 20
    if conf['model_name'] in ['GCN', 'GAT', 'GCNII']:
        data_size = 64
    else: # GraphSAGE
        data_size = 128
    masks = get_new_random_masks(num_masks, data_size, data_size/2)
    
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

        if bad_counter == 200: # modify patience 200 -> 50
            break
    
    acc = test()
    
    print('The number of parameters in the student: {:04d}'.format(count_params(model)))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.2f}".format(acc*100))
