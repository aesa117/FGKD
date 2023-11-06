import argparse
import numpy as np
import torch
import torch.optim as optim
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from models.model_KD import *
from models.PLP import *
from models.model_utils import *

from data.get_dataset import get_experiment_config
from data.utils import load_tensor_data, initialize_label
from utils.metrics import *
from utils.metrics import accuracy
from sklearn.metrics import f1_score

from mask import *
from models.selector import *

from pytorch_metric_learning import losses

def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--lbd_embd', type=float, default=1, help='selected embedding loss rate')
    parser.add_argument('--mask', type=int, default=20, help="mask size")
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--ptype', type=str, default='ind', help='plp type: ind(inductive); tra(transductive/onehot)')
    parser.add_argument('--mlp_layers', type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')

    return parser.parse_args()

def choose_teacher(conf, teacher_conf):
    if teacher_conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=teacher_conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=teacher_conf['num_layers'],
            activation=F.relu,
            dropout=teacher_conf['dropout']).to(conf['device'])
    elif teacher_conf['model_name'] =='GAT':
        num_heads = teacher_conf['num_heads']
        num_layers = teacher_conf['num_layers']
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=teacher_conf['hidden'],
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=teacher_conf['dropout'],
                    attn_drop=teacher_conf['att_dropout'],
                    negative_slope=teacher_conf['alpha'],     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif teacher_conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=teacher_conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=teacher_conf['num_layers'],
                          activation=F.relu,
                          dropout=teacher_conf['dropout'],
                          aggregator_type=teacher_conf['agg_type']).to(conf['device'])
    elif teacher_conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'citeseer':
            teacher_conf['hidden'] = 256
            teacher_conf['lamda'] = 0.6
            teacher_conf['dropout'] = 0.7
        elif conf['dataset'] == 'pubmed':
            teacher_conf['hidden'] = 256
            teacher_conf['lamda'] = 0.4
            teacher_conf['dropout'] = 0.5
        model = GCNII(nfeat=features.shape[1],
                      nlayers=teacher_conf['layer'],
                      nhidden=teacher_conf['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=teacher_conf['dropout'],
                      lamda=teacher_conf['lamda'],
                      alpha=teacher_conf['alpha'],
                      variant=False).to(conf['device'])
    else:
        raise ValueError(f'Undefined Model.')
    return model

def choose_model(conf):
    if conf['model_name'] == 'PLP':
        model = PLP(g=G,
                    num_layers=conf['num_layers'],
                    in_dim=G.ndata['feat'].shape[1],
                    emb_dim=conf['reduced_dim'],
                    num_classes=labels.max().item() + 1,
                    activation=F.relu,
                    feat_drop=conf['dropout'],
                    attn_drop=conf['att_dropout'],
                    residual=False,
                    byte_idx_train=byte_idx_train,
                    labels_one_hot=labels_one_hot,
                    ptype=conf['ptype'],
                    mlp_layers=conf['mlp_layers']).to(conf['device'])
    else:
        raise ValueError(f'Undefined Model.')
    return model

def selector_model_init(selector_path):
    embedding_size = 256
    out_size = 64
    
    selector_model = MLP(num_layers=5,
                         input_dim=8710, # number of features in CoraFull
                         hidden_dim=embedding_size, 
                         output_dim=40, # number of classes
                         dropout=0.5)
    
    selector_model.load_state_dict(torch.load(selector_path))
    
    # replace first layer & final layer
    selector_model.layers = selector_model.layers[1:-1]
    input = nn.Linear(embedding_size, embedding_size)
    output = nn.Linear(embedding_size, out_size)
    selector_model.layers = nn.Sequential(input, *selector_model.layers, output)
    
    return selector_model

def kl_kernel(t_x, s_x):
    kl_loss_op = torch.nn.KLDivLoss(reduction='none')
    t_x = F.softmax(t_x, dim=1)
    s_x = F.log_softmax(s_x, dim=1)
    return torch.mean(torch.sum(kl_loss_op(s_x, t_x), dim=1))

def train(masks, conf, teacher_conf):
    teacher.eval()
    model.train()
    optimizer.zero_grad()

    if teacher_conf['model_name'] == 'GCN':
        t_output, t_hidden = teacher(G.ndata['feat'])
    elif teacher_conf['model_name'] == 'GAT':
        t_output, t_hidden = teacher(G.ndata['feat'])[0:2]
    elif teacher_conf['model_name'] == 'GraphSAGE':
        t_output, t_hidden = teacher(G, G.ndata['feat'])
    elif teacher_conf['model_name'] == 'GCNII':
        t_output, t_hidden = teacher(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    
    s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
    s_out = F.log_softmax(s_output, dim=1)
    t_out = F.log_softmax(t_output, dim=1)
    
    acc_train = accuracy(s_out[idx_no_train], labels[idx_no_train].to(conf['device']))

    # mask selection - training and extract masks for clustering score
    updated_masks, sel_loss = selection(selector_model, t_hidden[idx_no_train], labels[idx_no_train], selector_loss, selector_optimizer, masks, num_masks, mask_size, unmask_size)
    best_mask = updated_masks[0]
    hidden_embedding = t_hidden[idx_no_train]
    
    idx = torch.nonzero(best_mask!=0).T
    masked_t_hidden = hidden_embedding[:, idx]
    
    # distance loss
    loss_dist = my_loss(s_out[idx_no_train], t_out[idx_no_train], 2).to(conf['device'])
    
    # embedding loss
    t_x = torch.squeeze(masked_t_hidden)
    s_x = s_hidden[idx_no_train]
    loss_hidden = kl_kernel(t_x, s_x)

    # loss_final
    loss_train = loss_dist + args.lbd_embd*loss_hidden
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train.item(), sel_loss, updated_masks

def validate(masks, conf, teacher_conf):
    teacher.eval()
    model.eval()

    with torch.no_grad():
        if teacher_conf['model_name'] == 'GCN':
            t_output, t_hidden = teacher(G.ndata['feat'])
        elif teacher_conf['model_name'] == 'GAT':
            t_output, t_hidden = teacher(G.ndata['feat'])[0:2]
        elif teacher_conf['model_name'] == 'GraphSAGE':
            t_output, t_hidden = teacher(G, G.ndata['feat'])
        elif teacher_conf['model_name'] == 'GCNII':
            t_output, t_hidden = teacher(features, adj)
        else:
            raise ValueError(f'Undefined Model')
        
        s_output, s_hidden = model(G.ndata['feat'], labels_init)[0:2]
        s_out = F.log_softmax(s_output, dim=1)
        t_out = F.log_softmax(t_output, dim=1)
        
        acc_val = accuracy(s_out[idx_val], labels[idx_val].to(conf['device']))

        # mask selection - training and extract masks for clustering score
        updated_masks, sel_loss = selection(selector_model, t_hidden[idx_val], labels[idx_val], selector_loss, selector_optimizer, masks, num_masks, mask_size, unmask_size)
        best_mask = updated_masks[0]
        hidden_embedding = t_hidden[idx_val]
        
        idx = torch.nonzero(best_mask!=0).T
        masked_t_hidden = hidden_embedding[:, idx]
        
        # distance loss
        loss_dist = my_loss(s_out[idx_val], t_out[idx_val], 2).to(conf['device'])
        
        # embedding loss
        t_x = torch.squeeze(masked_t_hidden)
        s_x = s_hidden[idx_val]
        loss_hidden = kl_kernel(t_x, s_x)

        # loss_final
        loss_val = loss_dist + args.lbd_embd*loss_hidden
        
    return loss_val.item(), acc_val.item(), sel_loss, updated_masks

def test(conf, checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(G.ndata['feat'], labels_init)[0]
        
        out = F.log_softmax(output, dim=1)
        
        acc_test = accuracy(out[idx_test], labels[idx_test].to(conf['device']))
        
    return acc_test.item()

if __name__ == '__main__':
    temperature = 2
    args = arg_parse(argparse.ArgumentParser())
    
    # teacher model-specific configuration
    teacher_config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    teacher_conf = get_training_config(teacher_config_path, model_name=args.teacher)
    
    # student model-specific configuration
    config_path = Path.cwd().joinpath('models', 'distill.conf.yaml')
    conf = get_training_config(config_path, model_name=args.student)
    
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
    
    # check point file path
    t_PATH = "./teacher/Teacher_"+str(teacher_conf['model_name'])+"dataset_"+str(conf['dataset'])+".pth"
    
    checkpt_file = "./KD/CPF/Student_"+str(conf['model_name'])+"dataset_"+str(conf['dataset'])
    checkpt_file += "lbd_embd"+str(conf['lbd_embd'])+".pth"
    
    # tensorboard name
    board_name = "CPF_student_"+str(conf['model_name'])+"dataset_"+str(conf['dataset'])+"lbd_embd"+str(conf['lbd_embd'])
    writer = SummaryWriter("./Log/Log_CPF/"+board_name)

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
    teacher = choose_teacher(conf, teacher_conf)
    teacher.load_state_dict(torch.load(t_PATH))
    teacher.eval()
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
    
    # initialize mask
    num_masks = conf['mask']
    mask_size = 64
    unmask_size = 256-mask_size
    masks = get_new_random_masks(num_masks, mask_size, unmask_size)
    masks = torch.Tensor(masks).to(conf['device'])
    
    # selector model generate
    selector_path = "./selector/MLP_lr:0.01_wd:0.001_mg:0.3_nl:5_ms1400_ms2800_gm0.1.pth"
    selector_model = selector_model_init(selector_path)
    selector_model.load_state_dict(torch.load(selector_path))
    selector_model = selector_model.to(conf['device'])
    selector_loss = losses.TripletMarginLoss(margin=0.3, 
                                            swap=False,
                                            smooth_loss=False,
                                            triplets_per_anchor="all",).to(conf['device'])
    selector_optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector_model.parameters()), lr=0.001, weight_decay=0.001)
    
    start = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(500):
        loss_train, acc_train, sel_train, masks = train(masks)
        loss_val, acc_val, sel_val, masks = validate(masks)
        if (epoch + 1) % 10 == 0:
            print('Epoch:{:04d}'.format(epoch+1),'train:','loss:{:.3f}'.format(loss_train), 'acc:{:.2f}'.format(acc_train*100),
            '| val','loss:{:.3f}'.format(loss_val), 'acc:{:.2f}'.format(acc_val*100))
            print('selector model train loss:{:.3f}'.format(sel_train), 'val loss{:.3f}'.format(sel_val))
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
    result_time = str(datetime.timedelta(seconds=end-start)).split(".")
    
    acc_test = test(conf, checkpt_file)
    
    print('The number of parameters in the teacher: {:04d}'.format(count_params(model)))
    print('Load {}th epoch'.format(best_epoch))
    print('Student test acc:{:.2f}'.format(acc_test*100))
    print('Training Time: ', result_time[0])