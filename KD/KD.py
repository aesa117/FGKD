import argparse
import numpy as np
import torch
import torch.optim as optim
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from models.model_KD import *
from models.model_utils import *

from data.get_dataset import get_experiment_config
from data.utils import load_tensor_data, initialize_label

from utils.metrics import accuracy
from sklearn.metrics import f1_score

from mask import *
from models.selector import *

from pytorch_metric_learning import losses


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--lbd_pred', type=float, default=0, help='lambda for prediction loss')
    parser.add_argument('--lbd_embd', type=float, default=0, help='lambda for embedding loss')
    parser.add_argument('--mask', type=int, default=20, help="mask size")
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')

    return parser.parse_args()

def kl_kernel(t_x, s_x):
    kl_loss_op = torch.nn.KLDivLoss(reduction='none')
    t_x = F.softmax(t_x, dim=1)
    s_x = F.log_softmax(s_x, dim=1)
    return torch.mean(torch.sum(kl_loss_op(s_x, t_x), dim=1))

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
                          n_layers=conf['layer'],
                          activation=F.relu,
                          dropout=conf['dropout'],
                          aggregator_type=conf['agg_type']).to(conf['device'])
    elif conf['model_name'] == 'GCNII':
        if args.dataset == 'citeseer':
            conf['hidden'] = 64
            teacher_conf['hidden'] = 256
            conf['lamda'] = 0.6
            conf['dropout'] = 0.7
        elif args.dataset == 'pubmed':
            conf['hidden'] = 64
            teacher_conf['hidden'] = 256
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

def selector_model_init(conf):
    if conf['model_name'] in ['GCN', 'GCNII', 'GAT']:
        embedding_size = 128
        inout_size = 32
    else: # GraphSAGE
        embedding_size = 256
        inout_size = 64
    
    selector_model = MLP(num_layers=conf['nlayer'],
                         input_dim=inout_size,
                         hidden_dim=embedding_size, 
                         output_dim=inout_size,
                         dropout=0.5)
    
    # delete first layer & final layer
    selector_model.layers = selector_model.layers[1:-1]
    input = nn.Linear(inout_size, embedding_size)
    output = nn.Linear(embedding_size, inout_size)
    selector_model.layers = nn.Sequential(input, *selector_model.layers, output)
    
    return selector_model

def train():
    teacher.eval()
    model.train()
    optimizer.zero_grad()

    if conf['model_name'] == 'GCN':
        t_output, t_hidden = teacher(G.ndata['feat'])
        s_output, s_hidden = model(G.ndata['feat'])
    elif conf['model_name'] == 'GAT':
        t_output, t_hidden = teacher(G.ndata['feat'])[0:2]
        s_output, s_hidden = model(G.ndata['feat'])[0:2]
    elif conf['model_name'] == 'GraphSAGE':
        t_output, t_hidden = teacher(G, G.ndata['feat'])
        s_output, s_hidden = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'GCNII':
        t_output, t_hidden = teacher(features, adj)
        s_output, s_hidden = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')

    s_out = F.log_softmax(s_output, dim=1)
    out_arg = np.argmax(s_out.detach().cpu(), axis=1)
    loss_CE = F.nll_loss(s_out[idx_train], labels[idx_train].to(conf['device']))
    acc_train = accuracy(s_out[idx_train], labels[idx_train].to(conf['device']))
    f1_macro = f1_score(labels[idx_train].detach().cpu(), out_arg, average='macro')
    f1_micro = f1_score(labels[idx_train].detach().cpu(), out_arg, average='micro')

    # mask selection - training and extract masks for clustering score
    updated_masks, sel_loss = selection(selector_model, t_hidden[idx_train], labels[idx_train], selector_loss, selector_optimizer, masks, num_masks, mask_size, unmask_size)
    best_mask = updated_masks[0]
    hidden_embedding = t_hidden[idx_train]
    
    idx = torch.nonzero(best_mask!=0).T
    masked_t_hidden = hidden_embedding[:, idx]
    
    # loss_task
    t_output = t_output/temperature
    t_y = t_output[idx_train]
    s_y = s_output[idx_train]
    loss_task = kl_kernel(t_y, s_y)

    # loss_hidden
    t_x = masked_t_hidden
    s_x = s_hidden[idx_train]
    loss_hidden = kl_kernel(t_x, s_x)

    # loss_final
    loss_train = loss_CE + args.lbd_pred*loss_task + args.lbd_embd*loss_hidden
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train.item(), f1_macro, f1_micro, sel_loss

def validate():
    teacher.eval()
    model.eval()

    with torch.no_grad():
        # loss_CE
        if conf['model_name'] == 'GCN':
            t_output, t_hidden = teacher(G.ndata['feat'])
            s_output, s_hidden = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            t_output, t_hidden = teacher(G.ndata['feat'])[0:2]
            s_output, s_hidden = model(G.ndata['feat'])[0:2]
        elif conf['model_name'] == 'GraphSAGE':
            t_output, t_hidden = teacher(G, G.ndata['feat'])
            s_output, s_hidden = model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            t_output, t_hidden = teacher(features, adj)
            s_output, s_hidden = model(features, adj)
        else:
            raise ValueError(f'Undefined Model')
        
        s_out = F.log_softmax(s_output, dim=1)
        out_arg = np.argmax(s_out.detach().cpu(), axis=1)
        loss_CE = F.nll_loss(s_out[idx_val], labels[idx_val].to(conf['device']))
        acc_val = accuracy(s_out[idx_val], labels[idx_train].to(conf['device']))
        f1_macro = f1_score(labels[idx_val].detach().cpu(), out_arg, average='macro')
        f1_micro = f1_score(labels[idx_val].detach().cpu(), out_arg, average='micro')
        
        # mask selection - validation and extract masks for clustering score
        updated_masks, sel_loss = selection_val(selector_model, t_hidden[idx_val], labels[idx_val], selector_loss, selector_optimizer, masks, num_masks, mask_size, unmask_size)
        best_mask = updated_masks[0]
        hidden_embedding = t_hidden[idx_train]
        
        idx = torch.nonzero(best_mask!=0).T
        masked_t_hidden = hidden_embedding[:, idx]

        # loss_task
        t_y = t_output[idx_val]
        s_y = s_output[idx_val]
        loss_task = kl_kernel(t_y, s_y)

        # loss_hidden
        t_x = masked_t_hidden
        s_x = s_hidden[idx_val]
        loss_hidden = kl_kernel(t_x, s_x)

        # loss_final- lbd_pred, lbe_embd are still not defined
        loss_val = loss_CE + args.lbd_pred*loss_task + args.lbd_embd*loss_hidden
    
    return loss_val.item(), acc_val.item(), f1_macro, f1_micro, sel_loss

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    
    with torch.no_grad():
        if conf['model_name'] == 'GCN':
            output, _ = model(G.ndata['feat'])
        elif conf['model_name'] == 'GAT':
            output, _ = model(G.ndata['feat'])[0:2]
        elif conf['model_name'] == 'GraphSAGE':
            output, _= model(G, G.ndata['feat'])
        elif conf['model_name'] == 'GCNII':
            output, _ = model(features, adj)
        else:
            raise ValueError(f'Undefined Model')
        
        out = F.log_softmax(output, dim=1)
        out_arg = np.argmax(out.detach().cpu(), axis=1)
        
        acc_test = accuracy(out[idx_test], labels[idx_test].to(conf['device']))
        f1_macro = f1_score(labels[idx_test].detach().cpu(), out_arg, average='macro')
        f1_micro = f1_score(labels[idx_test].detach().cpu(), out_arg, average='micro')
        
    return acc_test.item(), f1_macro, f1_micro


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
    t_PATH = "./teacher/Teacher_"+str(teacher_conf['model_name'])+"dataset_"+str(teacher_conf['dataset'])
    t_PATH += str(teacher_conf['dataset'])+"_lr:"+str(teacher_conf['learning_rate'])+"_wd:"+str(teacher_conf['weight_decay'])+"_nl:"+str(teacher_conf['num_layers'])+".pth"
    
    if conf['model_name'] in ['GCN', 'GCNII', 'GAT']:
        selector_path = "./selector/MLP_lr:0.01_wd:0.001_mg:0.3_nl:5_ms1400_ms2800_gm0.1.pth"
        mask_size = 32
        unmask_size = 96
    elif conf['model_name'] == 'GraphSAGE':
        selector_path = "./selector/MLP_large_lr:0.01_wd:0.001_mg:0.3_nl:5_ms1400_ms2800_gm0.1.pth"
        mask_size = 64
        unmask_size = 192
    
    checkpt_file = "./KD_student/Student_"+str(conf['model_name'])+"dataset_"+str(conf['dataset'])
    checkpt_file += str(conf['dataset'])+"_lr:"+str(conf['learning_rate'])+"_wd:"+str(conf['weight_decay'])+"_nl:"+str(conf['num_layers'])
    checkpt_file += "lbd_pred:"+str(conf['lbd_pred'])+"lbd_embd"+str(conf['lbd_embd'])+".pth"
    
    # tensorboard name
    board_name = "KD_student_"+str(conf['model_name'])+"dataset_"+str(conf['dataset'])+"_lr:"+str(conf['learning_rate'])+"_wd:"+str(conf['weight_decay'])+"_nl:"+str(conf['num_layers'])
    writer = SummaryWriter("./Log/Log_KD/"+board_name)

    # Load data
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['dataset'], conf['device'], config_data_path)
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    labels_init = initialize_label(idx_train, labels_one_hot).to(conf['device'])
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])

    # choose model type & setting optimizer for model
    teacher = choose_model(teacher_conf)
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
    
    # selector model generate
    selector_model = selector_model_init(conf)
    selector_model.load_state_dict(torch.load(selector_path))
    selector_model = selector_model.to(conf['device'])
    selector_loss = losses.TripletMarginLoss(margin=0.3, 
                                            swap=False,
                                            smooth_loss=False,
                                            triplets_per_anchor="all",).to(conf['device'])
    selector_optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector_model.parameters()), lr=0.001, weight_decay=0.001)
    
    # initialize mask
    num_masks = conf['mask']
    masks = get_new_random_masks(num_masks, mask_size, unmask_size)
    masks = torch.Tensor(masks).to(conf['device'])
    
    
    start = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    f1_macro = 0
    f1_micro = 0
    for epoch in range(500):
        loss_train, acc_train, macro_train, micro_train, sel_train = train()
        loss_val, acc_val, macro_val, micro_val, sel_val = validate()
        if (epoch + 1) % 10 == 0:
            print('Epoch:{:04d}'.format(epoch+1),'train:','loss:{:.3f}'.format(loss_train), 'acc:{:.2f}'.format(acc_train*100),'f1_macro:{:.2f}'.format(macro_train), 'f1_micro:{:.2f}'.format(micro_train),
            '| val','loss:{:.3f}'.format(loss_val), 'acc:{:.2f}'.format(acc_val*100), 'f1_macro:{:.2f}'.format(macro_val), 'f1_micro:{:.2f}'.format(micro_val))
            print('selector model train loss:{:.3f}'.format(sel_train), 'val loss{:.3f}'.format(sel_val))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            f1_macro = macro_val
            f1_micro = micro_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 200: # modify patience 200 -> 50
            break
        
        # write
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Acc/train', acc_train, epoch)
        writer.add_scalar('F1_macro/train', macro_train, epoch)
        writer.add_scalar('F1_micro/train', micro_train, epoch)
        
        writer.add_scalar('Loss/val', loss_val, epoch)
        writer.add_scalar('Acc/val', acc_val, epoch)
        writer.add_scalar('F1_macro/val', macro_val, epoch)
        writer.add_scalar('F1_micro/val', micro_val, epoch)
    writer.close()
        
    end = time.time()
    result_time = str(datetime.timedelta(seconds=end-start)).split(".")
    
    acc_test, macro_test, micro_test = test()
    
    print('The number of parameters in the teacher: {:04d}'.format(count_params(model)))
    print('Load {}th epoch'.format(best_epoch))
    print('Student test acc:{:.2f}'.format(acc_test*100), 'f1_macro:{:.2f}'.format(macro_test), 'f1_micro:{:.2f}'.format(micro_test))
    print('Training Time: ', result_time[0])