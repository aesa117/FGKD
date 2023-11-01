import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.model_utils import *

from data.get_dataset import get_experiment_config
from data.utils import load_tensor_data, initialize_label

from pytorch_metric_learning import losses
from sklearn.metrics import f1_score
# from utils.metrics import accuracy
from sklearn.metrics.cluster import normalized_mutual_info_score
# from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from models.selector import *

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sage', action='store_true', default=False, help='Student model type')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--ms1', type=int, default=500, help="First milestone")
    parser.add_argument('--ms2', type=int, default=750, help="First milestone")
    parser.add_argument('--gm', type=float, default=0.1, help="Multistep gamma")
    parser.add_argument('--margin', type=float, default=0.3, help="Triplet margin")
    parser.add_argument('--nlayer', type=int, default=5, help="Number of layer")
    parser.add_argument('--dataset', default="corafull", help="Dataset type")
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    return parser.parse_args()

def configuration(args):
    conf = dict()
    conf['seed'] = 2022
    conf['device'] = torch.device("cuda:" + str(args.device))
    conf = dict(conf, **args.__dict__)
    
    # dataset-specific configuration
    config_data_path = Path.cwd().joinpath('data', 'pretrain.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    
    print(conf)
    
    return conf, config_data_path

def selector_model_init(conf):
    if conf['sage'] == False:
        hidden_embedding = 128
    else:
        hidden_embedding = 256
    selector_model = MLP(num_layers=conf['nlayer'],
                         input_dim=features.shape[1],
                         hidden_dim=hidden_embedding, 
                         output_dim=labels.max().item() + 1, # 40
                         dropout=0.5)
    return selector_model

def train():
    selector_model.train()
    
    optimizer.zero_grad()
    loss_metric = 0
    f1_macro = 0
    f1_micro = 0
    nmi = 0
    count = 0
    # batch size 80 * 75 = 6000
    for idx in range(80, int(idx_train.shape[0]), 80):
        model_output = selector_model(features[int(idx-80):idx])
        m_output = F.log_softmax(model_output, dim=1)
        m_out = np.argmax(m_output.detach().cpu(), axis=1)
        loss_metric += (loss(m_output, labels[int(idx-80):idx]) + F.nll_loss(m_output.to(conf['device']), labels[int(idx-80):idx].to(conf['device'])))
        
        f1_macro += f1_score(labels[int(idx-80):idx].detach().cpu(), m_out, average='macro')
        f1_micro += f1_score(labels[int(idx-80):idx].detach().cpu(), m_out, average='micro')
        nmi += normalized_mutual_info_score(labels[int(idx-80):idx].detach().cpu(), m_out)
        count += 1
    
    loss_metric /= count
    f1_macro /= count
    f1_micro /= count
    nmi /= count
    
    loss_metric.backward()
    optimizer.step()
    
    return loss_metric.item(), nmi, f1_macro, f1_micro

def validate():
    selector_model.eval()
    
    with torch.no_grad():
        loss_metric = 0
        f1_macro = 0
        f1_micro = 0
        nmi = 0
        count = 0
        # batch size 80 * 25 = 2000
        for idx in range(80, int(idx_val.shape[0]), 80):
            model_output = selector_model(features[int(idx-80):idx])
            m_output = F.log_softmax(model_output, dim=1)
            m_out = np.argmax(m_output.detach().cpu(), axis=1)
            loss_metric += (loss(m_output, labels[int(idx-80):idx]) + F.nll_loss(m_output.to(conf['device']), labels[int(idx-80):idx].to(conf['device'])))
            
            f1_macro += f1_score(labels[int(idx-80):idx].detach().cpu(), m_out, average='macro')
            f1_micro += f1_score(labels[int(idx-80):idx].detach().cpu(), m_out, average='micro')
            nmi += normalized_mutual_info_score(labels[int(idx-80):idx].detach().cpu(), m_out)
            count += 1
    
        loss_metric /= count
        f1_macro /= count
        f1_micro /= count
        nmi /= count
    
    return loss_metric.item(), nmi, f1_macro, f1_micro
    
def test():
    selector_model.load_state_dict(torch.load(checkpt_file))
    selector_model.eval()
    
    with torch.no_grad():
        loss_metric = 0
        f1_macro = 0
        f1_micro = 0
        nmi = 0
        count = 0
        # batch size 80 * 26 = 2080 = 2100 - 20
        for idx in range(80, int(idx_test.shape[0])-20, 80):
            model_output = selector_model(features[int(idx-80):idx])
            m_output = F.log_softmax(model_output, dim=1)
            m_out = np.argmax(m_output.detach().cpu(), axis=1)
            loss_metric += (loss(m_output, labels[int(idx-80):idx]) + F.nll_loss(m_output.to(conf['device']), labels[int(idx-80):idx].to(conf['device'])))
            
            f1_macro += f1_score(labels[int(idx-80):idx].detach().cpu(), m_out, average='macro')
            f1_micro += f1_score(labels[int(idx-80):idx].detach().cpu(), m_out, average='micro')
            nmi += normalized_mutual_info_score(labels[int(idx-80):idx].detach().cpu(), m_out)
            count += 1
        
        loss_metric /= count
        f1_macro /= count
        f1_micro /= count
        nmi /= count
    
    return loss_metric.item(), nmi, f1_macro, f1_micro
        

if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    
    conf, conf_path = configuration(args)
    if conf['sage'] == True:
        checkpt_file = "./selector/"+"MLP_SAGE_lr:"+str(conf['lr'])+"_wd:"+str(conf['wd'])+"_mg:"+str(conf['margin'])+"_nl:"+str(conf['nlayer'])+"_ms1"+str(conf['ms1'])+"_ms2"+str(conf['ms2'])+"_gm"+str(conf['gm'])+".pth"
    else :
        checkpt_file = "./selector/"+"MLP_lr:"+str(conf['lr'])+"_wd:"+str(conf['wd'])+"_mg:"+str(conf['margin'])+"_nl:"+str(conf['nlayer'])+"_ms1"+str(conf['ms1'])+"_ms2"+str(conf['ms2'])+"_gm"+str(conf['gm'])+".pth"
    
    # tensorboard name
    board_name = "MLP_lr:"+str(conf['lr'])+"_wd:"+str(conf['wd'])+"_mg:"+str(conf['margin'])+"_nl:"+str(conf['nlayer'])+"_ms1"+str(conf['ms1'])+"_ms2"+str(conf['ms2'])+"_gm"+str(conf['gm'])
    writer = SummaryWriter("./Log/Log_sel/"+board_name)
    
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # dataset load
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['dataset'], conf['device'], conf_path)
    
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])
    labels = labels.to(conf['device'])
    
    selector_model = selector_model_init(conf)
    selector_model = selector_model.to(conf['device'])
    loss = losses.TripletMarginLoss(margin=conf['margin'], 
                                            swap=False,
                                            smooth_loss=False,
                                            triplets_per_anchor="all",).to(conf['device'])
    # calculator = AccuracyCalculator(include=("NMI", "AMI")).to(conf['device'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector_model.parameters()), lr=conf['lr'], weight_decay=conf['wd'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[conf['ms1'], conf['ms2']], gamma=conf['gm'])
    
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    nmi = 0
    f1_macro, f1_micro = 0, 0
    for epoch in range(1000):
        loss_train, nmi_train, macro_train, micro_train = train()
        loss_val, nmi_val, macro_val, micro_val = validate()
        if (epoch + 1) % 10 == 0:
            print('Epoch:{:04d}'.format(epoch+1),'train:','loss:{:.3f}'.format(loss_train), 'nmi:{:.2f}'.format(nmi_train),'f1_macro:{:.2f}'.format(macro_train), 'f1_micro:{:.2f}'.format(micro_train),
            '| val','loss:{:.3f}'.format(loss_val), 'nmi:{:.2f}'.format(nmi_val), 'f1_macro:{:.2f}'.format(macro_val), 'f1_micro:{:.2f}'.format(micro_val))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            f1_macro = macro_val
            f1_micro = micro_val
            nmi = nmi_val
            torch.save(selector_model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 200: # modify patience 200
            break
        
        # write
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('NMI/train', nmi_train, epoch)
        writer.add_scalar('F1_macro/train', macro_train, epoch)
        writer.add_scalar('F1_micro/train', micro_train, epoch)
        
        writer.add_scalar('Loss/val', loss_val, epoch)
        writer.add_scalar('NMI/val', nmi_val, epoch)
        writer.add_scalar('F1_macro/val', macro_val, epoch)
        writer.add_scalar('F1_micro/val', micro_val, epoch)
        
        writer.flush()
    writer.close()
    
    loss_test, nmi_test, macro_test, micro_test = test()
    
    print('The number of parameters in the selector: {:04d}'.format(count_params(selector_model)))
    print('Load {}th epoch'.format(best_epoch))
    print('Test loss:{:.2f}'.format(loss_test), 'nmi:{:.2f}'.format(nmi_test), 'f1_macro:{:.2f}'.format(macro_test), 'f1_micro:{:.2f}'.format(micro_test))
