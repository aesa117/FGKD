import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.model_utils import *

from data.get_dataset import get_experiment_config
from data.utils import load_tensor_data, initialize_label

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from models.selector import *

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sage', action='store_true', default=False, type=str2bool, help='Student model type')
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--nlayer', type=int, default=5, help="Number of layer")
    parser.add_argument('--dataset', default="corafull", help="Dataset type")
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def configuration(args):
    conf['sage'] = args.sage
    conf['device'] = torch.device("cuda:" + str(args.device))
    conf = dict(conf, **args.__dict__)
    
    # dataset-specific configuration
    config_data_path = Path.cwd().joinpath('data', 'pretrain.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    
    print(conf)
    
    return conf, config_data_path

def selector_model_init(conf):
    if conf['sage'] == False:
        embedding_size = 64
        hidden_embedding = 128
    else:
        embedding_size = 128
        hidden_embedding = 256
    selector_model = MLP(num_layers=conf['nlayer'],
                         input_dim=features.shape[1],
                         hidden_dim=hidden_embedding, 
                         output_dim=embedding_size,
                         dropout=0.5)
    return selector_model

def train():
    selector_model.train()
    
    optimizer.zero_grad()
    model_output = selector_model(features[idx_train])
    loss_metric = loss(model_output, labels)
    acc = calculator.get_accuracy(model_output, labels)
    
    loss_metric.backward()
    optimizer.step()
    
    return loss_metric.item(), acc['NMI']   

def validate():
    selector_model.eval()
    
    with torch.no_grad():
        model_output = selector_model(features[idx_val])
        loss_metric = loss(model_output, labels)
        acc = calculator.get_accuracy(model_output, labels)
    
    return loss_metric.item(), acc['NMI']
    
def test():
    selector_model.load_state_dict(torch.load(checkpt_file))
    selector_model.eval()
    
    with torch.no_grad():
        model_output = selector_model(features[idx_test])
        loss_metric = loss(model_output, labels)
        acc = calculator.get_accuracy(model_output, labels)
    
    return loss_metric.item(), acc['NMI']
        

if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    
    conf, conf_path = configuration(args)
    checkpt_file = "./selector/"+"MLP_lr:"+str(conf['lr'])+"_wd:"+str(conf['wd'])+".pth"
    
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # dataset load
    adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['dataset'], args.labelrate, conf['device'], conf_path)
    
    features = features.to(conf['device'])
    adj = adj.to(conf['device'])
    labels = labels.to(conf['device'])
    
    selector_model = selector_model_init(conf)
    selector_model = selector_model.to(conf['device'])
    loss = losses.TripletMarginLoss(margin=0.3, 
                                            swap=False,
                                            smooth_loss=False,
                                            triplets_per_anchor="all",).to(conf['device'])
    calculator = AccuracyCalculator(include=("NMI")).to(conf['device'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, selector_model.parameters()), lr=conf['lr'], weight_decay=conf['wd'])
    
    for epoch in range(500):
        loss_train, acc_train, sel_loss_train = train()
        loss_val, acc_val, sel_loss_val = validate()
        if (epoch + 1) % 10 == 0:
            print('Epoch:{:04d}'.format(epoch+1),'train','loss:{:.3f}'.format(loss_train),'acc:{:.2f}'.format(acc_train*100),
            '| val','loss:{:.3f}'.format(loss_val),'acc:{:.2f}'.format(acc_val*100))
            print('selector training loss : {:.3f}'.format(sel_loss_train), 'validation loss : {:.3f}'.format(sel_loss_val))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(selector_model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 200: # modify patience 200
            break
    
    acc = test()
    
    print('The number of parameters in the student: {:04d}'.format(count_params(selector_model)))
    print('Load {}th epoch'.format(best_epoch))
    print("Test acc.:{:.2f}".format(acc*100))
