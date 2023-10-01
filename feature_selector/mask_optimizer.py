import numpy as np
import torch
import torchvision
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import normalized_mutual_info_score

class Mask:
    def __init__(self, num_mask, data_size, unmasked_data_size, perturbation_size):
        self.num_mask = num_mask # number of masks per node
        self.data_size = data_size
        self.unmasked_data_size = unmasked_data_size
        self.perturbed_mask = perturbation_size
        self.metric_loss = losses.TripletMarginLoss(margin=0.3, 
                                                    swap=False,
                                                    smooth_loss=False,
                                                    triplets_per_anchor="all")
        self.mining_func = miners.TripletMarginMiner(margin=0.3, 
                                                     type_of_triplets="semihard")
        self.clustering_score = AccuracyCalculator(include=("NMI"))
        self.masks = np.array()
        
    
    # optimal performance of clustering mask return
    def get_optimal_mask(model, unmasked_size, input):
        grad, loss = Mask.gradient(model, input)
        
    
    # random initialized mask generate
    def random_mask(self):
        mask_zero = np.zeros(shape=(1, self.data_size - self.unmasked_data_size))
        mask_one = np.ones(shape=(1, self.unmasked_data_size))
        mask = np.concatenate([mask_zero, mask_one], axis=1)
        mask_permuted = np.apply_along_axis(np.random.permutation, 1, mask)
        return mask_permuted
    
    # take perturbation to original mask
    def perturbed_mask(mask):
        where_0 = np.nonzero(mask-1)[0]
        where_1 = np.nonzero(mask)[0]
        i0 = np.random.randint(0, len(where_0), 1)
        i1 = np.random.randint(0, len(where_1), 1)
        mask[where_0[i0]] = 1
        mask[where_1[i1]] = 0
        return mask
    
    def gradient(model, hidden, label):
        hidden_tensor = torchvision.transforms.ToTensor(hidden)
        
        
        
    