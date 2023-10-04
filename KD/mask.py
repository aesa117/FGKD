import torch
import torch.nn as nn
import numpy as np
from models.selector import *

# adapted from Feature Importance Ranking for Deep Learning

def get_new_random_masks(num_masks, data_size, unmasked_data_size):
    masks_zero = np.zeros(shape=(num_masks, data_size-int(unmasked_data_size)))
    masks_one = np.ones(shape=(num_masks, int(unmasked_data_size)))
    masks = np.concatenate([masks_zero, masks_one], axis=1)
    masks_permuted = np.apply_along_axis(np.random.permutation, 1, masks)

    return masks_permuted

def get_mutation_masks(masks, divide_val):
    def get_mutation_mask(mask):
        # 10%, 50% mutation 
        num_mutation_per_mask = int(len(mask)/divide_val)
        for _ in range(num_mutation_per_mask):
            where_0 = np.nonzero(mask-1)[0]
            where_1 = np.nonzero(mask)[0]
            i0 = np.random.randint(0, len(where_0), 1)
            i1 = np.random.randint(0, len(where_1), 1)
            mask[where_0[i0]] = 1
            mask[where_1[i1]] = 0
        return mask
    
    masks = np.apply_along_axis(get_mutation_mask, 1, masks)
    
    return masks

def selection(model, t_hiddens, labels, loss, optimizer, masks, num_mask, data_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        # calculate clustering score for all masks
        for i in range(num_mask):
            masked_hiddens = t_hiddens.mul(masks[i]).to(device)
            model_output = model(masked_hiddens)
            loss_metric = loss(model_output, labels)
            loss_metric = loss_metric.to(cpu)
            # score = clustering_score(model_output.to(cpu), labels.to(cpu))
            if i == 0:
                mask_losses = np.array([loss_metric])
                continue
            mask_losses = np.append(mask_losses, loss_metric)

        # sort mask_losses list for each metric loss (ascending order)
        sorted_index = np.argsort(mask_losses)
        sorted_masks = masks[sorted_index]
        sorted_masks = sorted_masks.to(cpu).numpy()
        
        # mask update
        # top 25% of masks => preserve
        new_mask = sorted_masks[0:int(num_mask/4)]
        # top 26%~50% of masks => 10% mutation
        new_mask = np.append(new_mask, get_mutation_masks(sorted_masks[int(num_mask/4):int(num_mask/4*2)], 10), axis=0)
        # top 51%~75% of masks => 50% mutation
        new_mask = np.append(new_mask, get_mutation_masks(sorted_masks[int(num_mask/4*2):int(num_mask/4*3)], 2), axis=0)
        # top 76%~100% of masks => random generate
        new_mask = np.append(new_mask, get_new_random_masks(int(num_mask/4), data_size, int(data_size/2)), axis=0)

        new_mask = torch.Tensor(new_mask).to(device)
        # best loss mask
        best_masked = t_hiddens.mul(new_mask[0])
        

    # selector model update using best mask
    model.train()
    
    optimizer.zero_grad()
    output = model(best_masked)
    loss_metric = loss(output, labels)
    loss_metric.backward()
    optimizer.step()
    
    return new_mask, loss_metric

def selection_val(model, t_hiddens, labels, loss, optimizer, masks, num_mask, data_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        # calculate clustering score for all masks
        for i in range(num_mask):
            masked_hiddens = t_hiddens.mul(masks[i]).to(device)
            model_output = model(masked_hiddens)
            loss_metric = loss(model_output, labels)
            loss_metric = loss_metric.to(cpu)
            # score = clustering_score(model_output.to(cpu), labels.to(cpu))
            if i == 0:
                mask_losses = np.array([loss_metric])
                continue
            mask_losses = np.append(mask_losses, loss_metric)

        # sort mask_losses list for each metric loss (ascending order)
        sorted_index = np.argsort(mask_losses)
        sorted_masks = masks[sorted_index]
        sorted_masks = sorted_masks.to(cpu).numpy()
        
        # mask update
        # top 25% of masks => preserve
        new_mask = sorted_masks[0:int(num_mask/4)]
        # top 26%~50% of masks => 10% mutation
        new_mask = np.append(new_mask, get_mutation_masks(sorted_masks[int(num_mask/4):int(num_mask/4*2)], 10), axis=0)
        # top 51%~75% of masks => 50% mutation
        new_mask = np.append(new_mask, get_mutation_masks(sorted_masks[int(num_mask/4*2):int(num_mask/4*3)], 2), axis=0)
        # top 76%~100% of masks => random generate
        new_mask = np.append(new_mask, get_new_random_masks(int(num_mask/4), data_size, int(data_size/2)), axis=0)

        new_mask = torch.Tensor(new_mask).to(device)
        best_mask = new_mask[0]
        best_masked = t_hiddens.mul(best_mask).to(device)
        
        output = model(best_masked)
        loss_metric = loss(output, labels)
    
    return new_mask, loss_metric.item()