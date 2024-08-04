import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import math
import argparse
from tqdm import tqdm
from config.conf import cfg
import torch.optim as optim
from data.data import get_loader
from copy import deepcopy
from tta_algo.build import build_tta_adapter
from tta_attack.dia import DIA
from tta_attack.u_dia import U_DIA
from utils.util import accuracy, AverageMeter
from torch.profiler import profile, record_function, ProfilerActivity 
from torch.func import functional_call, vmap, grad

def set_deterministic(seed=42, cudnn=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = cudnn

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def nullspace(feat, thresh=0.99):
  eigenvalues, eigenvecs = torch.linalg.eigh(feat)
  #eigenvalues= eigenvalues.real
  sorted_indices = torch.argsort(eigenvalues, descending=True)
  eigenvecs = eigenvecs[:, sorted_indices]
  eigenvalues = eigenvalues[sorted_indices]
  cum_energy = torch.cumsum(eigenvalues**2, dim=0)/torch.sum(eigenvalues**2)
  num_components = max(1, torch.sum(cum_energy < thresh).item())
  eigenvecs = eigenvecs[:, num_components:]
  #eigenvecs = eigenvecs[:, -10:]
  return eigenvecs

def center_kernel_matrix(kernel_matrix):
    """
    Centering the data in the feature space only using the (uncentered) Kernel-Matrix
    :param kernel_matrix: uncentered kernel matrix
    :return: centered kernel matrix
    """
    n = kernel_matrix.size(0)
    column_means = torch.mean(kernel_matrix, dim=0, keepdim=True)
    matrix_mean = torch.mean(column_means)
    centered_kernel_matrix = kernel_matrix - column_means - column_means.T + matrix_mean
    return centered_kernel_matrix

def create_nullspace(K, labels):
    """
    Compute the DNS projection
    :param K: Kernel matrix
    :param labels: Class labels
    :return: proj_null: Projection matrix for null space
    """
    classes = torch.unique(labels)

    # Check kernel matrix
    n, m = K.size()
    if n != m:
        raise ValueError('Kernel matrix must be quadratic')

    # Calculate weights of orthonormal basis in kernel space
    centeredK = center_kernel_matrix(K)
    basisvecsValues, basisvecs = torch.linalg.eigh(centeredK)
    basisvecs = basisvecs[:, basisvecsValues > 1e-12]
    basisvecsValues = basisvecsValues[basisvecsValues > 1e-12]
    # cum_energy = torch.cumsum(basisvecsValues**2, dim=0)/torch.sum(basisvecsValues**2)
    # num_components = max(1, torch.sum(cum_energy < 0.99).item())
    # basisvecs = basisvecs[:, :num_components]
    # basisvecsValues = basisvecsValues[ : num_components]
    basisvecsValues = torch.diag(1.0 / torch.sqrt(basisvecsValues))
    basisvecs = basisvecs @ basisvecsValues
    print(f'Shape of basis vectors {basisvecs.shape}')

    # Calculate transformation T of within class scatter Sw
    L = torch.zeros(n, n, dtype=K.dtype, device=K.device)
    for c in classes:
        mask = (labels == c)
        #print(mask)
        L[mask, mask] = 1.0 / mask.sum()

    M = torch.ones(m, m, dtype=K.dtype, device=K.device) / m
    H = ((torch.eye(m, dtype=K.dtype, device=K.device) - M) @ basisvecs).T @ K @ (torch.eye(n, dtype=K.dtype, device=K.device) - L)
    T = H @ H.T

    # Calculate weights for null space
    #eigenvecs = torch.linalg.null_space(T)
    eigenvecs = nullspace(T)
    print(f'Shape of Eigen vectors of T {eigenvecs.shape}')
    # if eigenvecs.size(1) < 1:
    #     eigenvals, eigenvecs = torch.linalg.eig(T)
    #     eigenvals = eigenvals.real
    #     min_val, min_ID = torch.min(eigenvals, dim=0)
    #     eigenvecs = eigenvecs[:, min_ID].unsqueeze(1)

    # Calculate null space projection
    proj_null = (torch.eye(m, dtype=K.dtype, device=K.device) - M) @ basisvecs @ eigenvecs
    return proj_null

# def RBF_kernel(X, Y):
#     # norm1 = torch.sum(X ** 2, dim=1, keepdim=True)
#     # norm2 = torch.sum(Y ** 2, dim=1, keepdim=True)
#     # dist = norm1 + norm2.T - 2 * torch.mm(X, Y.T)
#     dist = torch.cdist(X, Y, p=2)**2
#     mu = torch.sqrt(torch.mean(dist) / 2)
#     #gamma = 0.5 / (mu ** 2)
#     gamma = 1.0/10.0
#     K_train = torch.exp(-gamma * dist)

#     # norm1 = torch.sum(X ** 2, dim=1, keepdim=True)
#     # norm2 = torch.sum(Y ** 2, dim=1, keepdim=True)
#     # dist = norm1 + norm2.T - 2 * torch.mm(X, Y.T)
#     # K_test = torch.exp(-0.5 / mu ** 2 * dist)

#     return K_train

def RBF_kernel(X, Y):
    
    dist = torch.cdist(X, Y, p=2)**2
    mu = torch.sqrt(torch.mean(dist) / 2)
    #gamma = 0.5 / (mu ** 2)
    K_train = torch.exp(-0.5/mu**2 * dist)

    # norm1 = torch.sum(X ** 2, dim=1, keepdim=True)
    # norm2 = torch.sum(Y ** 2, dim=1, keepdim=True)
    # dist = norm1 + norm2.T - 2 * torch.mm(X, Y.T)
    # K_test = torch.exp(-0.5 / mu ** 2 * dist)

    return K_train


def log_activation_src(model, layers, data_loader, device):
  
  activations = {layer_id: torch.empty(0) for layer_id in layers}
  labels = torch.empty(0)
  handles = {layer_id:[] for layer_id in layers}
  activation_hook = Activation_Hook()


  model = model.to(device)
  model.eval()
  for layer_id in layers:
    layer = dict([*model.named_modules()])[layer_id]
    handles[layer_id] = layer.register_forward_hook(activation_hook)

  for batch_id, (data, label) in enumerate(data_loader):
    data, label = data.to(device), label
    labels = torch.cat((labels, label), dim=0)
    outputs = model(data)
    for layer_id in layers:
      layer = dict([*model.named_modules()])[layer_id]
      activations[layer_id] = torch.cat((activations[layer_id], layer.act), dim=0)
      del(layer.act)
    if batch_id > 8:
      break

  for layer_id in layers:
      handles[layer_id].remove()

  return activations, labels


class Activation_Hook():
  def __call__(self, module, input, output):
    module.act = output.detach().clone().flatten(start_dim=1).cpu()



def null_space_projection(model, layers, data_loader, activations, 
                          proj_null, proj_cls, cfg, device):

    activations_src = {layer_id:torch.empty(0) for layer_id in layers }

    # pred_labels_src = torch.empty(0)
    # scores_null_1 = torch.empty(0)
    # scores_null_2 = torch.empty(0)
    
    handles_src = {layer_id:[] for layer_id in layers} 
    activation_hook = Activation_Hook()
    model = model.to(device)
    src_model = deepcopy(model)
    src_model = src_model.to(device)
    src_model = src_model.eval()
    victim_model = deepcopy(model)
    victim_model = victim_model.to(device)

    for layer_id in layers:
        layer = dict([*src_model.named_modules()])[layer_id]
        handles_src[layer_id] = layer.register_forward_hook(activation_hook)
        
    tta_adapter_class = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_class(cfg, model)
    tta_adapter_victim = tta_adapter_class(cfg, victim_model)

    if cfg.BASE.ATTACK == "dia":
        attack = DIA(cfg=cfg)
    elif cfg.BASE.ATTACK == "u_dia":
        attack = U_DIA(cfg=cfg, layers=['avgpool'])
    else:
        raise NotImplementedError("Specify an attack Name that is implemeted")
    
    mal_num = math.ceil(cfg.DATA.BATCH_SIZE * cfg.DIA.MAL_PORTION)

    scores_mal_proj = torch.tensor([])
    scores_benign_proj = torch.tensor([])
    scores_mal_dist = torch.tensor([])
    scores_benign_dist = torch.tensor([])

    bar = tqdm(data_loader)
    for n_batch, (inputs,labels) in enumerate(bar):
        if inputs.shape[0] != cfg.DATA.BATCH_SIZE:
            continue

        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        x_adv = attack.generate_attack(sur_model=tta_adapter.model,
                                       x = inputs, y=labels)
        outputs_clean = tta_adapter.forward(inputs).detach().cpu()
        outputs_mal = tta_adapter_victim.forward(x_adv).detach().cpu()
        outputs_src = src_model(x_adv).detach().cpu()
        pred_labels_src = outputs_src.max(1)[1]

        for layer_id in layers:
            layer = dict([*src_model.named_modules()])[layer_id]
            activations_src[layer_id] = layer.act
            del layer.act
        
        K = {n:torch.empty(0) for n in activation_layers}
        # projection = {n:torch.empty(0) for n in activation_layers}
        dist = torch.empty(len(pred_labels_src),0)
        proj = torch.empty(len(pred_labels_src),0)
        scores_null = {}
        projection = {}
        # dist = {n:torch.empty(0) for n in activation_layers}
        a = torch.arange(num_classes)
        mask = pred_labels_src.unsqueeze(1) == a.unsqueeze(0)

        for layer_name in activation_layers:
            K[layer_name] = RBF_kernel(activations[layer_name], activations_src[layer_name])
            projection[layer_name] = K[layer_name].T @ proj_null[layer_name]
            scores_null[layer_name] = torch.cdist(projection[layer_name], proj_cls[layer_name])
            dist = torch.cat((dist,(scores_null[layer_name]*mask).sum(dim=1).unsqueeze(1)), dim=1)
            proj = torch.cat((proj, projection[layer_name]), dim=1)
            
        scores_benign_dist = torch.cat((scores_benign_dist, dist[:-mal_num]), dim=0)
        scores_mal_dist = torch.cat((scores_mal_dist, dist[-mal_num:]), dim=0)
        scores_benign_proj = torch.cat((scores_benign_proj, proj[:-mal_num]), dim=0)
        scores_mal_proj = torch.cat((scores_mal_proj, proj[-mal_num:]), dim=0)

        model_state, optimizer_state = copy_model_and_optimizer(tta_adapter.model, tta_adapter.optimizer)
        load_model_and_optimizer(tta_adapter_victim.model, tta_adapter_victim.optimizer, 
                                 model_state, optimizer_state)
        # if outputs_clean[:-mal_num].size()[0]:
        #     normal_accuracy = accuracy(outputs_clean[:-mal_num], labels[:-mal_num].cpu())
        #     attacked_accuracy = accuracy(outputs_mal[:-mal_num], labels[:-mal_num].cpu())
        #     acc_normal.update(normal_accuracy)
        #     acc_attacked.update(attacked_accuracy)
        bar.set_description(f"Batch => [{n_batch}/{len(data_loader)}]")
        # bar.set_postfix(normal = acc_normal.avg, attacked = acc_attacked.avg)
        if n_batch == 10:
           break

    for layer_id in layers:
       handles_src[layer_id].remove()
    return scores_benign_dist, scores_mal_dist, scores_benign_proj, scores_mal_proj


        
        
def update_configs(cfg,args):
    #gpu_id
    if args.gpu_id:
        cfg.BASE.GPU_ID = args.gpu_id
    #tta method 
    if args.tta:
        cfg.TTA.NAME = args.tta
    # batch size
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    if args.pseudo:
        cfg.DIA.PSEUDO = args.pseudo
    if args.attack:
        cfg.BASE.ATTACK=args.attack
    return cfg

if __name__ == "__main__":
    # Setting all the random seeds for reproducibility
    set_deterministic(seed=15, cudnn=True)

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs','--batch_size',type=int, help="Specify Batch Size",default=64)
    parser.add_argument('--gpu_id',type=int, help="Specify GPU ID")
    parser.add_argument('--tta',type=str, help="Specify the TTA algorithm to be used")
    parser.add_argument('--mal_num',type=int, help="Specify the number of malicious sample in the batch")
    parser.add_argument('--pseudo',type=bool, help="Specify whether to use pseudo labels or not")
    parser.add_argument('--loss', type=str,help="Specify the loss to be used by u_dia" )
    parser.add_argument('--attack', type=str, help="Specify the attack algorithm" )
    args = parser.parse_args()

    # update the default config values from command line arguments
    conf = update_configs(cfg,args)
    device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")

    # Define source dataloader and model for discriminative null space construction
    src_data_loader = get_loader(cfg=cfg, corruption='normal')
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                        "cifar10_resnet20", pretrained=True)
    
    # calculate class wise null space projections for source data
    activation_layers = ['layer1.1.conv1','layer1.2.conv2','layer2.1.conv1','layer3.1.conv1','layer3.2.conv2','fc']
    activations, labels = log_activation_src(model, activation_layers, 
                                             src_data_loader, device)
    
    K = {}
    proj_null = {}

    for layer_name in activation_layers:
        K[layer_name] = RBF_kernel(activations[layer_name], activations[layer_name])
        proj_null[layer_name] = create_nullspace(K[layer_name], labels)

    #calculate class wise projections for activations
    num_classes = 10 # num_classes = 10 for cifar10
    proj_cls = {n:torch.empty(0) for n in activation_layers}
    for layer_name in activation_layers:
        proj_cls[layer_name] = torch.zeros(num_classes, proj_null[layer_name].shape[1])

    projection = {}
    dist = torch.empty(len(labels),0)
    proj = torch.empty(len(labels),0)
    scores_null = {}

    a = torch.arange(num_classes)
    mask = labels.unsqueeze(1) == a.unsqueeze(0)

    for layer_name in activation_layers:
        for c in range(num_classes):
            proj_cls[layer_name][c,:] = torch.mean((K[layer_name].T @ proj_null[layer_name])[labels==c], dim=0, keepdim=False)
        projection[layer_name] = K[layer_name].T @ proj_null[layer_name]

        scores_null[layer_name] = torch.cdist(projection[layer_name], proj_cls[layer_name])
        dist = torch.cat((dist,(scores_null[layer_name]*mask).sum(dim=1).unsqueeze(1)), dim=1)
        proj = torch.cat((proj, projection[layer_name]), dim=1)

    np.save('score_proj_src.npy', proj.numpy())
    np.save('score_dist_src.npy', dist.numpy())
    
    #calculate null space projection scores for samples at tst time
    #corruptions = conf.CORRUPTION.TEST
    corruptions = ['brightness']
    for corruption in corruptions:

        data_loader = get_loader(cfg=cfg,
                              corruption = corruption
                              )
        scores_b_dist, scores_m_dist, scores_b_proj, scores_m_proj = null_space_projection(model, activation_layers, 
                                                   data_loader,activations,
                                                   proj_null, proj_cls, cfg, device)
        
        scores_b_dist, scores_m_dist, scores_b_proj, scores_m_proj = scores_b_dist.numpy(), scores_m_dist.numpy(), scores_b_proj.numpy(), scores_m_proj.numpy()
        np.save('score_benign_dist.npy', scores_b_dist)
        np.save('score_mal_dist.npy', scores_m_dist)
        np.save('score_benign_proj.npy', scores_b_proj)
        np.save('score_mal_proj.npy', scores_m_proj)