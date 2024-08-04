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


def null_space_projection(model, layers, data_loader, feat_src_1, feat_src_2, 
                          proj_null_1, proj_null_2, proj_cls_k1,
                          proj_cls_k2, cfg, device):

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
    scores_mal = torch.tensor([])
    scores_benign = torch.tensor([])

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
        
        
        feat_test = activations_src['layer1.1.conv1']
        #K_1_test = RBF_kernel(feat_src_1, feat_test)
        K_1_test = RBF_kernel(feat_test,feat_test)
        proj_null_1 = create_nullspace(K_1_test, pred_labels_src)
        
    
        feat_test = activations_src['fc']
        #K_2_test = RBF_kernel(feat_src_2, feat_test)
        K_2_test = RBF_kernel(feat_test, feat_test)
        proj_null_2 = create_nullspace(K_2_test, pred_labels_src)
            
            
        projection_1 = K_1_test.T @ proj_null_1
        projection_2 = K_2_test.T @ proj_null_2

        #class wise null space projection for predicted test data
        num_classes = torch.unique(pred_labels_src).numel() # num_classes = 10 for cifar10
        proj_cls_k1 = torch.zeros(num_classes, proj_null_1.shape[1])
        proj_cls_k2 = torch.zeros(num_classes, proj_null_2.shape[1])
        for c in range(num_classes): 
            proj_cls_k1[c,:] = torch.mean((projection_1)[pred_labels_src==c], dim=0, keepdim=False)
            proj_cls_k2[c,:] = torch.mean((projection_2)[pred_labels_src==c], dim=0, keepdim=False)
     

        scores_null_1 = torch.cdist(projection_1, proj_cls_k1)
        print(f'Null scores {scores_null_1}')
        scores_null_2 = torch.cdist(projection_2, proj_cls_k2)

        # ref_cls_vec_1 = proj_cls_k1[pred_labels_src.tolist(),:]
        # ref_cls_vec_2 = proj_cls_k2[pred_labels_src.tolist(),:]

        # A = projection_1
        # B = ref_cls_vec_1
        # dist1 = ((B * B.log()).sum (dim = 1) - torch.einsum ('ik, jk -> ij', A, B)).diag().unsqueeze(dim=1)
        # print(f'Shape of kl_div {dist1.shape}')
        # A = projection_1
        # B = ref_cls_vec_2
        # dist2 = ((B * B.log()).sum (dim = 1) - torch.einsum ('ik, jk -> ij', A, B)).diag().unsqueeze(dim=1)
        # calculate the ratio of class wise null space distances
        dist1 = torch.zeros(scores_null_1.shape[0])
        dist2 = torch.zeros(scores_null_2.shape[0])

        a = torch.arange(num_classes)
        mask = pred_labels_src.unsqueeze(1) == a.unsqueeze(0)
        dist1 = (scores_null_1*~mask).sum(dim=1)/(scores_null_1*mask).sum(dim=1)
        dist2 = (scores_null_2*~mask).sum(dim=1)/(scores_null_2*mask).sum(dim=1)
        #dist1 = (scores_null_1*mask).sum(dim=1)
        #dist2 = (scores_null_2*mask).sum(dim=1)
        dist_diff = dist2 - dist1
        print(f'dist_diff : {dist_diff}')
        scores_benign = torch.cat((scores_benign, dist_diff[:-mal_num]), dim=0)
        scores_mal = torch.cat((scores_mal, dist_diff[-mal_num:]), dim=0)

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
    return scores_benign, scores_mal


class Activation_Hook():
  def __call__(self, module, input, output):
    module.act = output.detach().clone().flatten(start_dim=1).cpu()
    

        
        
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
    activations, labels = log_activation_src(model, ['layer1.1.conv1','fc'], src_data_loader, device)
    
    feat_src_1 = activations['layer1.1.conv1']
    K_1 = RBF_kernel(feat_src_1, feat_src_1)
    feat_src_2 = activations['fc']
    K_2 = RBF_kernel(feat_src_2, feat_src_2)
    proj_null_1 = create_nullspace(K_1, labels)
    proj_null_2 = create_nullspace(K_2, labels)

    #calculate class wise projections for activations
    num_classes = 10 # num_classes = 10 for cifar10
    proj_cls_k1 = torch.zeros(num_classes, proj_null_1.shape[1])
    proj_cls_k2 = torch.zeros(num_classes, proj_null_2.shape[1])
    for c in range(num_classes): 
        proj_cls_k1[c,:] = torch.mean((K_1.T @ proj_null_1)[labels==c], dim=0, keepdim=False)
        proj_cls_k2[c,:] = torch.mean((K_2.T @ proj_null_2)[labels==c], dim=0, keepdim=False)
    #proj_cls_k1 = nn.functional.softmax(proj_cls_k1, dim=-1)
    #proj_cls_k2 = nn.functional.softmax(proj_cls_k2, dim=-1)
    #calculate null space projection scores for samples at tst time
    #corruptions = conf.CORRUPTION.TEST
    corruptions = ['shot_noise']
    for corruption in corruptions:

        data_loader = get_loader(cfg=cfg,
                              corruption = corruption
                              )
        scores_1, scores_2 = null_space_projection(model, ['layer1.1.conv1','fc'], data_loader, feat_src_1, feat_src_2,
                                                   proj_null_1, proj_null_2, proj_cls_k1, proj_cls_k2, cfg, device)
        
        scores_1, scores_2 = scores_1.numpy(), scores_2.numpy()
        np.save('score_benign_3.npy', scores_1)
        np.save('score_mal_3.npy', scores_2)