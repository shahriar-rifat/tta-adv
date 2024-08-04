import torch
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
from utils.loss_functions import *
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

def form_column_space(model, data_loader, cfg):
    pass


def log_entropy(model, data_loader, cfg,device):

    model = model.to(device)
    src_model = deepcopy(model)
    src_model = src_model.to(device)
    src_model = src_model.eval()
    victim_model = deepcopy(model)
    victim_model = victim_model.to(device)
    tta_adapter_class = build_tta_adapter(cfg)
    tta_adapter = tta_adapter_class(cfg,model)
    tta_adapter_victim = tta_adapter_class(cfg, victim_model)

    if cfg.BASE.ATTACK == "dia":
        attack = DIA(cfg=cfg)
    elif cfg.BASE.ATTACK == "u_dia":
        attack = U_DIA(cfg=cfg, layers=['avgpool'])
    else:
        raise NotImplementedError("Specify an attack Name that is implemeted")
    
    mal_num = math.ceil(cfg.DATA.BATCH_SIZE * cfg.DIA.MAL_PORTION)

    entropy_mal = torch.tensor([])
    entropy_benign = torch.tensor([])

    # output_kl_div_benign = torch.tensor([])
    # output_kl_div_mal = torch.tensor([])

    # acc_normal = AverageMeter()
    # acc_attacked = AverageMeter()
    bar = tqdm(data_loader)
    for n_batch, (inputs,labels) in enumerate(bar):
        # if len(labels)!= cfg.DATA.BATCH_SIZE:
        #     continue

        # inputs, labels = data['data'], data['label']
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        
        outputs_clean = tta_adapter.forward(inputs).detach().cpu()

        x_adv = attack.generate_attack(sur_model=tta_adapter.model,
                                       x = inputs, y=labels)
        outputs_clean = tta_adapter.forward(inputs).detach().cpu()
        outputs_victim = tta_adapter_victim.forward(x_adv).detach().cpu()
        outputs_src = src_model(x_adv).detach().cpu()
        # print(outputs_src.size(), outputs_src.size())
        # output_kl_div = nn.functional.kl_div(nn.functional.log_softmax(outputs_src, dim=-1), nn.functional.log_softmax(outputs_mal, dim=-1), log_target=True, dim=-1)
        # B = nn.functional.softmax(outputs_src, dim=-1)
        # A = nn.functional.softmax(outputs_mal, dim=-1)
        # output_kl_div = ((B * B.log()).sum (dim = 1) - torch.einsum ('ik, jk -> ij', A, B)).diag().unsqueeze(dim=1)
        #output_kl_div = torch.abs(outputs_mal - outputs_src).sum(1)

        #code for calculating entropy difference
        src_entropy = softmax_entropy(outputs_src)
        victim_entropy = softmax_entropy(outputs_victim)
        output_entropy = victim_entropy - src_entropy

        # outputs_mal = (outputs_mal - outputs_mal.min(dim=-1)[0].view(-1,1))/(outputs_mal.max(dim=-1)[0].view(-1,1)-outputs_mal.min(dim=-1)[0].view(-1,1))
        # output_entropy = softmax_entropy(outputs_mal)
        # output_entropy = normalized_renyi_entropy(outputs_mal, alpha=2)
        
        entropy_benign = torch.cat((entropy_benign, output_entropy[:-mal_num]), dim=0)
        entropy_mal = torch.cat((entropy_mal, output_entropy[-mal_num:]), dim=0)

        # output_kl_div_benign = torch.cat((entropy_benign, output_kl_div[:-mal_num]), dim=0)
        # output_kl_div_mal = torch.cat((entropy_mal, output_kl_div[-mal_num:]), dim=0)

        # output_kl_div_benign = torch.cat((output_kl_div_benign, output_kl_div[:-mal_num]), dim=0)
        # output_kl_div_mal = torch.cat((output_kl_div_mal, output_kl_div[-mal_num:]), dim=0)


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

    return entropy_benign.numpy(), entropy_mal.numpy()
    # return output_kl_div_benign.numpy(), output_kl_div_mal.numpy()

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
    
    set_deterministic(seed=15, cudnn=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-bs','--batch_size',type=int, help="Specify Batch Size",default=64)
    parser.add_argument('--gpu_id',type=int, help="Specify GPU ID")
    parser.add_argument('--tta',type=str, help="Specify the TTA algorithm to be used")
    parser.add_argument('--mal_num',type=int, help="Specify the number of malicious sample in the batch")
    parser.add_argument('--pseudo',type=bool, help="Specify whether to use pseudo labels or not")
    parser.add_argument('--loss', type=str,help="Specify the loss to be used by u_dia" )
    parser.add_argument('--attack', type=str, help="Specify the attack algorithm" )
    args = parser.parse_args()

    conf = update_configs(cfg,args)
    device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
    results_df = pd.DataFrame(columns=["Corruption","Adaptation Accuracy","Normal Accuracy"])
    
    corruptions = conf.CORRUPTION.TEST
    corruptions = ['fog']
    for corruption in corruptions:

        data_loader = get_loader(cfg=cfg,
                              corruption = corruption
                              )
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                        "cifar10_resnet20", pretrained=True)
        entropy_benign, entropy_mal = log_entropy(model, data_loader, cfg, device)
        np.save('entropy_benign_diff_3.npy', entropy_benign)
        np.save('entropy_mal_diff_3.npy', entropy_mal)