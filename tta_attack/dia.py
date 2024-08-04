import math
import torch
import torch.nn as nn
from config.conf import cfg 
from torchvision import transforms

class DIA:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2470, 0.2435, 0.2616]
        )


    def generate_attack(self, sur_model, x, y):
        x = self.denorm(x, mean =[0.4914, 0.4822, 0.4465], std =[0.2470, 0.2435, 0.2616])
        x,y = x.to(self.device), y.to(self.device)
        num_iter = self.cfg.DIA.STEPS
        epsilon = self.cfg.DIA.EPS
        alpha = self.cfg.DIA.ALPHA
        mal_num = math.ceil(self.cfg.DIA.MAL_PORTION * self.cfg.DATA.BATCH_SIZE)
        fixed = torch.zeros_like(x.clone()[:-mal_num], requires_grad=False)
        adv = (torch.zeros_like(x.clone()[-mal_num:])- x[-mal_num:] + 127.5/255).requires_grad_(True)
        adv_pad = torch.cat((fixed,adv), 0)
        adv_pad = adv_pad.to(self.device)

        for t in range(num_iter):
            x_adv = x + adv_pad
            out = sur_model(x_adv)
            if cfg.DIA.PSEUDO:
                loss = nn.CrossEntropyLoss()(out[:-mal_num], out[:-mal_num])
            else:
                loss = nn.CrossEntropyLoss()(out[:-mal_num], y[:-mal_num])
            loss.backward()

            adv.data = (adv + alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
            adv.data = (adv.data +x[-mal_num:]).clamp(0,1) -(x[-mal_num:])
            adv_pad.data = torch.cat((fixed, adv), 0) 
            adv.grad.zero_()

        x_adv = x + adv_pad
        x_adv = self.transform(x_adv)
        x_adv = x_adv.detach()
        return x_adv
    
    def denorm(self, batch, mean, std):

        if isinstance(mean, list):
            mean = torch.tensor(mean).to(batch.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(batch.device)
        return torch.clamp(batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1), 0, 1)

# class TTA_POISON:

#     def __init__(self,cfg):
#         self.cfg = cfg
    
#     def generate_attack(self, sur_model, x):
        


    
    



