import torch
import math
import torch.nn as nn
from config.conf import cfg 
from torchvision import transforms
from utils.loss_functions import *


class U_DIA:

    def __init__(self, cfg, layers):
        self.cfg = cfg
        self.layers=layers

        self._features = {layer_id:torch.empty(0) for layer_id in layers}
        self.handles = {layer_id:[] for layer_id in layers}
        self.device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2470, 0.2435, 0.2616]
        )


    def generate_attack(self, sur_model, x, y):
        x = self.denorm(x, mean =[0.4914, 0.4822, 0.4465], std =[0.2470, 0.2435, 0.2616])
        self.model = sur_model
        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.handles[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))

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
            #predict = torch.softmax(out, dim=1)
            entropy = softmax_entropy(out[-mal_num:]).mean(0)
            # loss = nn.CrossEntropyLoss()(out[:-mal_num], y[:-mal_num])
            feat_loss = self.featureloss()
            #print(f'entropy loss --> {entropy}')
            #print(f'feature loss {feat_loss}')
            loss = feat_loss + entropy
            loss.backward()

            #adv.data = (adv + alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
            adv.data = (adv + alpha*adv.grad.detach().sign()).clamp(-epsilon,epsilon)
            adv.data = (adv.data +x[-mal_num:]).clamp(0,1) -(x[-mal_num:])
            adv_pad.data = torch.cat((fixed, adv), 0) 
            adv.grad.zero_()

        x_adv = x + adv_pad
        x_adv = self.transform(x_adv)
        x_adv = x_adv.detach()
        return x_adv
        
    def featureloss(self):
        mal_num = math.ceil(self.cfg.DIA.MAL_PORTION * self.cfg.DATA.BATCH_SIZE)
        temp = self._features[self.layers[0]][:-mal_num].squeeze()
        mu = torch.mean(temp,0).squeeze()
        mu = mu.unsqueeze(dim=0)
        #temp2 = temp-mu
        #distance = torch.sqrt(torch.tensordot(temp2,temp2,dims=([1],[1])).diag())
        loss = nn.MSELoss()
        #loss = nn.L1Loss()
        #loss = nn.CosineSimilarity(dim=1, eps=1e-6)
        #distance = loss(temp, mu)
        distance = loss(temp, mu).mean()
        return distance

    def save_outputs_hook(self, layer_id:str):
        def fn(module, input , output):
            #self._features[layer_id] = output.detach()
            self._features[layer_id] = output.detach().clone()
        return fn

    def remove_handles(self):
        for layer_id in self.layers:
            self.handles[layer_id].remove()

    def denorm(self, batch, mean, std):

        if isinstance(mean, list):
            mean = torch.tensor(mean).to(batch.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(batch.device)
        return torch.clamp(batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1), 0, 1)


    
    



