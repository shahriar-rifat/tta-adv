from tta_algo.tta_base import TTA_BASE
import torch
import torch.nn as nn
#from config.conf import cfg
from utils.loss_functions import *

#device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")

class TENT(TTA_BASE):

    def __init__(self, cfg, model):
        super(TENT, self).__init__(cfg, model)


    def forward_and_adapt(self, data_batch, model, optimizer):

        outputs = model(data_batch)
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs


    def configure_model(self, model):
        
        for param in model.parameters():
            param.requires_grad = False
        
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
        return model
        
        
