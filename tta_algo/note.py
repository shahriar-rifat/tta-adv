import torch
import torch.nn as nn
from tta_algo.tta_base import TTA_BASE
from utils.bn_layers import InstanceAwareBatchNorm2d
from utils.memory import PBRS
from utils.loss_functions import HLoss
from copy import deepcopy



class NOTE(TTA_BASE): 
    def __init__(self, cfg, model):
        super(NOTE,self).__init__(cfg, model)
        self.cfg = cfg
        self.mem = PBRS(cfg, capacity=cfg.TTA.NOTE.MEMORY_SIZE)
        self.current_instance = 0
        self.update_frequency = cfg.TTA.NOTE.UPDATE_FREQUENCY
        self.device = torch.device("cuda:{:d}".format(cfg.BASE.GPU_ID) if torch.cuda.is_available() else "cpu")

    def forward_and_adapt(self,data_batch, model, optimizer):
        with torch.no_grad():
            model.eval()
            out = model(data_batch)
            predict = torch.softmax(out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
        
        # add samples into PBRS
        for i, data in enumerate(data_batch):
            p_l = pseudo_label[i].item()
            f , c= data_batch[i],p_l
            self.mem.add_instance([f, c])
            self.current_instance += 1
            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)
            return out
        
    def update_model(self, model, optimizer):
        model.train()
        feats,_ = self.mem.get_memory()
        feats = torch.stack(feats)
        feats = feats.to(self.device)
        entropy_loss = HLoss(temp_factor=self.cfg.TTA.NOTE.TEMP)
        pred = model(feats)
        loss = entropy_loss(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       

    def configure_model(self, model):
        convert_iabn(self.cfg, model)
        for param in model.parameters():
            param.requires_grad = False
        for m in model.modules():
            if isinstance(m, InstanceAwareBatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = True
        return model

def convert_iabn(cfg, module):
    module_output = module
    if isinstance(module, nn.BatchNorm1d): 
        module_output = InstanceAwareBatchNorm2d(
            cfg=cfg,
            num_channels=module.num_features,
            k=cfg.TTA.NOTE.IABN_K,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )

        module_output._bn = deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_iabn(cfg,child)
        )
    del module
    return module_output
