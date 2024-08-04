import torch
import torch.nn as nn
import torch.optim as optim


class TTA_BASE(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = self.configure_model(model)
        params, param_names = self.collect_params(self.model)
        if len(param_names) != 0:
            self.optimizer = self.setup_optimizer(params,cfg)
        self.steps = self.cfg.OPTIM.STEPS
    

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs
    
    def forward_and_adapt(self, *args):
        raise NotImplementedError("implement forward_and_adapt by yourself!")
    
    
    def configure_model(self, model):
        raise NotImplementedError("implement forward_and_adapt by yourself!")
    

    @staticmethod
    def collect_params(model):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names
    
    @staticmethod
    def setup_optimizer(params,cfg):

        lr_adapt = cfg.OPTIM.LR 
        if cfg.OPTIM.METHOD == 'Adam':
            return optim.Adam(params,
                        lr=lr_adapt,
                        betas=(cfg.OPTIM.BETA, 0.999),
                        weight_decay=cfg.OPTIM.WD)
        elif cfg.OPTIM.METHOD == 'SGD':
            return optim.SGD(params,
                    lr=lr_adapt,
                    momentum=cfg.OPTIM.MOMENTUM,
                    dampening=cfg.OPTIM.DAMPENING,
                    weight_decay=cfg.OPTIM.WD,
                    nesterov=cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError        


