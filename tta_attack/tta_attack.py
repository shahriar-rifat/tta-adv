from attack_model.dia import Attack_WB 
import torch
import torch.nn.functional as F

class TTA_ADV(Attack_WB):
    def __init__(self, model, config) -> None:
        super(TTA_ADV, self).__init__()
    
    def forward(self, x_m, x_t):
        x_adv = x_m.detach().clone()
        x_adv.requires_grad = True
        pass

    @staticmethod
    def entropyloss(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(dim=1) * x.log_softmax(dim=1).sum(1))




