import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_entropy(x):
    softmax = x.softmax(1)
    return -(softmax * torch.log(softmax + 1e-6)).sum(1)

def softmax_entropy_rotta(x, x_ema):
    return - (x_ema.softmax(1) * x.log_softmax(1)).sum(1)

class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = nn.functional.softmax(x/self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax+1e-6)
        b = entropy.mean()

        return b
    
def normalized_renyi_entropy(x, alpha=2):
    return (1/ (1-alpha)) * (F.softmax(x , dim=-1).pow(alpha).sum(-1)).log()