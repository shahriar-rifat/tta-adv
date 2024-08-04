import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import torch.functional as F

class MomentumBN(nn.Module):
    def __init__(self, bn_layer, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_mean is not None and bn_layer.running_var is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

        self.current_mu = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError

# class RobustBN1d(MomentumBN):
#     def forward(self, x):
#         if self.training:
#             b_var, b_mean = torch.var_mean(x, dim=[0,2], unbiased=False, keepdim=False)  # (C,)
#             mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
#             var = (1 - self.momentum) * self.source_var + self.momentum * b_var
#             self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
#             mean, var = mean.view(1, -1, 1), var.view(1, -1, 1)
#             #print(mean.shape)
#         else:
#             mean, var = self.source_mean.view(1, -1, 1), self.source_var.view(1, -1, 1)
#             #print(mean.shape)
#         x = (x - mean) / torch.sqrt(var + self.eps)
#         weight = self.weight.view(1, -1, 1)
#         bias = self.bias.view(1, -1, 1)

#         return x * weight + bias

class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias

# class InstanceAwareBatchNorm1d(nn.Module):
#     def __init__(self, cfg, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
#         super(InstanceAwareBatchNorm1d, self).__init__()
#         self.cfg = cfg
#         self.num_channels = num_channels
#         self.k = k
#         self.eps = eps
#         self.affine = affine
#         self._bn = nn.BatchNorm1d(num_channels, eps=eps,
#                                   momentum=momentum, affine=affine)

#     def _softshrink(self, x, lbd):
#         x_p = torch.nn.functional.relu(x - lbd, inplace=True)
#         x_n = torch.nn.functional.relu(-(x + lbd), inplace=True)
#         y = x_p - x_n
#         return y

#     def forward(self, x):
#         b, c, l = x.size()
#         sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
#         if self.training:
#             _ = self._bn(x)
#             sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
#         else:
#             if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
#                 sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
#             else:
#                 mu_b = self._bn.running_mean.view(1, c, 1)
#                 sigma2_b = self._bn.running_var.view(1, c, 1)

#         if l <=self.cfg.TTA.NOTE.SKIP_THRESH:
#             mu_adj = mu_b
#             sigma2_adj = sigma2_b
        
#         else:
#             s_mu = torch.sqrt((sigma2_b + self.eps) / l) ##
#             s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

#             mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
#             sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
#             sigma2_adj = torch.nn.functional.relu(sigma2_adj)


#         x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

#         if self.affine:
#             weight = self._bn.weight.view(c, 1)
#             bias = self._bn.bias.view(c, 1)
#             x_n = x_n * weight + bias

#         return x_n

class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k=k
        self.affine = affine
        self._bn = nn.BatchNorm2d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) #IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)


        if h*w <=conf.args.skip_thres:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)

            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)

            sigma2_adj = F.relu(sigma2_adj) #non negative

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n
