from turtle import pos
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

class NPRatio(torch.nn.Module):
    """My implementation of HardWay Loss
    """
    def __init__(self):
        super(NPRatio, self).__init__()

    def forward(self, attention, threshold, device):
        loss = torch.tensor([attention.size(0)]).to(device)
        attention[attention > threshold] = 1
        attention[attention < 1] = 0
        for i in range(attention.size(0) - 1):
            ratio_current = torch.divide(torch.sum(attention[i]), torch.float(attention.size(1) * attention.size(2)))
            ratio_next = torch.divide(torch.sum(attention[i + 1]), torch.float(attention.size(1) * attention.size(2)))
            loss[i] = torch.abs(ratio_current - ratio_next)
        return torch.sum(loss)