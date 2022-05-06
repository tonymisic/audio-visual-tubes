from turtle import pos
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

class NPRatio(torch.nn.Module):
    """Negative / Positive Ratio Loss
    """
    def __init__(self, img_size=None):
        super(NPRatio, self).__init__()
        assert img_size != None
        self.img_size = img_size
    def forward(self, attention, threshold, device):
        attention[attention > threshold] = 1
        attention[attention < 1] = 0
        ratios = torch.divide(torch.sum(attention, dim=(2,3)), self.img_size)
        loss = torch.abs(torch.diff(ratios, dim=1)).to(device)
        return torch.divide(torch.sum(loss), loss.size(0) * loss.size(1))

class FlowLoss(torch.nn.Module):
    """Flow Loss
    """
    def __init__(self, img_size=None):
        super(FlowLoss, self).__init__()
        assert img_size != None
        self.img_size = img_size
        
    def forward(self, attention, threshold, device):
        loss = torch.tensor([attention.size(0)]).to(device)
        attention[attention > threshold] = 1
        attention[attention < 1] = 0
        ratios = torch.divide(torch.sum(attention, dim=1), self.img_size)
        loss = torch.abs(torch.diff(ratios, dim=1))
        return torch.sum(loss)