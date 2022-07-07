from turtle import pos
import numpy as np, torch, torch.nn as nn, einops
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torchvision.transforms import *
from utils import *
class NPRatio(torch.nn.Module):
    """Negative / Positive Ratio Loss
    """
    def __init__(self):
        super(NPRatio, self).__init__()

    def forward(self, heatmap):
        return torch.abs(torch.diff(torch.sum(heatmap, dim=(2,3)), dim=1)).mean(dim=1).mean(dim=0)

class PropagationLoss(torch.nn.Module):
    """
    """
    def __init__(self):
        super(PropagationLoss, self).__init__()
        
    def forward(self, heatmap):
        return torch.abs(torch.diff(heatmap, dim=1)).mean(dim=(2,3)).mean(dim=1).mean(dim=0)

class FlipLoss(torch.nn.Module):
    """
    """
    def __init__(self):
        super(FlipLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.loss2 = nn.MSELoss()

    def forward(self, heatmap, flipped_heatmap):
        pseudo_labels = RandomHorizontalFlip(p=1)(heatmap)
        loss = self.loss(flipped_heatmap, pseudo_labels)
        return loss