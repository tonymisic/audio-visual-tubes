from turtle import pos
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F

class HardWayLoss(torch.nn.Module):
    """My implementation of HardWay Loss
    """
    def __init__(self):
        super(HardWayLoss, self).__init__()

    def forward(self, positives, negatives, device):
        assert positives.size(0) == negatives.size(0)
        loss = torch.tensor(0.0).to(device)
        for i in range(positives.size(0)):
            loss += torch.log(torch.div( torch.exp(positives[i]), torch.exp(positives[i])+torch.exp(negatives[i]) ))
        return -torch.div(loss, positives.size(0))