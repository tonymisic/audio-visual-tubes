from turtle import pos
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

class TC_Hardway(torch.nn.Module):
    """My implementation of HardWay Loss
    """
    def __init__(self):
        super(TC_Hardway, self).__init__()

    def forward(self, attention, device):
        loss = torch.tensor(0.0).to(device)
        
        for sample in attention:
            print()    
        return loss