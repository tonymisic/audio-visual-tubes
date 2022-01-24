import torch
from torch import nn
import torch.nn.functional as F
from models import base_models
from models import resnet3D

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class FullModel(nn.Module):
    def __init__(self, args):
        super(FullModel, self).__init__()
        self.vidnet = resnet3D.resnet18()
        self.audnet = base_models.resnet18(modal='audio')
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.vidnet.layer4.register_forward_hook(get_activation('layer4'))
        
    def forward(self, video, audio):
        # audio editing
        B = audio.shape[0]
        aud = self.audnet(audio)
        aud = self.avgpool(aud).view(B,-1)
        aud = nn.functional.normalize(aud, dim=1)
        # video editing
        _ = self.vidnet(video)
        vid = activation['layer4']