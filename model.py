from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from models import base_models
from models import resnet3D

activation, selected_layer = {}, 'layer3'
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class FullModel(nn.Module):
    def __init__(self, args):
        super(FullModel, self).__init__()
        self.vidnet = resnet3D.generate_model(model_depth=18, no_max_pool=False)
        self.audnet = base_models.resnet18(modal='audio')
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.vidnet.layer4.register_forward_hook(get_activation(selected_layer))
        self.attention = AttentionModel(512)
        
    def forward(self, audio, video):
        # audio editing
        B = audio.shape[0]
        aud = self.audnet(audio)
        aud = self.avgpool(aud).view(B,-1)
        aud = nn.functional.normalize(aud, dim=1) # batch_size, 512
        # video editing
        _ = self.vidnet(video)
        vid = activation[selected_layer]  # batch_size, 512, 88, 7, 7 
        attn_output = self.attention(aud, vid.permute([0, 2, 3, 4, 1]))
        return attn_output

class AttentionModel(nn.Module):
    def __init__(self, latent):
        super(AttentionModel, self).__init__()
        self.key_linear = nn.Linear(latent, latent)
        self.query_linear = nn.Linear(latent, latent)
        self.value_linear = nn.Linear(latent, latent)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, audio_features, video_features):
        key = self.key_linear(video_features)
        query = self.query_linear(audio_features)
        weights = torch.einsum("abcde, ae -> abcd", key, query)
        value = self.value_linear(video_features)
        attention = torch.einsum("abcde, abcd -> abcd", value, self.softmax(weights))

        return attention