import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()

        self.resnet = models.resnet50(pretrained=False, num_classes=num_classes)
        
    def forward(self, v):    
        output = self.resnet(v)

        return output

