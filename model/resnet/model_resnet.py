import torch
import torch.nn as nn
from .resnet import ResNet        
from .formulanet import formulanet50  

class Model1s(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        self.resnet = ResNet(num_classes=num_class)

    def forward(self, x):
        out = self.resnet(x)
        return out

    
class FormulaNet(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        self.model = formulanet50(ratio=8, lateral_multiplier=2, class_num=num_class)

    def forward(self, x1, x2):
        out = self.model(x1, x2)
        return out