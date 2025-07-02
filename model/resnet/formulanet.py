import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet50', 'resnet101', 'resnet152', 'resnet200']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FormulaNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=3, dropout=0.5, ratio=8, lateral_multiplier=2):
        super(FormulaNet, self).__init__()
        self.ratio = ratio
        self.lateral_multiplier = lateral_multiplier

        initial_auxiliary_inplanes = 64 // ratio

        self.auxiliary_conv1 = nn.Conv2d(3, initial_auxiliary_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.auxiliary_bn1 = nn.BatchNorm2d(initial_auxiliary_inplanes)
        self.auxiliary_relu = nn.ReLU(inplace=True)
        self.auxiliary_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Precompute the number of output channels at each stage
        auxiliary_stage_channels = [
            initial_auxiliary_inplanes,          # pool1
            (64 // ratio) * block.expansion,     # res2
            (128 // ratio) * block.expansion,    # res3
            (256 // ratio) * block.expansion,    # res4
            (512 // ratio) * block.expansion     # res5
        ]

        # Lateral connection
        self.lateral_p1 = nn.Conv2d(auxiliary_stage_channels[0], auxiliary_stage_channels[0] * lateral_multiplier, kernel_size=1, bias=False)
        self.lateral_res2 = nn.Conv2d(auxiliary_stage_channels[1], auxiliary_stage_channels[1] * lateral_multiplier, kernel_size=1, bias=False)
        self.lateral_res3 = nn.Conv2d(auxiliary_stage_channels[2], auxiliary_stage_channels[2] * lateral_multiplier, kernel_size=1, bias=False)
        self.lateral_res4 = nn.Conv2d(auxiliary_stage_channels[3], auxiliary_stage_channels[3] * lateral_multiplier, kernel_size=1, bias=False)

        # Auxiliary pathway
        self.auxiliary_inplanes = initial_auxiliary_inplanes
        self.auxiliary_res2 = self._make_layer_auxiliary(block, 64 // ratio, layers[0], head_conv=3)
        self.auxiliary_res3 = self._make_layer_auxiliary(block, 128 // ratio, layers[1], stride=2, head_conv=3)
        self.auxiliary_res4 = self._make_layer_auxiliary(block, 256 // ratio, layers[2], stride=2, head_conv=3)
        self.auxiliary_res5 = self._make_layer_auxiliary(block, 512 // ratio, layers[3], stride=2, head_conv=3)

        # Main pathway
        self.main_inplanes = 64
        self.main_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.main_bn1 = nn.BatchNorm2d(64)
        self.main_relu = nn.ReLU(inplace=True)
        self.main_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.main_res2 = self._make_layer_main(block, 64, layers[0], inplanes=64 + auxiliary_stage_channels[0] * lateral_multiplier, head_conv=1)
        self.main_res3 = self._make_layer_main(block, 128, layers[1], inplanes=256 + auxiliary_stage_channels[1] * lateral_multiplier, stride=2, head_conv=1)
        self.main_res4 = self._make_layer_main(block, 256, layers[2], inplanes=512 + auxiliary_stage_channels[2] * lateral_multiplier, stride=2, head_conv=3)
        self.main_res5 = self._make_layer_main(block, 512, layers[3], inplanes=1024 + auxiliary_stage_channels[3] * lateral_multiplier, stride=2, head_conv=3)

        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(auxiliary_stage_channels[4] + 2048, class_num, bias=False)

    def mainPath(self, input_main, lateral):
        x = self.main_conv1(input_main)
        x = self.main_bn1(x)
        x = self.main_relu(x)
        x = self.main_maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)
        x = self.main_res2(x)
        x = torch.cat([x, lateral[1]], dim=1)
        x = self.main_res3(x)
        x = torch.cat([x, lateral[2]], dim=1)
        x = self.main_res4(x)
        x = torch.cat([x, lateral[3]], dim=1)
        x = self.main_res5(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def auxiliaryPath(self, input_auxiliary):
        lateral = []
        x = self.auxiliary_conv1(input_auxiliary)
        x = self.auxiliary_bn1(x)
        x = self.auxiliary_relu(x)
        pool1 = self.auxiliary_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)

        res2 = self.auxiliary_res2(pool1)
        lateral_res2 = self.lateral_res2(res2)
        lateral.append(lateral_res2)

        res3 = self.auxiliary_res3(res2)
        lateral_res3 = self.lateral_res3(res3)
        lateral.append(lateral_res3)

        res4 = self.auxiliary_res4(res3)
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.auxiliary_res5(res4)
        x = nn.AdaptiveAvgPool2d(1)(res5)
        x = x.view(-1, x.size(1))
        return x, lateral

    def _make_layer_auxiliary(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.auxiliary_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.auxiliary_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.auxiliary_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.auxiliary_inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.auxiliary_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_main(self, block, planes, blocks, inplanes, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, head_conv=head_conv))
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def forward(self, input_main, input_auxiliary):
        auxiliary, lateral = self.auxiliaryPath(input_auxiliary)
        main_output = self.mainPath(input_main, lateral)
        x = torch.cat([main_output, auxiliary], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

def formulanet50(ratio=8, **kwargs):
    model = FormulaNet(Bottleneck, [3, 4, 6, 3], ratio=ratio, **kwargs)
    return model

def formula101(ratio=8, **kwargs):
    model = FormulaNet(Bottleneck, [3, 4, 23, 3], ratio=ratio, **kwargs)
    return model

def formula152(ratio=8, **kwargs):
    model = FormulaNet(Bottleneck, [3, 8, 36, 3], ratio=ratio, **kwargs)
    return model

def formula200(ratio=8, **kwargs):
    model = FormulaNet(Bottleneck, [3, 24, 36, 3], ratio=ratio, **kwargs)
    return model

if __name__ == "__main__":
    num_classes = 3
    ori_tensor = torch.randn(16, 3, 540, 320).cuda()
    fci_tensor = torch.randn(16, 3, 540, 320).cuda()
    model = formulanet50(class_num=num_classes, ratio=8, lateral_multiplier=2).cuda()
    output = model(ori_tensor, fci_tensor)
    print(output.size())
