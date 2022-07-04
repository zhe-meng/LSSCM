'''
Meng, Z.; Jiao, L.; Liang, M.; Zhao, F. A lightweight spectral-spatial convolution module for hyperspectral image classification.
IEEE Geosci. Remote Sens. Lett. 2021. doi:10.1109/LGRS.2021.3069202.
'''
import torch.nn as nn
import math
import torch


class LSSCM(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(LSSCM, self).__init__()
        '''
        Let inp = 64, oup = 128
        '''
        self.oup = oup  # 128
        init_channels = math.ceil(oup / ratio)  # 64
        new_channels = init_channels * (ratio - 1)  # 64
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = LSSCM(in_planes, planes, relu=True)
        self.conv2 = LSSCM(planes, planes, relu=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class LSSCM_ResNet(nn.Module):
    def __init__(self, num_classes, channels):
        super(LSSCM_ResNet, self).__init__()
        # the proposed LSSCM-based residual network.
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.in_planes = 16
        self.layer1 = self._make_layer(ResBlock, 16, num_blocks=1, stride=1)
        self.layer2 = self._make_layer(ResBlock, 32, num_blocks=1, stride=1)
        self.layer3 = self._make_layer(ResBlock, 64, num_blocks=1, stride=1)
        self.relu = nn.ReLU(True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        output = self.avgpool(out3)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output


if __name__ == '__main__':

    model = LSSCM_ResNet(num_classes=16, channels=200)
    model.eval()
    print(model)
    input = torch.randn(100, 200, 11, 11)
    y = model(input)
    print(y.size())