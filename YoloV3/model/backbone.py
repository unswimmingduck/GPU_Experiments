import math
from collections import OrderedDict

import torch.nn as nn
import torch

from blocks import ResidualBlock, DBL

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#



class ResNet(nn.Sequential):
    def __init__(self, input_channel, output_channel, id) -> None:
        module = []
        module.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=False))
        module.append(nn.BatchNorm2d(output_channel))
        module.append(nn.LeakyReLU(0.1))
        for i in range(id):
            module.append(ResidualBlock(output_channel, input_channel, output_channel))

        return super().__init__(*module)
        


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        
        self.dbl = DBL(1, 3, 32, [3], [1], [1], False)

        # 416,416,32 -> 208,208,64
        self.layer1 = ResNet(32, 64, 1)
        # 208,208,64 -> 104,104,128
        self.layer2 = ResNet(64, 128, 2)
        # 104,104,128 -> 52,52,256
        self.layer3 = ResNet(128, 256, 8)
        # 52,52,256 -> 26,26,512
        self.layer4 = ResNet(256, 512, 8)
        # 26,26,512 -> 13,13,1024
        self.layer5 = ResNet(512, 1024, 4)


        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dbl(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5



