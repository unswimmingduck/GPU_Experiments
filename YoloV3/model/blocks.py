import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, input_channel, center_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.conv1  = nn.Conv2d(input_channel, center_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(center_channel)
        self.relu1  = nn.LeakyReLU(0.1)
        
        self.conv2  = nn.Conv2d(center_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(out_channel)
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu2(out)

        return out

class DBL(nn.Sequential):
    def __init__(self, num, input_channels, output_channel, kernel, stride, pad, bias=False):
        module = []
        for i in range(num):
            module.append(nn.Conv2d(input_channels, output_channel, 
                                    kernel_size=kernel[i], stride=stride[i], padding=pad[i], 
                                    bias=bias))
            module.append(nn.BatchNorm2d(output_channel))
            module.append(nn.LeakyReLU(0.1))

            input_channels ^= output_channel
            output_channel = input_channels ^ output_channel
            input_channels ^= output_channel

        return super(DBL, self).__init__(*module)
