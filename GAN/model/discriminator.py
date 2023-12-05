import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.module = nn.Sequential(
                                    nn.Linear(input_size, 512),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(128, 1),
                                    nn.Sigmoid())
    
    def forward(self, img):
        img = img.view(img.size(0), -1)
        result = self.module(img)
        return result