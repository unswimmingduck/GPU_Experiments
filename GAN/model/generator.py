import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, img_area):
        super(Generator, self).__init__()
        ## 模型中间块儿
        def block(in_feat, out_feat, normalize=True):           ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]             ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))    ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))      ## 非线性激活函数
            return layers
        ## prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),       ## 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),                                   ## 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),                                   ## 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),                                  ## 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, img_area),                          ## 线性变化将输入映射 1024 to 784
            nn.Tanh()                                           ## 将(784)的数据每一个都映射到[-1, 1]之间
        )
        self.img_shape = img_shape
    ## view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z):                                       ## 输入的是(64， 100)的噪声数据
        imgs = self.model(z)                                     ## 噪声数据通过生成器模型
        imgs = imgs.view(imgs.size(0), *self.img_shape)                 ## reshape成(64, 1, 28, 28)
        return imgs      
