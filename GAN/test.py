from model.discriminator import Discriminator
from model.generator import Generator

from torchvision import transforms,datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable

import time
import torch
import logging



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(nimble=True):
    mnist = datasets.MNIST(
    root='./datasets/', train=True, download=True, transform=transforms.Compose(
            [transforms.Resize((28)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ), 
    )

    ## build dataloader
    dataloader = DataLoader(
        mnist,
        batch_size=64,
        shuffle=True,
    )

    # init
    model_d = Discriminator((1,28,28))
    model_g = Generator(100, (1,28,28), 28*28)

    criterion = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator_n.parameters(), lr=0.01, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator_n.parameters(), lr=0.01, betas=(0.5, 0.999))

    start_time = time.time()
    if nimble:
        generator_n = torch.cuda.Nimble(model_g)
        discriminator_n = torch.cuda.Nimble(model_d)
        input_d = torch.rand([64, 784]).cuda()
        input_g = torch.rand([64, 100]).cuda()
        generator_n.prepare(input_g, training=True)
        discriminator_n.prepare(input_d, training=True)
        init_time = time.time() - start_time
        logger.info(f"nimble init time: {init_time}")

    for i in range(60):
        train_start = time.time()
        for i, (imgs, _) in enumerate(dataloader):                  
            imgs = imgs.view(imgs.size(0), -1)                          
            real_img = Variable(imgs).cuda()                            
            real_label = Variable(torch.ones(imgs.size(0), 1)).cuda()      
            fake_label = Variable(torch.zeros(imgs.size(0), 1)).cuda()     


            real_out = discriminator_n(real_img)                         
            loss_real_D = criterion(real_out, real_label)               
            real_scores = real_out                                      
            
            z = Variable(torch.randn(imgs.size(0), 100)).cuda()     
            fake_img = generator_n(z).detach() 
            fake_img = fake_img.view(fake_img.size(0), -1)                                     
            fake_out = discriminator_n(fake_img)                                  
            loss_fake_D = criterion(fake_out, fake_label)                       
            fake_scores = fake_out                                              
            
            loss_D = loss_real_D + loss_fake_D                  
            optimizer_D.zero_grad()                             
            loss_D.backward()                                   
            optimizer_D.step()                                  


            z = Variable(torch.randn(imgs.size(0), 100)).cuda()      
            fake_img = generator_n(z)                                             
            fake_img = fake_img.view(fake_img.size(0), -1)                          
            output = discriminator_n(fake_img)                                    
            
            loss_G = criterion(output, real_label)                              
            optimizer_G.zero_grad()                                             
            loss_G.backward()                                                   
            optimizer_G.step()                                                  
            
        endtiem_n = time.time()    
        break

    if nimble:
        logger.info(f"nimble train a epoch using time: {endtiem_n - train_start}")
    else:
        logger.info(f"model train a epoch using time: {endtiem_n - train_start}")

if __name__ == "__main__":
    main(nimble=True)