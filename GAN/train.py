import argparse

import torch

from torchvision import transforms,datasets
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

from model.discriminator import Discriminator
from model.generator import Generator

import os

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
    
    return parser.parse_args()


def main():
    args = get_args()

    img_shape = (args.channels, args.img_size, args.img_size)
    img_area= args.img_size * args.img_size

    cuda = True if torch.cuda.is_available() else False

    #  recording the training process
    if not os.path.exists("./images/gan/"):
        os.makedirs("./images/gan/", exist_ok=True)    
    #  checkpoint file  
    if not os.path.exists("./checkpoint/"):    
        os.makedirs("./checkpoint/", exist_ok=True)    
    #  dataset file    
    if not os.path.exists("./datasets/mnist"): 
        os.makedirs("./datasets/mnist", exist_ok=True) 


    ## mnist data download
    mnist = datasets.MNIST(
        root='./datasets/', train=True, download=True, transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ), 
    )

    ## build dataloader
    dataloader = DataLoader(
        mnist,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # init
    model_d = Discriminator(img_area)
    model_g = Generator(args.latent_dim, img_shape, img_area)

    if cuda:
        model_d.cuda()
        model_g.cuda()
    
    # generate loss function
    criterion = torch.nn.BCELoss()

    # built optimizer
    optimizer_G = torch.optim.Adam(model_g.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(model_d.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # start training
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.view(imgs.size(0), -1)                          
            real_img = Variable(imgs).cuda()                            
            real_label = Variable(torch.ones(imgs.size(0), 1)).cuda()      
            fake_label = Variable(torch.zeros(imgs.size(0), 1)).cuda()             

            # Train deiscriminator
            ## train the abillity of discriminating real img
            real_out = model_d(real_img)                         
            loss_real_D = criterion(real_out, real_label)               
            real_scores = real_out  

            ## train the abillity of discriminating fake img
            ### generate fake img
            z = Variable(torch.randn(imgs.size(0), args.latent_dim)).cuda()      
            fake_img = model_g(z)
            ### discriminate fake img
            fake_out = model_d(fake_img)                                 
            loss_fake_D = criterion(fake_out, fake_label)                       
            fake_scores = fake_out    
                                                     
            ## backward and optimize the weight of model
            loss_D = loss_real_D + loss_fake_D                  
            optimizer_D.zero_grad()                             
            loss_D.backward()                                   
            optimizer_D.step()          


            # Train the generator    
            ## train the abillity of generating img through discriminator                    
            z = Variable(torch.randn(imgs.size(0), args.latent_dim)).cuda()      
            fake_img = model_g(z)                                             
            output = model_d(fake_img) 

            ## backward and optimize the weight
            loss_G = criterion(output, real_label)                              
            optimizer_G.zero_grad()                                             
            loss_G.backward()                                                   
            optimizer_G.step()   

        logger.info(f"epoch: [{epoch+1}/{args.n_epochs}], Generator loss: {loss_G}, Discriminator loss: {loss_D}")

        if (i + 1) % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D real: %f] [D fake: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), real_scores.data.mean(), fake_scores.data.mean())
            )
        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            save_image(fake_img.data[:25], "./images/gan/%d.png" % batches_done, nrow=5, normalize=True)

    torch.save(model_g.state_dict(), './save/gan/generator.pth')
    torch.save(model_d.state_dict(), './save/gan/discriminator.pth')

if __name__ == "__main__":
    main()