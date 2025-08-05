import os, sys
import numpy as np
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data_pulling import Latent_dimension_dataset
import config as config
from GLOBAL_FUNCTIONS import save_on_report

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

def train_discriminator(imgs,generator,discriminator,optimizer_D,z):
# Configure input
    real_imgs = Variable(imgs.type(torch.float32))
#  Train Discriminator
    optimizer_D.zero_grad()
# Generate a batch of images
    fake_imgs = generator(z).detach()
# Adversarial loss
    loss_D = -torch.mean(discriminator(real_imgs))# + torch.mean(discriminator(fake_imgs))
    loss_D += torch.mean(discriminator(fake_imgs))
    loss_D.backward()
    optimizer_D.step()
# Clip weights of discriminator
    for p in discriminator.parameters():
        p.data.clamp_(-config.CLIP_VALUE, config.CLIP_VALUE)
    return loss_D.item()

def train_generator(generator, discriminator,optimizer_G,z):
    optimizer_G.zero_grad()
# Generate a batch of images
    gen_imgs = generator(z)

    loss_G = -torch.mean(discriminator(gen_imgs))
    loss_G.backward()
    optimizer_G.step()
    return loss_G.item()

def train_wgans(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Configure data loader
    for LATENT_DIM in config.LATENT_DIMENSIONS:
        start_time = time.time()
        Save_dir = str(LATENT_DIM)

        train_dataset = Latent_dimension_dataset(str(LATENT_DIM)+"/encoded_train")
        test_dataset  = Latent_dimension_dataset(str(LATENT_DIM)+"/encoded_test")
        train_dataset = [data.float().to(device) for data in train_dataset]
        test_dataset = [data.float().to(device) for data in test_dataset]
        
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.Batch_size,
            shuffle=True,
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.Batch_size,
            shuffle=True,
        )
    # Initialize generator and discriminator
        generator = Generator(LATENT_DIM, config.INNER_LATENT_DIM)
        discriminator = Discriminator(LATENT_DIM)
        
        if cuda:
            generator.cuda()
            discriminator.cuda()
        
    # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.Learning_rate_Wgan)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.Learning_rate_Wgan)
#/home/jcolombini/Purpose/Labeler/Generative_architecture/Models/wgan.py:127: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:78.)
# z = Variable(Tensor(np.random.normal(0, 1, (config.BATCH_SIZE, config.INNER_LATENT_DIM))))
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # #
    #  Training
    # #
        gen_loss = []
        dis_loss = []
        batches_done = 0
        for epoch in range(args.Epochs_Wgan):  
            for i, imgs in enumerate(dataloader):
            # Sample noise as generator input
                z = Variable(torch.tensor( np.random.normal(0, 1, (config.BATCH_SIZE, config.INNER_LATENT_DIM)), dtype = torch.float32, device='cuda' ))

                loss_D = train_discriminator(imgs,generator,discriminator, optimizer_D, z)
                if i==0 and epoch%10==0:
                    dis_loss.append(loss_D)

            # Train the generator every n_critic iterations
                if i % config.N_CRITIC == 0:
                    loss_G = train_generator(generator, discriminator,optimizer_G,z)
                    if i==0 and epoch%10==0:
                        gen_loss.append(loss_G)

                    if epoch%20 ==0:
                        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, args.Epochs_Wgan, batches_done % len(dataloader), len(dataloader), loss_D, loss_G))
                batches_done += 1
        torch.save(generator.state_dict(), os.path.join(Save_dir, config.wgan_model_file))
        save_on_report(os.path.join(Save_dir, "report.json"), "wgan_generatorloss", gen_loss)
        save_on_report(os.path.join(Save_dir, "report.json"), "wgan_criticloss", dis_loss)
        save_on_report(os.path.join(Save_dir, "report.json"), "wgan_traintime", time.time()-start_time)

def end_to_end_Wgan(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_DIM = 32*(32+args.Num_classes)
    train_dataset = torch.stack([X.view(START_DIM).float().to(device) for X in torch.load('trainset.pt', weights_only=False)])
    test_dataset = torch.stack([X.view(START_DIM).float().to(device) for X in torch.load('testset.pt', weights_only=False)])
    if not os.path.exists("ETE"):
        os.mkdir("ETE")
    os.chdir("ETE")
    start_time = time.time()
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.Batch_size,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.Batch_size,
        shuffle=True,
    )
# Initialize generator and discriminator
    generator = Generator(START_DIM, config.INNER_LATENT_DIM)
    discriminator = Discriminator(START_DIM)
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.LR_ete_Wgan)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.LR_ete_Wgan)
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# #
#  Training
# #
    gen_loss = []
    dis_loss = []
    batches_done = 0
    for epoch in range(args.Epochs_ete_Wgan):  
        for i, imgs in enumerate(dataloader):
        # Sample noise as generator input
            z = Variable(torch.tensor( np.random.normal(0, 1, (config.BATCH_SIZE, config.INNER_LATENT_DIM)), dtype = torch.float32, device='cuda' ))
            loss_D = train_discriminator(imgs,generator,discriminator, optimizer_D, z)
            if i==0 and epoch%10==0:
                dis_loss.append(loss_D)
        
        # Train the generator every n_critic iterations
            if i % config.N_CRITIC == 0:
                loss_G = train_generator(generator, discriminator,optimizer_G,z)
                if i==0 and epoch%10==0:
                    gen_loss.append(loss_G)

                if epoch%20 ==0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, args.Epochs_ete_Wgan, batches_done % len(dataloader), len(dataloader), loss_D, loss_G))
            batches_done += 1
    torch.save(generator.state_dict(), 'Wgan_end_to_end.pt')
    np.savetxt("wgan_generatorloss.txt",gen_loss)
    np.savetxt("wgan_criticloss.txt",dis_loss)
    np.savetxt("wgan_traintime.txt",np.array([time.time()-start_time]))
    os.chdir("..")
