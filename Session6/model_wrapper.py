import os
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from torcheval.metrics import FrechetInceptionDistance
# from torch.utils.tensorboard import SummaryWriter



class Trainer:
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, latent_dim=128, conditional_gan=False,writer=None):
        """ Initialzer """
        assert writer is not None, f"Tensorboard writer not set..."
    
        self.latent_dim = latent_dim
        self.writer = writer 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.conditional_gan = conditional_gan
        
        self.optim_discriminator = optimizer_d
        self.optim_generator = optimizer_g
        
        self.fid = FrechetInceptionDistance(device=self.device) 
        # REAL LABEL = 1
        # FAKE LABEL = 0
        # eps = 1e-10
        # self.criterion_d_real = lambda pred: torch.clip(-torch.log(1 - pred + eps), min=-10).mean()
        # self.criterion_d_fake = lambda pred: torch.clip(-torch.log(pred + eps), min=-10).mean()
        # self.criterion_g = lambda pred: torch.clip(-torch.log(1 - pred + eps), min=-10).mean()
        
        self.criterion_g = lambda pred: F.binary_cross_entropy(pred, torch.ones(pred.shape[0], device=pred.device))
        self.criterion_d_real = lambda pred: F.binary_cross_entropy(pred, torch.ones(pred.shape[0], device=pred.device))
        self.criterion_d_fake = lambda pred: F.binary_cross_entropy(pred, torch.zeros(pred.shape[0], device=pred.device))
        
        
        self.hist = {
            "d_real": [],
            "d_fake": [],
            "g_loss": []
        }
        self.fid_score_hist = []

        return
        
    def train_one_step(self, imgs):
        """ 
        Training both models for one optimization step
        """
        self.generator.train()
        self.discriminator.train()
        
        # Sample from the latent distribution
        B = imgs.shape[0]
        latent = torch.randn(B, self.latent_dim, 1, 1).to(self.device)
        
        # ==== Training Discriminator ====
        self.optim_discriminator.zero_grad()
        # Get discriminator outputs for the real samples
        prediction_real = self.discriminator(imgs)
        # Compute the loss function
        d_loss_real = self.criterion_d_real(prediction_real.view(B))

        # Generating fake samples with the generator
        fake_samples = self.generator(latent)
        # Get discriminator outputs for the fake samples
        prediction_fake_d = self.discriminator(fake_samples.detach())  # why detach?
        # Compute the loss function
        d_loss_fake = self.criterion_d_fake(prediction_fake_d.view(B))
        (d_loss_real + d_loss_fake).backward()
        assert fake_samples.shape == imgs.shape
        
        # optimization step
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 3.0)
        self.optim_discriminator.step()
        
        # === Train the generator ===
        self.optim_generator.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples)
        # Compute the loss function
        g_loss = self.criterion_g(prediction_fake_g.view(B))
        g_loss.backward()
        # optimization step
        self.optim_generator.step()
        
        return d_loss_real, d_loss_fake, g_loss
    
    def train_one_step_conditional(self, imgs, labels):

        self.generator.train()
        self.discriminator.train()

        criterion = torch.nn.BCELoss()
        batch_size = imgs.size(0)
        real_images, labels = imgs.to(self.device), labels.to(self.device)
        
        # Create labels for real and fake images
        real_labels = torch.ones(batch_size, 1, 1, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1).to(self.device)
        
        # Train Discriminator
        self.optim_discriminator.zero_grad()
        
        # Real images
        real_outputs = self.discriminator(real_images, labels)
        d_loss_real = criterion(real_outputs, real_labels)
        
        
        # Fake images
        z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(z, labels)
        fake_outputs = self.discriminator(fake_images.detach(), labels)
        d_loss_fake = criterion(fake_outputs, fake_labels)

        (d_loss_real + d_loss_fake).backward()
        self.optim_discriminator.step()
        
        # Train Generator
        self.optim_generator.zero_grad()
        
        fake_outputs = self.discriminator(fake_images, labels)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        self.optim_generator.step()

        return d_loss_real, d_loss_fake, g_loss




    @torch.no_grad()
    def generate(self, labels=None,N=64):
        """ Generating a bunch of images using current state of generator """
        self.generator.eval()
        latent = torch.randn(N, self.latent_dim, 1, 1).to(self.device)
        if(self.conditional_gan):
            imgs = self.generator(latent, labels=labels)
        else:
            imgs = self.generator(latent)
        imgs = imgs * 0.5 + 0.5
        return imgs
        
    def train(self, data_loader, epochs=50, init_step=0):
        """ Training the models for several iterations """
        
        
        running_d_loss = 0
        running_g_loss = 0
        
        iter_ = 0
        for i in range(epochs):

            progress_bar = tqdm(data_loader, total=len(data_loader))
            for j, (real_batch, labels) in enumerate(progress_bar):           
                real_batch = real_batch.to(self.device)
                labels = labels.to(self.device)
                if(self.conditional_gan):
                    d_loss_real, d_loss_fake, g_loss = self.train_one_step_conditional(imgs=real_batch, labels=labels)    
                else:
                    d_loss_real, d_loss_fake, g_loss = self.train_one_step(imgs=real_batch)
                
                

                d_loss = d_loss_real + d_loss_fake
            
                # updating progress bar
                progress_bar.set_description(f"Epoch {i+1} Iter {iter_}: D_Loss={round(d_loss.item(),5)}, G_Loss={round(g_loss.item(),5)})")
                
                # adding stuff to tensorboard
                self.writer.add_scalar(f'Loss/Generator Loss', g_loss.item(), global_step=iter_)
                self.writer.add_scalar(f'Loss/Discriminator Loss', d_loss.item(), global_step=iter_)
                self.writer.add_scalars(f'Loss/Discriminator Losses', {
                        "Real Images Loss": d_loss_real.item(),
                        "Fake Images Loss": d_loss_fake.item(),
                    }, global_step=iter_)
                self.writer.add_scalars(f'Comb_Loss/Losses', {
                            'Discriminator': d_loss.item(),
                            'Generator':  g_loss.item()
                        }, iter_)   
                 
                if(iter_ % 200 == 0):
                    if(self.conditional_gan):            
                        imgs = self.generate(labels=labels)
                        grid = torchvision.utils.make_grid(imgs, nrow=8)
                        self.writer.add_image('images', grid, global_step=iter_)
                        torchvision.utils.save_image(grid, os.path.join(os.getcwd(), "imgs", "training", f"conditional_imgs_{iter_}.png"))
                        if(iter_==11400):
                            print(labels)
                    else:
                        imgs = self.generate()  
                        grid = torchvision.utils.make_grid(imgs, nrow=8)
                        self.writer.add_image('images', grid, global_step=iter_)
                        torchvision.utils.save_image(grid, os.path.join(os.getcwd(), "imgs", "training", f"imgs_{iter_}.png"))
                iter_ = iter_ + 1

                

            self.fid.update(real_batch, is_real=True)
            self.fid.update(imgs, is_real=False)
            fid_score = self.fid.compute()
            self.writer.add_scalar(f'FID score', fid_score, global_step=i)
            self.fid_score_hist.append(fid_score.cpu())
            self.fid.reset()
            self.hist["d_real"].append(d_loss_real.item())
            self.hist["d_fake"].append(d_loss_fake.item())
            self.hist["g_loss"].append(g_loss.item())
                
        self.plot_loss_acc()    
            # print("Fid score", self.fid)
        return
    

    def plot_loss_acc(self,line=None):
        # Create subplots for loss and accuracy
        # plt.style.use('seaborn')
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(16, 10)

        # Plot training and testing loss
        ax[0][0].plot(self.hist["d_fake"], label="Discriminator Fake Loss", color="blue", linewidth=3)
        ax[0][0].set_xlabel("Epochs")
        ax[0][0].set_ylabel("Loss")
        ax[0][0].legend(loc="best")

        # Plot Independent Loss Curves
        ax[0][1].plot(self.hist["d_real"], label= "Discriminator Real Loss")
        ax[0][1].set_xlabel("Epochs")
        ax[0][1].set_ylabel("loss")
        ax[0][1].legend(loc="best")
 
        ax[1][0].plot(self.hist["g_loss"], label = "Generator Loss")
        ax[1][0].set_xlabel("Epochs")
        ax[1][0].set_ylabel("score")
        ax[1][0].legend(loc="best")

        ax[1][1].plot(self.fid_score_hist, label = "FID score")
        ax[1][1].set_xlabel("Epochs")
        ax[1][1].set_ylabel("score")
        ax[1][1].legend(loc="best")


        plt.savefig("accuracy_plots.png")
        plt.show()

