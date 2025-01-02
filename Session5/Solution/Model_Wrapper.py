import os
import torch
import numpy as np
import seaborn as sn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torcheval.metrics import FrechetInceptionDistance
from torchvision.utils import save_image,make_grid 


class Wrapper():
    def __init__(self, model_name,model, device, criterion, optimizer, writer=None, scheduler=None, warmup_lr=None, show_progress_bar=True,print_epoch_values=True,lambda_kld=1e-3,condition = False ):
        # Track training and testing loss for each epoch
        self.loss_hist = [] 
        self.loss_kld_hist = []
        self.loss_recons_hist = []

        self.loss_test_hist = []
        self.loss_test_kld_hist = []
        self.loss_test_recons_hist = []

        self.fid_score_hist = []

        # Store scheduler and warmup learning rate if provided
        self.scheduler = scheduler
        self.warmup_lr = warmup_lr
        
        self.model_name = model_name

        # Store the tensorboard writer 
        self.writer = writer

        # Define model parameters
        self.criterion = criterion
        self.lambda_kld = lambda_kld
        self.optimizer = optimizer
        self.model = model
        self.device = device

        # Initialize other attributes to None for later setup
        self.testloader = None
        self.trainloader = None

        # Flag for showing progress bar
        self.show_progress_bar = show_progress_bar
        self.print_epoch_values= print_epoch_values

        self.img_save_path = 'img'

        # definr the french inception distance 
        self.fid = FrechetInceptionDistance(device=device) 
        
        self.condition = condition
   
    def train(self, num_epochs, trainloader, testloader):
        # Set classes and loaders
        # self.classes = classes
        self.trainloader = trainloader
        self.testloader = testloader

        self.model.train()
        for epoch in range(num_epochs):

            loss_list = []
            recons_loss = []
            vae_loss = []

            # Display a progress bar for each epoch if the flag is True
            if self.show_progress_bar:
                progress_bar = tqdm(self.trainloader, total=len(self.trainloader))
            else:
                progress_bar = self.trainloader

            for i, (inputs, labels) in enumerate(progress_bar):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=3).float().to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                if self.condition:
                    recons, (z, mu, log_var) = self.model(inputs,one_hot_labels)
                else:
                    recons, (z, mu, log_var) = self.model(inputs)

                # Calculate Loss
                loss, (mse, kld) = self.criterion(recons, inputs, mu, log_var,self.lambda_kld)
                
                loss_list.append(loss.item())
                recons_loss.append(mse.item())
                vae_loss.append(kld.item())
                
                # Backpropagation and optimizer step
                loss.backward()
                self.optimizer.step()
                
                # Apply learning rate scheduling if provided
                if self.warmup_lr:
                    self.warmup_lr.step(epoch)
                if self.scheduler:
                    self.scheduler.step()

                if labels.ndimension() > 1:
                    labels = torch.argmax(labels, dim=1)

                # Update progress bar description every 10 batches
                if (i % 10 == 0 or i == len(self.trainloader) - 1) and self.show_progress_bar == True:
                    progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}.")

            loss_mean = round(np.mean(loss_list),5)
            recons_loss_mean = round(np.mean (recons_loss),5)
            vae_loss_mean = round(np.mean(vae_loss),5)


            # Log epoch loss 
            self.loss_hist.append(loss_mean)
            self.loss_kld_hist.append(vae_loss_mean)
            self.loss_recons_hist.append(recons_loss_mean)


            # Log metrics in tensorboard
            if self.writer :
                self.writer.add_scalar(f"Loss/train", loss_mean, global_step=epoch)
                self.writer.add_scalar(f"recon_Loss/train", recons_loss_mean, global_step=epoch)
                self.writer.add_scalar(f"kld_Loss/train", vae_loss_mean, global_step=epoch)

            # Evaluate on the test data and log test metrics
            loss_test_list,recons_test_loss,kld_test_loss = self.eval(epoch)
            # Compute FID after processing all batches
            fid_score = self.fid.compute()
            self.fid.reset()  # Reset FID state after computation


            self.fid_score_hist.append(fid_score.cpu())

            loss_test_mean = round(np.mean(loss_test_list),5)
            recons_test_loss_mean = round(np.mean(recons_test_loss),5)
            kld_test_loss_mean = round(np.mean(kld_test_loss),5)

            
            self.loss_test_hist.append(loss_test_mean)
            self.loss_test_recons_hist.append(recons_test_loss_mean)
            self.loss_test_kld_hist.append(kld_test_loss_mean)
            if self.print_epoch_values:
                print(f"Train Loss", loss_mean)
                print(f"Test Loss", loss_test_mean)
                print(f"    Test recons_Loss", recons_test_loss_mean)
                print(f"    Test kld_Loss", kld_test_loss_mean)
            # Log metrics in tensorboard
            if self.writer :
                self.writer.add_scalar(f"Loss/test", loss_test_mean, global_step=epoch)
                self.writer.add_scalar(f"recons_Loss/test", recons_test_loss_mean, global_step=epoch)
                self.writer.add_scalar(f"kld_Loss/test", kld_test_loss_mean, global_step=epoch)
                self.writer.add_scalar(f"fid_score", fid_score, global_step=epoch)


            # self.save_model(self.model, self.optimizer, epoch, {"test_loss":loss_test_mean, "train_loss": loss_mean})


    def eval(self,epoch):

        loss_list = []
        recons_loss = []
        kld_loss = []

        # Reset FID state before evaluation
        self.fid.reset()
        
        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i,(inputs, labels) in enumerate(self.testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=3).float().to(self.device)

                
                # Compute loss for this batch
                if self.condition:
                    recons, (z, mu, log_var) = self.model(inputs, one_hot_labels)
                else:
                    recons, (z, mu, log_var) = self.model(inputs)
                loss, (mse, kld) = self.criterion(recons, inputs, mu, log_var,self.lambda_kld)

                # Convert grayscale to RGB 
                rgb_inputs = inputs.repeat(1, 3, 1, 1)  # [batch_size, 1, H, W] -> [batch_size, 3, H, W]
                rgb_recons = recons.repeat(1, 3, 1, 1)
                
                self.fid.update(rgb_inputs,is_real= True)
                self.fid.update(rgb_recons, is_real = False)

                loss_list.append(loss.item())
                recons_loss.append(mse.item())
                kld_loss.append(kld.item())

                if i==0:
                    if not os.path.exists(self.img_save_path):
                        os.makedirs(self.img_save_path)
                    save_image( inputs[:64].cpu(), os.path.join(self.img_save_path, f"{self.model_name}input_{epoch}.png") )
                    save_image( recons[:64].cpu(), os.path.join(self.img_save_path, f"{self.model_name}recons{epoch}.png") )
                    if self.writer is not None:
                        grid = make_grid(inputs[:64].cpu())
                        self.writer.add_image(f'images', grid, epoch)
                        grid = make_grid(recons[:64].cpu())
                        self.writer.add_image('output_images', grid, epoch)

        return loss_list, recons_loss, kld_loss
    
    def confusion_matrix(self):
        # Check if predictions and true_labels are available
        if self.true_labels and self.predictions:
            # Generate the confusion matrix
            cf_matrix = confusion_matrix(self.true_labels, self.predictions)
            # Plot the heatmap
            plt.figure(figsize=(12, 7))  # Configure the figure size first
            sn.heatmap(cf_matrix, annot=True, fmt="d", xticklabels=self.classes, yticklabels=self.classes, cmap="Blues")
            
            # Save and display
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            plt.savefig("output.png")
            plt.show()
        else:
            print("No predictions available. Please run `train` first.")

    
    def plot_loss_acc(self,line=None):
        # Create subplots for loss and accuracy
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(16, 5)

        # Plot training and testing loss
        ax[0].plot(self.loss_hist, label="Train Loss", color="blue", linewidth=3)
        ax[0].plot(self.loss_test_hist, color="red", label="Test Loss", linewidth=3)
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(loc="best")

        # Plot Independent Loss Curves
        ax[1].plot(self.loss_test_hist,label = "Test Loss Totall")
        ax[1].plot(self.loss_test_recons_hist, label="recons. Loss")
        ax[1].plot(self.loss_test_kld_hist, label="KLD Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("loss")
        ax[1].set_yscale("log")
        ax[1].legend(loc="best")

        # Plot the Fid score curve 
        ax[2].plot(self.fid_score_hist, label = "French inception score")
        ax[2].set_xlabel("Epochs")
        ax[2].set_ylabel("score")
        ax[2].legend(loc="best")

        plt.show()

    
    def save_model(self, model, optimizer, epoch, stats):
        if(not os.path.exists("models")):
            os.makedirs("models")
        savepath = f"models/{self.model_name}checkpoint_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats
        }, savepath)
        return

