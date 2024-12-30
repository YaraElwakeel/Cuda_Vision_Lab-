import os
import torch
import numpy as np
import seaborn as sn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class Wrapper():
    def __init__(self, model_name,model, device, criterion, optimizer, writer=None, scheduler=None, warmup_lr=None, show_progress_bar=True):
        # Track training and testing loss and accuracy for each epoch
        self.loss_hist = [] 
        self.loss_kld_hist = []
        self.loss_recons_hist = []

        # self.acc_hist = []
        self.loss_test_hist = []
        self.loss_test_kld_hist = []
        self.loss_test_recons_hist = []
        # self.acc_test_hist = []

        # Store scheduler and warmup learning rate if provided
        self.scheduler = scheduler
        self.warmup_lr = warmup_lr
        
        # Initialize lists for predictions and true labels for confusion matrix
        self.predictions = []
        self.true_labels = []
        

        self.model_name = model_name
        # Store the tensorboard writer 
        self.writer = writer

        # Define model parameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.device = device

        # Initialize other attributes to None for later setup
        self.classes = None
        self.testloader = None
        self.trainloader = None

        # Flag for showing progress bar
        self.show_progress_bar = show_progress_bar

    def train(self, num_epochs, trainloader, testloader):
        # Set classes and loaders
        # self.classes = classes
        self.trainloader = trainloader
        self.testloader = testloader

        self.model.train()
        for epoch in range(num_epochs):
            # correct = 0 
            # total = 0 
            # acc_list = []

            loss_list = []
            recons_loss = []
            vae_loss = []

            # Reset predictions and true_labels for this epoch
            # self.predictions = []
            # self.true_labels = []

            # Display a progress bar for each epoch if the flag is True
            if self.show_progress_bar:
                progress_bar = tqdm(self.trainloader, total=len(self.trainloader))
            else:
                progress_bar = self.trainloader

            for i, (inputs, labels) in enumerate(progress_bar):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                recons, (z, mu, log_var) = self.model(inputs)


                # Calculate Loss
                loss, (mse, kld) = self.criterion(recons, inputs, mu, log_var)
                
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


                # Calculate accuracy for current batch
                # preds = torch.argmax(outputs, dim=1)
                # correct += (preds == labels).sum().item()
                # total += labels.size(0)
                # accuracy = 100 * correct / total
                # acc_list.append(accuracy)

                # Update progress bar description every 10 batches
                if (i % 10 == 0 or i == len(self.trainloader) - 1) and self.show_progress_bar == True:
                    progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}.")

                
            
            loss_mean = np.mean(loss_list)
            recons_loss_mean = np.mean (recons_loss)
            vae_loss_mean = np.mean(vae_loss)
            # acc_mean = np.mean(acc_list)


            # Log epoch loss and accuracy
            self.loss_hist.append(loss_mean)
            self.loss_kld_hist.append(vae_loss_mean)
            self.loss_recons_hist.append(recons_loss_mean)
            # self.acc_hist.append(acc_mean)

            # print(f"Accuracy/train", acc_mean)
            print(f"Loss/train", loss_mean)

            # Log metrics in tensorboard
            if self.writer :
                # self.writer.add_scalar(f"Accuracy/train", acc_mean, global_step=epoch)
                self.writer.add_scalar(f"Loss/train", loss_mean, global_step=epoch)

            # Evaluate on the test data and log test metrics
            loss_test_list,recons_test_loss,kld_test_loss = self.eval()
            loss_test_mean = np.mean(loss_test_list)
            recons_test_loss_mean = np.mean(recons_test_loss)
            kld_test_loss_mean = np.mean(kld_test_loss)

            
            self.loss_test_hist.append(loss_test_mean)
            self.loss_test_recons_hist.append(recons_test_loss_mean)
            self.loss_test_kld_hist.append(kld_test_loss_mean)
            # self.acc_test_hist.append(test_accuracy)

            # print(f"Accuracy/test", test_accuracy)
            print(f"Loss/test", loss_test_mean)
            print(f"Loss/test", recons_test_loss_mean)
            print(f"Loss/test", kld_test_loss_mean)
            # Log metrics in tensorboard
            if self.writer :
                # self.writer.add_scalar(f"Accuracy/test", test_accuracy, global_step=epoch)
                self.writer.add_scalar(f"Loss/test", loss_test_mean, global_step=epoch)

            # Store predictions and true labels for confusion matrix
            # self.predictions.extend(epoch_predictions)
            # self.true_labels.extend(epoch_true_labels)
            self.save_model(self.model, self.optimizer, epoch, {"test_loss":loss_test_mean, "train_loss": loss_mean})


    def eval(self):
        # correct = 0
        # total = 0
        # true_labels = []
        # predictions = []
        loss_list = []
        recons_loss = []
        kld_loss = []
        
        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Compute loss for this batch
                recons, (z, mu, log_var) = self.model(inputs)
                loss, (mse, kld) = self.criterion(recons, inputs, mu, log_var)
                
                loss_list.append(loss.item())
                recons_loss.append(mse.item())
                kld_loss.append(kld.item())

                # Compute accuracy for this batch
                # _, predicted = torch.max(outputs, dim=1)
                # correct += (predicted == labels).sum().item()
                # total += labels.size(0)
                # predictions.extend(predicted.cpu().numpy())
                # true_labels.extend(labels.cpu().numpy())

        # Calculate the overall test accuracy
        # accuracy = 100 * correct / total
        # return loss_list, accuracy, predictions, true_labels
        return loss_list, recons_loss, kld_loss
    
    def confusion_matrix(self):
        # Check if predictions and true_labels are available
        if self.true_labels and self.predictions:
            # Generate the confusion matrix
            cf_matrix = confusion_matrix(self.true_labels, self.predictions)
            print(cf_matrix)
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

    
    def valid_accuracy(self):
        # Print validation accuracy
        accuracy = accuracy_score(self.true_labels, self.predictions) * 100
        print(f"Validation accuracy: {round(accuracy, 2)}%")


    def concatenate_matrices(self,loss, acc, loss_test, acc_test ):
        self.loss_hist.extend(loss) 
        self.loss_test_hist.extend( loss_test)
        self.acc_hist.extend(acc) 
        self.acc_test_hist.extend(acc_test) 

    def plot_loss_acc(self,line=None):
        # Create subplots for loss and accuracy
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(16, 5)

        # Plot training and testing loss
        ax[0].plot(self.loss_hist, label="Train Loss", color="blue", linewidth=3)
        ax[0].plot(self.loss_test_hist, color="red", label="Test Loss", linewidth=3)
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(loc="best")
        if line:
            ax[0].axvline(x=line, color="red", ls="--", linewidth=3)



        # Plot training and testing accuracy
        ax[1].plot(self.acc_hist, label="Train Accuracy", color="blue", linewidth=3)
        ax[1].plot(self.acc_test_hist, color="red", label="Test Accuracy", linewidth=3)
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend(loc="best")
        if line: 
            ax[1].axvline(x=line, color="red", ls="--", linewidth=3)
        
        plt.savefig("accuracy_plot.png")
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

