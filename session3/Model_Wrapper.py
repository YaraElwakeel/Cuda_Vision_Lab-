import torch
import numpy as np
import seaborn as sn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class Wrapper():
    def __init__(self, model, device, criterion, optimizer, scheduler=None, warmup_lr=None,writer = None):
        # Track training and testing loss and accuracy for each epoch
        self.loss_hist = [] 
        self.acc_hist = []
        self.loss_test_hist = []
        self.acc_test_hist = []

        # Store scheduler and warmup learning rate if provided
        self.scheduler = scheduler
        self.warmup_lr = warmup_lr

        # store the tensorboard writer 
        self.writer = writer 
        
        # Initialize lists for predictions and true labels for confusion matrix
        self.predictions = []
        self.true_labels = []
        
        # Define model parameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.device = device

        # Initialize other attributes to None for later setup
        self.classes = None
        self.testloader = None
        self.trainloader = None

    def train(self, num_epochs, trainloader, testloader, classes):
        # Set classes and loaders
        self.classes = classes
        self.trainloader = trainloader
        self.testloader = testloader

        for epoch in range(num_epochs):
            correct = 0 
            total = 0 
            loss_list = []
            acc_list = []

            # Reset predictions and true_labels for this epoch
            self.predictions = []
            self.true_labels = []

            # Display a progress bar for each epoch
            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))
            for i, (inputs, labels) in enumerate(progress_bar):
                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss_list.append(loss.item())
                
                # Backpropagation and optimizer step
                loss.backward()
                self.optimizer.step()
                
                # Apply learning rate scheduling if provided
                if self.warmup_lr:
                    self.warmup_lr.step(epoch)
                if self.scheduler:
                    self.scheduler.step(T_curr=epoch, T_max=num_epochs)

                # Calculate accuracy for current batch
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                accuracy = 100 * correct / total
                acc_list.append(accuracy)

                # Update progress bar description every 10 batches
                if (i % 10 == 0 or i == len(self.trainloader) - 1):
                    progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}.")
            
            # compute loss and accuracy mean per epoch
            acc_mean= np.mean(loss_list)
            loss_mean = np.mean(acc_list)

            # Log train matrices 
            self.loss_hist.append(loss_mean)
            self.acc_hist.append(acc_mean)

            # log metrices in tensorboard
            self.writer.add_scalar(f"Accuracy/train",acc_mean,global_step = epoch)
            self.writer.add_scalar(f"Loss/train",loss_mean,global_step = epoch)

            # Evaluate on the test data and log test metrics
            loss_test_list, test_accuracy, epoch_predictions, epoch_true_labels = self.eval()
            
            # compute loss mean across an epoch 
            loss_test_mean = np.mean(loss_test_list)

            # log test metrices
            self.loss_test_hist.append(loss_test_mean)
            self.acc_test_hist.append(test_accuracy)

            # log metrices in tensorboard
            self.writer.add_scalar(f"Accuracy/test",test_accuracy,global_step = epoch)
            self.writer.add_scalar(f"Loss/test",loss_test_mean,global_step = epoch)

            # Store predictions and true labels for confusion matrix
            self.predictions.extend(epoch_predictions)
            self.true_labels.extend(epoch_true_labels)

    def eval(self):
        correct = 0
        total = 0
        true_labels = []
        predictions = []
        loss_list = []
        
        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Compute loss for this batch
                loss = self.criterion(outputs, labels)
                loss_list.append(loss.item())

                # Compute accuracy for this batch
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate the overall test accuracy
        accuracy = 100 * correct / total
        return loss_list, accuracy, predictions, true_labels

    def confusion_matrix(self):
        # Check if predictions and true_labels are available
        if self.true_labels and self.predictions:        
            # Generate and display the confusion matrix
            cf_matrix = confusion_matrix(self.true_labels, self.predictions)
            sn.heatmap(cf_matrix, annot=True, fmt="d", xticklabels=self.classes, yticklabels=self.classes)

            plt.figure(figsize=(12, 7))
            plt.savefig('output.png')
        else:
            print("No predictions available. Please run `train` first.")
    
    def valid_accuracy (self):
            # Print validation accuracy
            accuracy = accuracy_score(self.true_labels, self.predictions) * 100
            print(f"Validation accuracy: {round(accuracy, 2)}%")

    def plot_loss_acc(self):
        # Create subplots for loss and accuracy
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(16, 5)

        # Plot training and testing loss
        ax[0].plot(self.loss_hist, label="Train Loss", color="blue", linewidth=3)
        ax[0].plot(self.loss_test_hist, color="red", label="Test Loss", linewidth=3)
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(loc="best")

        # Plot training and testing accuracy
        ax[1].plot(self.acc_hist, label="Train Accuracy", color="blue", linewidth=3)
        ax[1].plot(self.acc_test_hist, color="red", label="Test Accuracy", linewidth=3)
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend(loc="best")

        plt.show()
