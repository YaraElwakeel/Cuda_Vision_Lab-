import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import seaborn as sn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
# import utilities

class Wrapper():
    def __init__(self, model, device, criterion, optimizer,scheduler=None,warmup_lr = None):
        self.loss_hist = [] 
        self.acc_hist = []
        self.loss_test_hist = []
        self.acc_test_hist = []
        self.scheduler = scheduler
        self.warmup_lr = warmup_lr
        
        # Initialize predictions and true_labels as empty lists
        self.predictions = []
        self.true_labels = []
        
        self.criterion = criterion
        self.optimizer = optimizer

        self.model = model
        self.device = device
        
        self.classes = None
        self.testloader = None
        self.trainloader = None

    def train(self, num_epochs, trainloader, testloader, classes):
        self.classes = classes
        self.trainloader = trainloader
        self.testloader = testloader

        for epoch in range(num_epochs):
            correct = 0 
            total = 0 

            loss_list = []
            acc_list = []

            # Reset predictions and true_labels for each epoch
            self.predictions = []
            self.true_labels = []

            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                loss_list.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                if self.warmup_lr : self.warmup_lr.step(epoch) 

        
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                accuracy = 100 * correct / total
                acc_list.append(accuracy)
        
                if (i % 10 == 0 or i == len(self.trainloader) - 1):
                    progress_bar.set_description(f"Epoch {epoch + 1} Iter {i + 1}: loss {loss.item():.5f}.")
            
            self.loss_hist.append(np.mean(loss_list))
            self.acc_hist.append(np.mean(acc_list))

            # Evaluate on the test data
            loss_test_list, test_accuracy, epoch_predictions, epoch_true_labels = self.eval()
            self.loss_test_hist.append(np.mean(loss_test_list))
            self.acc_test_hist.append(test_accuracy)
            
            # Append predictions and true labels from the eval function
            self.predictions.extend(epoch_predictions)
            self.true_labels.extend(epoch_true_labels)


    def eval(self):
        correct = 0
        total = 0

        true_labels = []
        predictions = []
        loss_list = []
        
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss_list.append(loss.item())

                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate the overall accuracy
        accuracy = 100 * correct / total
        return loss_list, accuracy, predictions, true_labels

    def confusion_matrix(self):
        # Ensure predictions and true_labels are available
        if self.true_labels and self.predictions:
            accuracy = accuracy_score(self.true_labels, self.predictions) * 100
            print(f"Validation accuracy: {round(accuracy, 2)}%")
            
            # Generate confusion matrix
            print(len(self.true_labels), len(self.predictions))
            cf_matrix = confusion_matrix(self.true_labels, self.predictions)
            sn.heatmap(cf_matrix, annot=True, fmt="d")

            plt.figure(figsize=(12, 7))
            plt.savefig('output.png')
        else:
            print("No predictions available. Please run `train` first.")
    
    
    def plot_loss_acc(self):
        # plt.style.use('seaborn')

        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(16,5)

        
        ax[0].plot(self.loss_hist,label="Train Loss",c="blue", linewidth=3)
        ax[0].plot(self.loss_test_hist, c="red", label="Test Loss", linewidth=3)
        ax[0].set_xlabel("Epochs")
        ax[0].legend(loc="best")
        ax[0].set_ylabel("CE Loss value")

        ax[1].plot(self.acc_hist, label="Train Accuracy",c="blue", linewidth=3)
        ax[1].plot(self.acc_test_hist, c="red", label="Test Accuracy", linewidth=3)
        ax[1].set_xlabel("Epochs")
        ax[1].legend(loc="best")
        ax[1].set_ylabel("CE Accuracy value")

        plt.show()
        
