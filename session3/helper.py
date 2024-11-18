import os 
from torch.utils.tensorboard import SummaryWriter 
import shutil
import numpy as np 
import torch
import matplotlib.pyplot as plt
from Model_Wrapper import Wrapper


def new_writer(file,model):
    TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", file, model)
    if not os.path.exists(TBOARD_LOGS):
        os.makedirs(TBOARD_LOGS)
    
    shutil.rmtree(TBOARD_LOGS)
    writer = SummaryWriter(TBOARD_LOGS)
    return writer 

def set_random_seed(random_seed=None):
    if random_seed is None:
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {random_seed}")

def show_grid(data, titles=None):
    """Imshow for Tensor."""
    data = data.numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    data = std * data + mean 
    data = np.clip(data, 0, 1)
    
    plt.figure(figsize=(8*2, 4*2))
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(data[i])
        plt.axis("off")
        if titles is not None:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

def plot_combined_loss_acc(train, train_combined, line=None):
    # Combine histories for losses and accuracies
    combined_loss_hist = train.loss_hist + train_combined.loss_hist
    combined_loss_test_hist = train.loss_test_hist + train_combined.loss_test_hist
    combined_acc_hist = train.acc_hist + train_combined.acc_hist
    combined_acc_test_hist = train.acc_test_hist + train_combined.acc_test_hist

    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot combined loss
    axes[0].plot(combined_loss_hist, label="Train Loss", color="blue", linewidth=3)
    axes[0].plot(combined_loss_test_hist, label="Test Loss", color="red", linewidth=3)
    axes[0].set_title("Combined Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")
    if line:
        axes[0].axvline(x=line, color="red", ls="--", linewidth=3)  # Show where phase 1 ends

    # Plot combined accuracy
    axes[1].plot(combined_acc_hist, label="Train Accuracy", color="blue", linewidth=3)
    axes[1].plot(combined_acc_test_hist, label="Test Accuracy", color="red", linewidth=3)
    axes[1].set_title("Combined Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="best")
    if line:
        axes[1].axvline(x=line, color="red", ls="--", linewidth=3)  # Show where phase 1 ends

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()