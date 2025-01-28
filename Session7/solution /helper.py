import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import numpy as np 
import os 

def db_statistics(df):
    # ----------- Summary Statistics -----------
    total_people = len(df)
    total_images = df['images'].sum()
    max_images = df['images'].max()
    single_count = (df['images'] == 1).sum()
    multi_count = (df['images'] > 1).sum()

    print(f"Total People: {total_people}")
    print(f"Total Images: {int(total_images)}")
    print(f"Max Images for a Person: {int(max_images)}")
    print(f"Total People with one Image: {single_count}")
    print(f"Total People with multiple Images: {multi_count}")

    # ----------- Top 10 People with Most Images -----------
    top_people = df.sort_values(by='images', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(y=top_people['name'], x=top_people['images'])
    plt.ylabel("Person Name")
    plt.title("Top 10 People with Most Images")
    plt.show()

    # ----------- Percentage of People with Only 1 Image -----------
    single_count = (df['images'] == 1).sum()
    multi_count = (df['images'] > 1).sum()
    labels = ['Single Image', 'Multiple Images']
    sizes = [single_count, multi_count]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'skyblue'])
    plt.title("Proportion of People with Single vs. Multiple Images")
    plt.show()
    return top_people 

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
    
    
def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def visualize_progress(train_loss, val_loss, start=0):
    """ Visualizing loss and accuracy """
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_train = smooth(train_loss, 31)
    ax[0].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_yscale("linear")
    ax[0].set_title("Training Progress (linear)")
    
    ax[1].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (log)")

    smooth_val = smooth(val_loss, 31)
    N_ITERS = len(val_loss)
    ax[2].plot(np.arange(start, N_ITERS)+start, val_loss[start:], c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[2].plot(np.arange(start, N_ITERS)+start, smooth_val[start:], c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("CE Loss")
    ax[2].set_yscale("log")
    ax[2].set_title(f"Valid Progress")

    return