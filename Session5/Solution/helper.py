import os 
from torch.utils.tensorboard import SummaryWriter 
import shutil
import numpy as np 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt




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

def vae_loss_function(recons, target, mu, log_var, lambda_kld):
    # """
    #     Combined loss function for joint optimization of 
    #     reconstruction and ELBO
    #     """
        recons_loss = F.mse_loss(recons, target)
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)  # closed-form solution of KLD in Gaussian
        loss = recons_loss + lambda_kld * kld
        return loss, (recons_loss, kld)

@torch.no_grad()
def plot_reconstructed(model,device, xrange=(-3, 3), yrange=(-3, 3), resolution=12, image_size=32):
    """
    Sampling equispaced points from the latent space given the xrange and yrange, 
    decoding latents and visualizing the distribution of the latent space.
    """
    # Generate equispaced points in the latent space
    x = torch.linspace(*xrange, resolution, device=device)
    y = torch.linspace(*yrange, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Create latent vectors
    latent_grid = torch.zeros((resolution**2, model.latent_dim), device=device)
    latent_grid[:, 0] = xx.flatten()
    latent_grid[:, 1] = yy.flatten()

    # Decode latents
    gen_imgs = model.decode(latent_grid).cpu()
    gen_imgs = gen_imgs.view(-1, image_size, image_size).numpy()

    # Arrange decoded images into a grid
    grid = np.empty((resolution * image_size, resolution * image_size))
    for i in range(resolution):
        for j in range(resolution):
            img_idx = i * resolution + j
            grid[
                i * image_size:(i + 1) * image_size,
                j * image_size:(j + 1) * image_size,
            ] = gen_imgs[img_idx]

    # Plot the grid
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, extent=[*xrange, *yrange], cmap="gray")
    plt.axis("off")
    plt.title("Latent Space Visualization")
    plt.show()
    
def show_recons_test(data,model,device,condition = False):
    imgs, labels = next(iter(data)) 
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=3).float().to(device)


    model.eval()
    with torch.no_grad():
        if condition:
            recons, _ = model(imgs.to(device),one_hot_labels)
        else:
            recons, _ = model(imgs.to(device))

        
    fig, ax = plt.subplots(2, 8)
    fig.set_size_inches(18, 8)
    for i in range(8):
        ax[0, i].imshow(imgs[i, 0], cmap="gray")
        ax[0, i].axis("off")
        ax[1, i].imshow(recons[i, 0].cpu(), cmap="gray")
        ax[1, i].axis("off")

    ax[0, 3].set_title("Original Image")
    ax[1, 3].set_title("Reconstruction")
    plt.tight_layout()
    plt.show()
