import os 
from torch.utils.tensorboard import SummaryWriter 
import shutil
import numpy as np 
import torch
import torch.nn.functional as F



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

def vae_loss_function(recons, target, mu, log_var, lambda_kld=1e-3):
    # """
    #     Combined loss function for joint optimization of 
    #     reconstruction and ELBO
    #     """
        recons_loss = F.mse_loss(recons, target)
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)  # closed-form solution of KLD in Gaussian
        loss = recons_loss + lambda_kld * kld
        return loss, (recons_loss, kld)

