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

