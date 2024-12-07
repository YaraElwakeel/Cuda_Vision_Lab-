#!/usr/bin/env python
# coding: utf-8

# In[1]:


from kth_dataset import KTH_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from LSTMscratch import LSTMscratch
import torch
import numpy as np
import torch.nn as nn
from Model_Wrapper import Wrapper
import tqdm
import helper 


# In[2]:


helper.set_random_seed()


# In[3]:


# Create a dataset and a DataLoader
transforms1 = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Resize([32, 32]),  # Normalize to [-1, 1]
    ])
# Initialize dataset
test_dataset = KTH_Dataset(root_dir="/home/nfs/inf6/data/datasets/kth_actions/processed", sequence_length=50,split="test", transform=transforms1,use_saved_samples=True)
train_dataset = KTH_Dataset(root_dir="/home/nfs/inf6/data/datasets/kth_actions/processed", sequence_length=50,split="train", transform=transforms1,use_saved_samples=True)


# In[4]:


train_dataset.__len__()


# In[5]:


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=8,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,num_workers=8,pin_memory=True)
print("test_dataset per epoch",next(iter(test_loader))[0].size())


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


from ConvLSTMscratch import ConvLSTM


# In[8]:


model = LSTMscratch(input_dim=64, hidden_dim=16, number_of_layers=2,device=device)

# In[9]:


# model_t = SequentialClassifierWithCells(emb_dim=128, hidden_dim=128, num_layers=2, mode="zeros")
model = model.to(device)


# In[10]:


# classification loss function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Decay LR by a factor of 0.1 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)


# In[ ]:


classes = ["Boxing","HandClapping","HandWaving","Jogging","Running","Walking"]
# model = SequentialClassifierWithCells(emb_dim=128, hidden_dim=128, num_layers=2, mode="zeros").to(device)
writer = helper.new_writer("models","LSTMscratch")
train = Wrapper(model_name="LSTM", model = model, device = device, criterion = criterion, optimizer = optimizer,writer=writer,show_progress_bar= True)
train.train(10,train_loader,test_loader,classes)
writer.close()
train.valid_accuracy()
train.plot_loss_acc()



# In[ ]:




