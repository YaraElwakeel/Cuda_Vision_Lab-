import torch
from torchvision.transforms import transforms
from ConvLSTMscratch import ConvLSTM
from kth_dataset import KTH_Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ConvLSTM(input_dim=1, hidden_dim=4, kernel_size=3, num_layers=2,batch_first=True)
model = model.to(device)

# Create a dataset and a DataLoader
transforms1 = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Resize([32,32]),  # Normalize to [-1, 1]
    ])
# Initialize dataset
test_dataset = KTH_Dataset(root_dir="/home/nfs/inf6/data/datasets/kth_actions/processed", sequence_length=15,split="test", transform=transforms1,use_saved_samples=True)
train_dataset = KTH_Dataset(root_dir="/home/nfs/inf6/data/datasets/kth_actions/processed", sequence_length=15,split="train", transform=transforms1,use_saved_samples=True)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=2,pin_memory=True)
data, labels = next(iter(train_loader))

data = data.to(device)
output = model(data)

# print(output)