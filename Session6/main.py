from Dataset import Dataset
# from DCGAN import Generator, Discriminator
from DCGAN import Generator, Discriminator
from model_wrapper import Trainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


GAN_LOGS = os.path.join(os.getcwd(), "tboard_logs", "gan")
if not os.path.exists(GAN_LOGS):
    os.makedirs(GAN_LOGS)

shutil.rmtree(GAN_LOGS)
writer = SummaryWriter(GAN_LOGS)


# Create a dataset and a DataLoader
transforms = {'val':transforms.Compose([
    transforms.Resize([32, 32]),  # Resize the image first
    transforms.ToTensor(),        # Convert PIL image to tensor
]),
'train':transforms.Compose([
    transforms.Resize([32, 32]),  # Resize the image first
    transforms.RandomHorizontalFlip(p=0.5),       # Random horizontal flip
    transforms.ToTensor(),        # Convert PIL image to tensor
])
}


root_dir = "/home/user/zafara1/Cuda_Vision_Lab-/Session5/Solution/data/afhq"

# Initialize dataset
train_dataset = Dataset(root_dir=root_dir, split="train", transform=transforms["train"])
test_dataset = Dataset(root_dir=root_dir, split="val", transform=transforms["val"])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=8,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=8,pin_memory=True)
print("test_dataset per epoch",next(iter(test_loader))[0].size())


generator = Generator(latent_dim=128, num_channels=3, base_channels=32)
discriminator = Discriminator(in_channels=3, out_dim=1, base_channels=32)
trainer = Trainer(generator=generator, discriminator=discriminator, latent_dim=128, writer=writer)


trainer.train(data_loader=train_loader)