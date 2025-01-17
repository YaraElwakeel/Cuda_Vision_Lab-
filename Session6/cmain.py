from Dataset import Dataset
from conditonalDCGAN import Generator, Discriminator
from model_wrapper import Trainer
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import shutil
import torch



# Hyperparameters
latent_dim = 128
num_classes = 3
image_size = 32
num_channels = 3  # AFHQ images are RGB
base_channels = 32
batch_size = 64
epochs = 50
learning_rate = 0.0002
beta1 = 0.5




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

@torch.no_grad()
def generate(generator, label,N=64):
    """ Generating a bunch of images using current state of generator """
    generator.eval()
    latent = torch.randn(N, latent_dim, 1, 1).to("cuda:0")
    # print(latent.shape, label.shape)
    imgs = generator(latent, labels)
    imgs = imgs * 0.5 + 0.5
    return imgs
        


root_dir = "/home/user/zafara1/Cuda_Vision_Lab-/Session5/Solution/data/afhq"

# Initialize dataset
train_dataset = Dataset(root_dir=root_dir, split="train", transform=transforms["train"])
test_dataset = Dataset(root_dir=root_dir, split="val", transform=transforms["val"])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=8,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=8,pin_memory=True)
print("test_dataset per epoch",next(iter(test_loader))[0].size())


# Initialize models, loss function, and optimizers
generator = Generator(latent_dim, num_channels, base_channels, num_classes).to("cuda:0")
discriminator = Discriminator(num_channels, 1, base_channels, num_classes=num_classes).to("cuda:0")
criterion = torch.nn.BCELoss()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Training loop
iter=0
for epoch in range(epochs):
    
    for real_images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch_size = real_images.size(0)
        real_images, labels = real_images.to("cuda:0"), labels.to("cuda:0")
        
        # Create labels for real and fake images
        real_labels = torch.ones(batch_size, 1, 1, 1).to("cuda:0")
        fake_labels = torch.zeros(batch_size, 1, 1, 1).to("cuda:0")
        
        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Real images
        real_outputs = discriminator(real_images, labels)
        real_loss = criterion(real_outputs, real_labels)
        
        
        # Fake images
        z = torch.randn(batch_size, latent_dim, 1, 1).to("cuda:0")
        fake_images = generator(z, labels)
        fake_outputs = discriminator(fake_images.detach(), labels)
        fake_loss = criterion(fake_outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        
        fake_outputs = discriminator(fake_images, labels)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

        iter+=1
        if(iter % 200==0):    
            imgs = generate(generator=generator, label=labels)
            grid = torchvision.utils.make_grid(imgs, nrow=8)
            grid = torchvision.utils.make_grid(grid, nrow=8)
            torchvision.utils.save_image(grid, os.path.join(os.getcwd(), "imgs", "training", f"conditional_imgs_{iter}.png"))
        
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")