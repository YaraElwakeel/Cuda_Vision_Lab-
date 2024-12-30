import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, root_dir=None, split = None, transform=None):
        """
        Args:
            root_dir (str): Directory containing image frames.
            transform (callable, optional): Transformation to apply to the images.
        """
        self.transform = transform
        self.root_dir = root_dir

        self.image_paths = []
        self.labels = []

        for subdir in os.listdir(root_dir):
            if subdir == split:
                subdir_path = os.path.join(root_dir, subdir)
                for label,subsubdir in enumerate(os.listdir(subdir_path)):
                    subsubdir_path = os.path.join(subdir_path,subsubdir)
                    for file in os.listdir(subsubdir_path):
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(subsubdir_path, file))
                            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    # def save_samples(self, output_file):
    #     """
    #     Save the samples to a file.
    #     Args:
    #         output_file (str): Path to save the samples.
    #     """
    #     with open(output_file, 'wb') as f:
    #         pickle.dump(self.samples, f)
    
    def __getitem__(self, idx):

        # Load image and label
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Ensure 3-channel RGB
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, label
