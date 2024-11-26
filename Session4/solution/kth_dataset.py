import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class KTH_Dataset(Dataset):
    def __init__(self, root_dir, sequence_length=15, transform=None):
        """
        Args:
            root_dir (str): Directory containing image frames, organized in sequential order.
            sequence_length (int): Number of frames in each subsequence.
            transform (callable, optional): Transformation to apply to the images.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.class_dirs = sorted([d for d in os.listdir(root_dir) 
                                  if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_dirs)}

        # Collect all sequences and their labels
        self.samples = self._load_samples()

        

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        """
        Load all image paths and corresponding labels as sequences.
        """
        samples = []
        for cls in self.class_dirs:
            cls_dir = os.path.join(self.root_dir, cls)
            video_directory = sorted([os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                                  if os.path.isdir(os.path.join(cls_dir, f))])
            # Generate sequences for this class
            for video in video_directory:
                video_dir = os.path.join(cls_dir, video)
                image_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir)
                                      if os.path.isfile(os.path.join(video_dir, f))])

                # Generate sequences for this video
                for i in range(len(image_files) - self.sequence_length + 1):
                    sequence = image_files[i:i + self.sequence_length]
                    samples.append((sequence, self.class_to_idx[cls]))
        return samples


    def __getitem__(self, idx):
        sequence_files, label = self.samples[idx]
        sequence = []

        for file in sequence_files:
            image = Image.open(file).convert("RGB")
            if self.transform:
                image = self.transform(image)
            sequence.append(image)

        # Stack sequence into a tensor of shape (sequence_length, C, H, W)
        sequence = torch.stack(sequence)
        return sequence, label