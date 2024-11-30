import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class KTH_Dataset(Dataset):
    def __init__(self, root_dir=None, sequence_length=15, transform=None, use_saved_samples=False):
        """
        Args:
            root_dir (str): Directory containing image frames.
            sequence_length (int): Number of frames in each subsequence.
            transform (callable, optional): Transformation to apply to the images.
            use_saved_samples (bool): Whether to load `samples` from a file.
            samples_file (str): Path to the saved samples file.
        """
        self.sequence_length = sequence_length
        self.transform = transform

        if use_saved_samples:
            # Load pre-saved samples
            with open("samples.pkl", 'rb') as f:
                self.samples = pickle.load(f)
        else:
            self.root_dir = root_dir
            self.class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_dirs)}
            self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        """
        Generate all sequences and their labels.
        """
        samples = []
        for cls in self.class_dirs:
            cls_dir = os.path.join(self.root_dir, cls)
            video_directory = sorted([os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                                       if os.path.isdir(os.path.join(cls_dir, f))])
            for video in video_directory:
                video_dir = os.path.join(cls_dir, video)
                image_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir)
                                      if os.path.isfile(os.path.join(video_dir, f))])
                i=0
                for i in range(len(image_files) - self.sequence_length + 1):
                    if (i%15 == 0):
                        sequence = image_files[i:i + self.sequence_length]
                        samples.append((sequence, self.class_to_idx[cls]))
                        i = i + 1
                    else:
                        i = i+1
                        continue
        return samples

    def save_samples(self, output_file):
        """
        Save the samples to a file.
        Args:
            output_file (str): Path to save the samples.
        """
        with open(output_file, 'wb') as f:
            pickle.dump(self.samples, f)

    def __getitem__(self, idx):
        sequence_files, label = self.samples[idx]
        sequence = []
        for file in sequence_files:
            image = Image.open(file).convert("L") # transfrom to greyscale
            if self.transform:
                image = self.transform(image)
            sequence.append(image)

        # Stack sequence into a tensor of shape (sequence_length, C, H, W)
        sequence = torch.stack(sequence)
        return sequence, label
