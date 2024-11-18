from torchvision.transforms.v2 import CutMix, MixUp
import torch

class DataLoader:
    def __init__(self, ds, batch_size, shuffle, num_classes, augmentation=None, alpha=None):
        # Initialize parameters 
        self.dataloader = torch.utils.data.DataLoader(ds, batch_size, shuffle)
        self.num_classes = num_classes
        self.alpha = alpha
        self.augmentation = augmentation

    def __iter__(self):
        # Select the augmentation method based on input
        if self.augmentation == "CutMix":
            augmentation = CutMix(num_classes=self.num_classes, alpha=self.alpha)
        elif self.augmentation == "MixUp": 
            augmentation = MixUp(num_classes=self.num_classes, alpha=self.alpha)
        else:
            augmentation = None
        
        # Apply augmentation and yield images and labels
        for images, labels in self.dataloader:
            if augmentation:
                images, labels = augmentation((images, labels))
            yield images, labels

    def __len__(self):
        return len(self.dataloader)  # Return number of batches

    def get_trainer(self):
        return self.dataloader  # Return the DataLoader instance
