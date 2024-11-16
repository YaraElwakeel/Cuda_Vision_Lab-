import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class dataset(Dataset):
    def __init__(self, root, transform=None,split="train", test_size=0.3, random_state=None):

        self.root_dir = root
        self.transform = transform
        self.split = split

        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root)))}
        
        for cls_name, idx in self.class_to_idx.items():
            cls_folder = os.path.join(root, cls_name)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(idx)

        train_paths, test_paths, train_labels, test_labels = train_test_split(
                self.image_paths, self.labels, test_size=test_size, random_state=random_state, stratify=self.labels
            )
        
        # Select the subset based on the split argument
        if self.split == "train":
            self.image_paths = train_paths
            self.labels = train_labels
        elif self.split == "test":
            self.image_paths = test_paths
            self.labels = test_labels
        else:
            raise ValueError("Invalid split! Use 'train' or 'test'.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label