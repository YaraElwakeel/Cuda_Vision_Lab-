import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class dataset(Dataset):
    def __init__(self, root, transform=None, split="train", test_size=0.3, random_state=None):
        # Initialize dataset with root directory, transformations, and split configuration
        self.root_dir = root
        self.transform = transform
        self.split = split

        # Lists to store image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        
        # Create a mapping from class name to index (numeric labels)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root)))}
        
        # Iterate through each class folder and gather image paths and labels
        for cls_name, idx in self.class_to_idx.items():
            cls_folder = os.path.join(root, cls_name)  # Full path to class folder
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))  # Add image path
                self.labels.append(idx)  # Add corresponding label

        # Split the data into train and test sets
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.image_paths, self.labels, test_size=test_size, random_state=random_state, stratify=self.labels
        )
        
        # Select train or test data based on the split argument
        if self.split == "train":
            self.image_paths = train_paths
            self.labels = train_labels
        elif self.split == "test":
            self.image_paths = test_paths
            self.labels = test_labels
        else:
            raise ValueError("Invalid split! Use 'train' or 'test'.")
    
    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path and corresponding label for a given index
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if specified (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)

        # Return the image and its corresponding label
        return image, label
