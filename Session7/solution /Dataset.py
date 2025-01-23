import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image
import pandas as pd


class LFWtripletDataset(Dataset):
    
    def __init__(self, root_dir=None, split = None, transform=None):
        """
        Args:
            root_dir (str): Directory containing dataset.
            split(str) :{train, val} 
            transform (callable, optional): Transformation to apply to the images.
            
        """
        self.transform = transform
        self.root_dir = root_dir

        self.image_paths = []
        self.labels = []

        csv_file_name = "pairs.csv"  

        # Construct the full path for the file
        csv_file_path = os.path.join(self.root_dir, csv_file_name)
        df = pd.read_csv(csv_file_path)
        df = self.modify_CSV(df)
        print(df)
        
        for index, row in df.iterrows():
            image_dir_0 = os.path.join(root_dir, row.iloc[0] , row.iloc[1])
            image_dir_1 = os.path.join(row.iloc[2] , row.iloc[3])
            
            image_0 = Image.open(image_dir_0).convert("L")
            image_1 = Image.open(image_dir_1).convert("L")
            
            
            
        
        
        
    def modify_CSV(self,df):
        # Check for NaN in "imagenum2"
        mask = df.iloc[:, 3].isna()

        # Apply the modifications:
        df.loc[mask, df.columns[3]] = df.loc[mask, df.columns[2]]  # Shift "imagenum2" to "Unnamed: 3"
        df.loc[mask, df.columns[2]] = df.loc[mask, df.columns[0]]  # Copy "name" to "imagenum2"

        # Now, add leading zeros to the third column (imagenum2)
        df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: f"{int(x):04d}" )
        df.iloc[:, 3] = df.iloc[:, 3].apply(lambda x: f"{int(x):04d}" )
        
        df.iloc[:,1] = df.iloc[:, 0].astype(str) + "_" + df.iloc[:, 1].astype(str) + ".jpg"
        df.iloc[:,3] = df.iloc[:, 2].astype(str) + "_" + df.iloc[:, 3].astype(str) + ".jpg"
        return df 


        # # Open and process the file
        # with open(csv_file_path, "r") as f:
        #     for line in f:
        #         print(line.strip())  # Incremental processing here
        
        # for subdir in os.listdir(root_dir):
        #     print(subdir)
        #     if subdir == "pairs.csv":
                
    # def __len__(self):
    #     return len(self.images)

    # def __getitem__(self, idx):
    #     image1 = self.images[idx, 0]  # First image in pair
    #     image2 = self.images[idx, 1]  # Second image in pair
    #     label = self.labels[idx]  # 1 if same person, 0 otherwise

    #     # Convert grayscale (H, W) to (C, H, W)
    #     image1 = np.expand_dims(image1, axis=0)
    #     image2 = np.expand_dims(image2, axis=0)

    #     # Convert images to PIL for transformation
    #     image1 = Image.fromarray(image1[0])  # Convert to PIL Image
    #     image2 = Image.fromarray(image2[0])  # Convert to PIL Image

    #     # Apply transformations (like resizing, normalizing)
    #     if self.transform:
    #         image1 = self.transform(image1)
    #         image2 = self.transform(image2)

    #     return image1, image2, torch.tensor(label, dtype=torch.float32)
