import os 
import pickle 
from PIL import Image 
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd 

class Freiburg_dataset(Dataset):
    def __init__(self,root_dir= None,split= "train"):
        super().__init__()
        """_Args_
            root_dir(str): Directory containing the dataset 
            split(str): split type (train, test) 
        """
        self.root_dir = os.path.join(root_dir,split)
        if split == "train":
            png_files = sorted(list(Path(self.root_dir).rglob("*.png")))
            data_df = pd.DataFrame({"file_path": png_files})
            data_df["seq_name"] = data_df["file_path"].apply(lambda x: Path(x).parts[-4])
            data_df[["seq_number", "dayOrnight"]] = data_df.pop("seq_name").str.extract(r"seq_(\d+)_(\w+)")
            data_df["image_type"] = data_df["file_path"].apply(lambda x: Path(x).parts[-2])                                            
            data_df["image_num"] = data_df["file_path"].apply(lambda x: "_".join(Path(x).stem.rsplit("_", 2)[-2:]))
            self.data = data_df.groupby(["image_num"])
            self.group_nums= list(self.data.groups.keys())
            
    def __len__(self):
        return self.data.ngroups
    
    def __getitem__(self, idx):
        
        scene_num = self.group_nums[idx]
        group_df = self.data.get_group(scene_num)
        
        # Load images
        ir_aligned = group_df.loc[group_df['image_type'] == "fl_ir_aligned", "file_path"].values[0]
        rgb = group_df.loc[group_df['image_type'] == "fl_rgb", "file_path"].values[0]
        rgb_labels = group_df.loc[group_df['image_type'] == "fl_rgb_labels", "file_path"].values[0]

        img_ir_aligned = Image.open(ir_aligned)
        print(img_ir_aligned)
        img_rgb = Image.open(rgb)
        img_rgb_labels = Image.open(rgb_labels)
        

        ## Apply transformations
        # if self.transform:
        #     image = self.transform(image)
        # The line `return rgb` in the `__getitem__` method of the `Freiburg_dataset` class is
        # returning the path to the RGB image file corresponding to the given index `idx`.
        # return (img_ir_aligned, img_rgb,img_rgb_labels)
        return  rgb
        
        
        