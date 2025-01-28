import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np 
import os 
from PIL import Image
import pandas as pd
import random
import helper

class LFWtripletDataset(Dataset):
    
    def __init__(self, root_dir=None, split = None, transform=None):
        """
        Args:
            root_dir (str): Directory containing dataset.
            split(str){train, test}  : data split type 
            transform (callable, optional): Transformation to apply to the images.
            
        """
        self.transform = transform
        self.root_dir = root_dir
        self.split = split
        
        self.people_df  = None
        self.people_sev_img = None
        self.people_one_img = None
        

        self.triplets = []
        self.triplets_labels = []

        self.splits = {None:"people.csv","train":"peopleDevTrain.csv","test":"peopleDevTest.csv"}
        self.read_data()

    def statistics(self):
        top_people = helper.db_statistics(self.people_df)
        return top_people
        
    def read_data(self):
        
        csv_file_path = os.path.join(self.root_dir, self.splits[self.split])
        self.people_df = pd.read_csv(csv_file_path)  

        if self.split != None:
            # Create a new DataFrame excluding rows with the only one picture
            self.people_sev_img = self.people_df[self.people_df['images'] != 1]
            self.expand_CSV()
            self.people_sev_img= self.add_subdir_images(self.people_sev_img)
            

            # Create another DataFrame containing only the excluded rows
            self.people_one_img = self.people_df[self.people_df['images'] == 1]
            self.people_one_img= self.add_subdir_images(self.people_one_img)
        
    def expand_CSV(self):
        
        self.people_sev_img = self.people_sev_img.loc[self.people_sev_img.index.repeat(self.people_sev_img['images'])].copy()
        self.people_sev_img['images'] = self.people_sev_img.groupby('name').cumcount() + 1


    def add_subdir_images(self,df):
        
        # Now, add leading zeros to images number
        df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: f"{int(x):04d}" )
        
        # add the name to the image number 
        df.iloc[:,1] = df.iloc[:, 0].astype(str) + "_" + df.iloc[:, 1].astype(str) + ".jpg"
        
        return df

                
    def __len__(self):
        return len(self.people_sev_img)

    def __getitem__(self, idx):
        
        image_dir = os.path.join(self.root_dir,'lfw-deepfunneled', 'lfw-deepfunneled')        
        anchor_dir = os.path.join(image_dir, self.people_sev_img.iloc[idx,0] ,  self.people_sev_img.iloc[idx,1])
        anchor_img = Image.open(anchor_dir)
        anchor_name = self.people_sev_img.iloc[idx,0]
        
        
        # sample from all the possible positive samples 
        selected_name = self.people_sev_img.iloc[idx, 0]
        possible_pos = self.people_sev_img[self.people_sev_img['name'] == selected_name].reset_index(drop=True)
        pos_sample = possible_pos.sample(n=1)
        pos_name = pos_sample.iloc[0]["name"]
        pos_dir = os.path.join(image_dir,pos_name,   pos_sample.iloc[0]["images"])
        pos_img = Image.open(pos_dir)     

        # sample from all the possible negative samples 
        neg_sample = self.people_one_img.sample(n=1)
        neg_name = neg_sample.iloc[0]["name"]
        neg_dir = os.path.join(image_dir, neg_name , neg_sample.iloc[0]["images"])
        neg_img = Image.open(neg_dir)
            
        # Apply transformations 
        if self.transform:
            anchor = self.transform(anchor_img)
            positive = self.transform(pos_img)
            negative = self.transform(neg_img)

        return (anchor, positive,negative),(anchor_name,pos_name,neg_name) 
