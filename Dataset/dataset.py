from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pandas as pd
import numpy as np
import os

class VQADataset(Dataset):
    def __init__(self, df : str, image_dir:str):
        """
        args:
        df: path of dataframe
        """
        super().__init__()
        self.df = pd.read_csv(df)
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.df.iloc[idx, 1]))
        prompt= self.df.iloc[idx, 2]
        
        return { 'images': image, 'text': prompt}
        
dataset = VQADataset('data.csv', '')
