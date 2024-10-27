import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1]) 

        if self.transform:
            img = self.transform(img)

        return img, label
