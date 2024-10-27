import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels 
    from a CSV file and a directory.

    Attributes:
    ----------
    data_frame : pandas.DataFrame
        A DataFrame containing image file names and corresponding labels.
    root_dir : str
        The root directory where the images are stored.
    transform : callable, optional
        Optional transform to be applied on an image.
    """
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Initializes the CustomImageDataset with the given CSV file, root 
        directory, and optional transform.

        Parameters:
        ----------
        csv_file : str
            Path to the CSV file containing image file names and labels.
        root_dir : str
            The root directory where the images are located.
        transform : callable, optional
            Optional transform to be applied to the images.
        """
        
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        -------
        int
            The number of samples in the dataset.
        """
        
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Parameters:
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns:
        -------
        tuple
            A tuple containing the transformed image and its corresponding 
            label.
        """
        
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1]) 

        if self.transform:
            img = self.transform(img)

        return img, label
