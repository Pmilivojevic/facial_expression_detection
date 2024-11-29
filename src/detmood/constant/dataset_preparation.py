import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import random

class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading and balancing image data for training.

    This dataset class reads image file paths and labels from a CSV file, balances the dataset 
    by augmenting minority classes to a specified size, and applies transformations if provided.
    
    Attributes:
        data_frame (DataFrame): Dataframe containing the initial image paths and labels.
        root_dir (str): Directory where images are stored.
        aug_size (int): Size to which each class should be augmented to balance the dataset.
        transform (callable, optional): Optional transform to be applied on an image.
        balanced_frame (DataFrame): DataFrame containing balanced image paths and labels.
    """
    
    def __init__(self, csv_file, root_dir, aug_size, phase, transform=None):
        """
        Initializes CustomImageDataset with data file, directory, augmentation size, and transform.

        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            root_dir (str): Directory with all images.
            aug_size (int): Target size for each class after augmentation.
            transform (callable, optional): Transform to be applied on an image.
        """
        
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.aug_size = aug_size
        self.transform = transform
        self.phase = phase
        self.balanced_frame = self.balance_data()
    
    def balance_data(self):
        """
        Balances the dataset by augmenting classes to the specified target size.

        Each class is duplicated or augmented through sampling to ensure equal representation 
        across classes as per the target `aug_size`.

        Returns:
            DataFrame: A balanced DataFrame with equal representation of each class.
        """
        
        target_class_counts = {
            label: self.aug_size * self.data_frame['label'].value_counts().max() for label in self.data_frame['label'].unique()
        }
        
        balanced_data = []
        for label, count in target_class_counts.items():
            class_data = self.data_frame[self.data_frame['label'] == label]
            class_samples = class_data.values.tolist()

            balanced_data.extend(class_samples)

            aug_count = count - len(class_samples)
            if aug_count > 0:
                balanced_data.extend(random.choices(class_samples, k=aug_count))
        
        images = []
        labels = []
        for item in balanced_data:
            images.append(item[0])
            labels.append(item[1])
            
        balanced_dict = {
            'image': images,
            'label': labels
        }
        
        balanced_frame = pd.DataFrame.from_dict(balanced_dict)
        
        return balanced_frame

    def __len__(self):
        """
        Returns the total number of samples in the balanced dataset.

        Returns:
            int: The length of `balanced_frame`, representing the number of images in the dataset.
        """
        if self.phase == 'train':
            return len(self.balanced_frame)
        elif self.phase == 'test':
            return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label by index, applies transformations if
        available.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (image, label) where `image` is the transformed image tensor and `label`
            is the class label.
        """
        
        if self.phase == 'train':
            label = self.balanced_frame.iloc[idx, 1]
            img_path = os.path.join(self.root_dir, self.balanced_frame.iloc[idx, 0])
            img = Image.open(img_path).convert('RGB')
        elif self.phase == 'test':
            label = self.data_frame.iloc[idx, 1]
            img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
            img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
