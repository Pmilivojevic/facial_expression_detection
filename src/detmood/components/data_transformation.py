import os
import cv2
from tqdm import tqdm
import pandas as pd
from src.detmood.constant import MOOD_DICT
from src.detmood.entity.config_entity import DataTransformationConfig

class DataTransformation:
    """
    A class for handling various data transformation tasks, such as image 
    processing and label transformation.

    Attributes:
    ----------
    config : DataTransformationConfig
        Configuration object that contains the settings for data transformation, 
        including file paths and transformation parameters.
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with the given configuration.

        Parameters:
        ----------
        config : DataTransformationConfig
            The configuration object containing transformation parameters, such as 
            filter size, dataset folder path, and labels file paths.
        """
        
        self.config = config
    
    def equalize_histogram(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
        img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return img_eq
    
    def noise_reduction(self, img):
        """
        Applies histogram equalization to an image to enhance its contrast.

        The method converts the image from BGR to HSV, equalizes the histogram 
        of the V channel, and then converts it back to BGR.

        Parameters:
        ----------
        img : numpy.ndarray
            The input image in BGR format.

        Returns:
        -------
        numpy.ndarray
            The histogram-equalized image in BGR format.
        """
        
        img_filt = cv2.medianBlur(img, self.config.median_filter_size)
        
        return img_filt
    
    def labels_csv_transform(self):
        """
        Transforms the labels in the dataset by mapping them to new values.

        The method reads a CSV file containing labels, transforms the labels 
        according to a predefined dictionary (`MOOD_DICT`), and saves the 
        transformed labels to a new CSV file.

        Uses:
        -----
        - `MOOD_DICT` : A dictionary that maps original labels to new values.

        Side Effects:
        -------------
        - Modifies the contents of the labels file.
        """
        
        labels_df = pd.read_csv(self.config.dataset_labels_src)
        
        for ind in labels_df.index:
            labels_df.loc[ind, 'label'] = MOOD_DICT[labels_df.loc[ind, 'label']]
        
        labels_df.to_csv(self.config.dataset_labels, index=False)
    
    def transformation_compose(self):
        """
        Applies a series of transformations to images in the dataset.

        The method processes each image in the specified dataset folder by 
        performing histogram equalization followed by noise reduction. The 
        transformed images are then saved to the specified output folder.

        After transforming the images, the method transforms the labels CSV 
        file.

        Side Effects:
        -------------
        - Saves the transformed images to the output folder.
        - Modifies the contents of the labels file.

        Uses:
        -----
        - tqdm : For displaying a progress bar during the transformation process.
        """
        
        # if not os.listdir(self.config.dataset_folder):
        for img_name in tqdm(os.listdir(self.config.dataset_folder)):
            img = cv2.imread(os.path.join(self.config.dataset_folder, img_name))
            img_eq = self.equalize_histogram(img)
            img_filt = self.noise_reduction(img_eq)
            
            cv2.imwrite(os.path.join(self.config.transformed_dataset, img_name), img_filt)
        
        self.labels_csv_transform()
        # else:
        #     print("Data transformation already performed!")
