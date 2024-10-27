import os
import cv2
from tqdm import tqdm
import pandas as pd
from src.detmood.constant import MOOD_DICT
from src.detmood.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def equalize_histogram(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
        img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return img_eq
    
    def noise_reduction(self, img):
        img_filt = cv2.medianBlur(img, self.config.median_filter_size)
        
        return img_filt
    
    def labels_csv_transform(self):
        labels_df = pd.read_csv(self.config.dataset_labels_src)
        
        for ind in labels_df.index:
            labels_df[ind, 'label'] = MOOD_DICT[labels_df[ind, 'label']]
        
        labels_df.to_csv(self.config.dataset_labels, index=False)
    
    def transformation_compose(self):
        for img_name in tqdm(os.listdir(self.config.dataset_folder)):
            img = cv2.imread(os.path.join(self.config.dataset_folder, img_name))
            img_eq = self.equalize_histogram(img)
            img_filt = self.noise_reduction(img_eq)
            
            cv2.imwrite(os.path.join(self.config.transformed_dataset, img_name), img_filt)
        
        self.labels_csv_transform()
