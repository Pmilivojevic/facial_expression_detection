import os
import cv2
from tqdm import tqdm
import pandas as pd
from src.detmood.constant import MOOD_DICT, MOOD_DICT_BENCHMARK
from src.detmood.entity.config_entity import DataTransformationConfig
from torchvision import transforms
from src.detmood.constant.dataset_preparation import CustomImageDataset
from sklearn.model_selection import StratifiedKFold
import shutil

class DataTransformation:
    """
    Handles the transformation and preparation of image data for training.

    This class applies preprocessing steps such as histogram equalization, noise reduction, and
    label transformations. It also prepares the dataset with augmentations and divides it into
    training and validation folds.

    Attributes:
        config (DataTransformationConfig): Configuration containing paths and transformation 
                                           parameters.
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with configuration settings.

        Args:
            config (DataTransformationConfig): A configuration object with necessary paths and
                                               parameters.
        """
        
        self.config = config
    
    def equalize_histogram(self, img):
        """
        Applies histogram equalization to the brightness channel of an image.

        Args:
            img (numpy.ndarray): An image in BGR format.

        Returns:
            numpy.ndarray: The histogram-equalized image in BGR format.
        """
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
        img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return img_eq
    
    def noise_reduction(self, img):
        """
        Reduces noise in an image using median filtering.

        Args:
            img (numpy.ndarray): An image in BGR format.

        Returns:
            numpy.ndarray: The noise-reduced image in BGR format.
        """
        
        img_filt = cv2.medianBlur(
            img,
            self.config.params.transform.noise_reduction.median_filter_size
        )
        
        return img_filt
    
    def labels_csv_transform(self, csv_path, save_path, dict):
        """
        Transforms label values in the CSV to mapped values based on a predefined dictionary
        (MOOD_DICT).

        Reads the CSV file containing labels, updates the labels to standardized mood names,
        and saves the modified file to a new path.
        """
        
        labels_df = pd.read_csv(csv_path)
        
        for ind in labels_df.index:
            labels_df.loc[ind, 'label'] = dict[labels_df.loc[ind, 'label']]
        
        labels_df.to_csv(save_path, index=False)
    
    def ungroup_folder_classes(self):
        """
        Restructures the dataset by ungrouping images from hierarchical class directories
        into a single flat directory within the transformed dataset directory.
        
        Steps:
        - Checks if the transformed dataset directory is empty.
        - Iterates over the main dataset folder structure.
        - Creates directories for train or test chunk in the transformed dataset folder.
        - Copies images from the original hierarchical structure into the respective chunk
          directories.
        - Copies the original train_label.csv file to the transformed dataset location.
        """
        
        if not os.listdir(self.config.transformed_dataset):
            for dir in os.listdir(self.config.dataset_folder):
                os.makedirs(os.path.join(self.config.transformed_dataset, dir), exist_ok=True)
                for class_dir in tqdm(os.listdir(os.path.join(self.config.dataset_folder, dir))):
                    for img in tqdm(os.listdir(os.path.join(
                                        self.config.dataset_folder,
                                        dir,
                                        class_dir
                                    ))):
                        
                        img_path = os.path.join(self.config.dataset_folder, dir, class_dir, img)
                        if dir == 'train':
                            img_file = cv2.imread(img_path)
                            img_eq = self.equalize_histogram(img_file)
                            img_filt = self.noise_reduction(img_eq)
                            
                            cv2.imwrite(os.path.join(
                                self.config.transformed_dataset,
                                dir,
                                img
                            ), img_filt)
                        else:
                            shutil.copy2(
                                os.path.join(self.config.dataset_folder, dir, class_dir, img),
                                os.path.join(self.config.transformed_dataset, dir)
                            )
            
            # shutil.copy2(self.config.dataset_labels_src, self.config.dataset_labels)
    
    def dataset_folds_preparation(self):
        """
        Prepares dataset folds and transformations for cross-validation.

        This method applies several augmentations to the dataset, such as resizing,
        flipping, and rotation, and divides it into cross-validation folds.

        Returns:
            Tuple[CustomImageDataset, Iterator]: Transformed dataset and stratified fold splits.
        """
        
        transform = transforms.Compose([
            transforms.Resize((
                self.config.params.model.img_in_size,
                self.config.params.model.img_in_size
            )),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(translate=(0.1, 0.1), degrees=15),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = CustomImageDataset(
            self.config.dataset_labels,
            # self.config.transformed_dataset,
            os.path.join(self.config.transformed_dataset, 'train'),
            self.config.params.model.data_aug_size,
            'train',
            transform=transform
        )
        
        skf = StratifiedKFold(
            n_splits=self.config.params.model.num_folds,
            shuffle=True,
            random_state=42
        )
        
        splits = skf.split(dataset.balanced_frame, dataset.balanced_frame['label'])
        
        return dataset, splits
    
    def transformation_compose(self):
        """
        Coordinates the complete transformation process including histogram equalization, noise
        reduction, and dataset splitting.

        If data transformation has not been performed, this method applies all preprocessing steps.
        It also updates the label CSV and prepares dataset splits for cross-validation.

        Returns:
            Tuple[CustomImageDataset, Iterator] or None: Prepared dataset and fold splits if
                                                         validation is successful, else None.
        """
        
        # if self.config.dataset_val_status:
        #     if not os.listdir(self.config.transformed_dataset):
        #         for img_name in tqdm(os.listdir(self.config.dataset_folder)):
        #             img = cv2.imread(os.path.join(self.config.dataset_folder, img_name))
        #             img_eq = self.equalize_histogram(img)
        #             img_filt = self.noise_reduction(img_eq)
                    
        #             cv2.imwrite(os.path.join(self.config.transformed_dataset, img_name), img_filt)
                
        #         self.labels_csv_transform()
        #     else:
        #         print("Data transformation already performed!")
            
        #     dataset, splits = self.dataset_folds_preparation()
            
        #     return dataset, splits
        # else:
        #     print("Dataset is not valid!")
        
        if self.config.dataset_val_status:
            self.ungroup_folder_classes()
            self.labels_csv_transform(self.config.dataset_labels_src, self.config.dataset_labels, MOOD_DICT_BENCHMARK)
            self.labels_csv_transform(self.config.test_labels_src, self.config.test_labels, MOOD_DICT_BENCHMARK)
            dataset, splits = self.dataset_folds_preparation()
            
            return dataset, splits
        else:
            print("Dataset is not valid!")
