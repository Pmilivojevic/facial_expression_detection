import os
import pandas as pd
from src.detmood.entity.config_entity import DataValidationConfig

class DataValidation:
    """
    A class for performing data validation on a dataset to ensure it meets 
    specific schema and consistency requirements.

    Attributes:
    ----------
    config : DataValidationConfig
        Configuration object containing settings for data validation, such as 
        schema, dataset folder path, and status file path.
    """
    
    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataTransformation class with the given configuration.

        Parameters:
        ----------
        config : DataTransformationConfig
            The configuration object containing transformation parameters, such as 
            filter size, dataset folder path, and labels file paths.
        """
        
        self.config = config
    
    def validate_dataset(self)-> bool:
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
        
        try:
            validation_status = None

            data_df = pd.read_csv(self.config.dataset_labels)
            all_cols = list(data_df.columns)

            for col in self.config.all_schema:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                
                    return validation_status
                
                elif validation_status == None:
                    validation_status = True

            for img in os.listdir(self.config.dataset_folder):
                if img not in list(data_df[all_cols[0]]):
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                
                    return validation_status
                
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                        
            return validation_status
        
        except Exception as e:
            raise e
