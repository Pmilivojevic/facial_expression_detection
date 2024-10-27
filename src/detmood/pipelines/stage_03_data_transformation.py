from src.detmood.config.configuration import ConfigurationManager
from src.detmood.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    """
    A class to manage the data transformation process for the training pipeline.

    This class is responsible for orchestrating the data transformation steps, 
    which may include operations like noise reduction and histogram equalization, 
    to prepare the dataset for model training.
    """
    
    def __init__(self):
        """
        Initializes the DataTransformationTrainingPipeline instance.

        Currently, no parameters are required, but this method can be 
        extended in the future to include configuration or other dependencies.
        """
        
        pass
    
    def main(self):
        """
        The main entry point for the data transformation pipeline.

        This method performs the following steps:
        1. Loads configuration settings using the ConfigurationManager.
        2. Initializes the DataTransformation class with the loaded configuration.
        3. Executes the transformation process, which includes various data 
           preprocessing techniques to prepare the dataset for training.
        """
        
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transformation_compose()
