from src.detmood.config.configuration import ConfigurationManager
from src.detmood.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    """
    A class to manage the data validation process for the training pipeline.

    This class is responsible for orchestrating the data validation steps, 
    ensuring that the dataset adheres to the specified schema and integrity 
    checks before further processing in the training pipeline.
    """
    
    def __init__(self):
        """
        Initializes the DataValidationTrainingPipeline instance.

        Currently, no parameters are required, but this method can be 
        extended in the future to include configuration or other dependencies.
        """
        
        pass
    
    def main(self):
        """
        The main entry point for the data validation pipeline.

        This method performs the following steps:
        1. Loads configuration settings using the ConfigurationManager.
        2. Initializes the DataValidation class with the loaded configuration.
        3. Validates the dataset against the defined schema and checks for 
           data integrity issues.
        """
        
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_dataset()
