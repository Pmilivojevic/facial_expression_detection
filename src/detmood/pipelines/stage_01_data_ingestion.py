from src.detmood.config.configuration import ConfigurationManager
from src.detmood.components.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    """
    A class to manage the data ingestion process for the training pipeline.

    This class is responsible for orchestrating the data ingestion steps, 
    including downloading the dataset and extracting it for further use 
    in the training pipeline.
    """
    
    def __init__(self):
        """
        Initializes the DataIngestionTrainingPipeline instance.
        
        Currently, no parameters are required, but this method can be 
        extended in the future to include configuration or other dependencies.
        """
        
        pass
    
    def main(self):
        """
        The main entry point for the data ingestion pipeline.

        This method performs the following steps:
        1. Loads configuration settings using the ConfigurationManager.
        2. Initializes the DataIngestion class with the loaded configuration.
        3. Downloads the dataset from the specified source URL.
        4. Extracts the downloaded dataset from a zip file for further processing.
        """
        
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_dataset()
        data_ingestion.extract_zip_file()
