from src.detmood.constant import *
from src.detmood.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.detmood.utils.main_utils import create_directories, read_yaml

class ConfigurationManager:
    """
    A class to manage configuration settings for different stages of a data 
    pipeline, including data ingestion, validation, transformation, and model 
    training.

    Attributes:
    ----------
    config : dict
        Configuration settings loaded from the configuration file.
    params : dict
        Parameters loaded from the parameters file.
    schema : dict
        Schema definitions loaded from the schema file.
    """
    
    def __init__(
        self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH,
        schema_file_path = SCHEMA_FILE_PATH
    ):
        """
        Initializes the ConfigurationManager class by loading the configuration, 
        parameters, and schema from the specified file paths. It also creates 
        necessary directories for the artifacts.

        Parameters:
        ----------
        config_file_path : str, optional
            Path to the configuration YAML file. Defaults to CONFIG_FILE_PATH.
        params_file_path : str, optional
            Path to the parameters YAML file. Defaults to PARAMS_FILE_PATH.
        schema_file_path : str, optional
            Path to the schema YAML file. Defaults to SCHEMA_FILE_PATH.
        """
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves and prepares the data ingestion configuration settings.

        The method reads the data ingestion configuration from the loaded 
        configuration file, creates necessary directories, and returns a 
        DataIngestionConfig object.

        Returns:
        -------
        DataIngestionConfig
            An object containing data ingestion configuration settings.
        """
        
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
    
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )
        
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves and prepares the data validation configuration settings.

        The method reads the data validation configuration from the loaded 
        configuration file, including schema information, creates necessary 
        directories, and returns a DataValidationConfig object.

        Returns:
        -------
        DataValidationConfig
            An object containing data validation configuration settings.
        """
        
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            dataset_folder=config.dataset_folder,
            dataset_labels=config.dataset_labels,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Generates and returns a DataTransformationConfig object for data transformation.

        This method reads the dataset validation status from a file, ensures that the
        transformed dataset directory is created, and initializes a DataTransformationConfig
        instance with the relevant paths and parameters for data transformation.

        Returns:
            DataTransformationConfig: A configuration object containing paths and parameters
                                      necessary for data transformation and validation setup.
        """
        
        config = self.config.data_transformation
        dataset_val_status_file = self.config.data_validation.STATUS_FILE
        
        with open(dataset_val_status_file, 'r') as f:
            status = f.read()
        
        status = bool(str.split(status)[-1])
        
        create_directories([config.transformed_dataset])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            dataset_folder=config.dataset_folder,
            transformed_dataset=config.transformed_dataset,
            dataset_labels_src=config.dataset_labels_src,
            dataset_labels=config.dataset_labels,
            params=self.params,
            dataset_val_status=status
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves and prepares the model training configuration settings.

        The method reads the model training configuration from the loaded 
        configuration file, including model parameters, creates necessary 
        directories for saving models and figures, and returns a 
        ModelTrainerConfig object.

        Returns:
        -------
        ModelTrainerConfig
            An object containing model training configuration settings.
        """

        config = self.config.model_trainer
        params = self.params.model
        
        create_directories([config.models, config.figures])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            models=config.models,
            figures=config.figures,
            dataset_folder=config.dataset_folder,
            dataset_labels=config.dataset_labels,
            model_params=params
        )
        
        return model_trainer_config
