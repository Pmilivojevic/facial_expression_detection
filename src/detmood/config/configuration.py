from src.detmood.constant import *
from src.detmood.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)
from src.detmood.utils.main_utils import create_directories, read_yaml

class ConfigurationManager:
    def __init__(
        self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH,
        schema_file_path = SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
    
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )
        
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
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
        config = self.config.data_transformation
        params = self.params.transform.noise_reduction
        
        create_directories([config.transformed_dataset])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            dataset_folder=config.dataset_folder,
            transformed_dataset=config.transformed_dataset,
            dataset_labels_src=config.dataset_labels_src,
            dataset_labels=config.dataset_labels,
            median_filter_size=params.median_filter_size
        )
        
        return data_transformation_config
