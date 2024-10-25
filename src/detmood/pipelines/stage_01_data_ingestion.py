from src.detmood.config.configuration import ConfigurationMananger
from src.detmood.components.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationMananger()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_dataset()
        data_ingestion.extract_zip_file()
