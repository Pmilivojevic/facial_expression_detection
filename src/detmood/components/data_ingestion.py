import os
from pathlib import Path
from src.detmood.utils.main_utils import get_size, create_directories
from src.detmood.entity.config_entity import DataIngestionConfig
from src.detmood import logger
import zipfile
import subprocess

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_dataset(self):
        if not os.path.exists(self.config.local_data_file):
            info = subprocess.run(
                f"wget '{self.config.source_URL}' -O {self.config.local_data_file}",
                shell=True
            )
            logger.info(f"Dataset downloaded with folowing info: \n{info}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.root_dir)
        logger.info("Zip file extacted!")
        
        os.remove(self.config.local_data_file)
        logger.info("Zip file removed!")
