import os
from pathlib import Path
from src.detmood.utils.main_utils import get_size, create_directories
from src.detmood.entity.config_entity import DataIngestionConfig
from src.detmood import logger
import zipfile
import subprocess

class DataIngestion:
    """
    A class used for handling data ingestion processes, including downloading 
    and extracting datasets.

    Attributes:
    ----------
    config : DataIngestionConfig
        Configuration object containing settings for data ingestion such as
        source URL, local data file path, and root directory.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with a given configuration.

        Parameters:
        ----------
        config : DataIngestionConfig
            The configuration object containing data ingestion parameters 
            such as the source URL, local file path, and extraction directory.
        """
        
        self.config = config
    
    def download_dataset(self):
        """
        Downloads the dataset from the specified source URL to a local file 
        path if it does not already exist.

        If the file already exists, logs the current file size. Otherwise, 
        downloads the file using the `wget` command.

        Logs:
        -----
        - Dataset download information if the download occurs.
        - File size if the dataset already exists.
        """
        
        if not os.path.exists(self.config.local_data_file):
            info = subprocess.run(
                f"wget '{self.config.source_URL}' -O {self.config.local_data_file}",
                shell=True
            )
            logger.info(f"Dataset downloaded with folowing info: \n{info}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
        Extracts the downloaded ZIP file to the specified root directory.

        After extracting the contents, deletes the original ZIP file to save 
        space.

        Logs:
        -----
        - A message indicating successful extraction of the ZIP file.
        - A message indicating successful deletion of the ZIP file.
        """
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.root_dir)
        logger.info("Zip file extacted!")
        
        os.remove(self.config.local_data_file)
        logger.info("Zip file removed!")
