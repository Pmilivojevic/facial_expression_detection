from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    A configuration class for data ingestion settings.

    This class holds configuration details required for the data ingestion 
    process, including the root directory, source URL for downloading the 
    dataset, and the path for storing the local data file.

    Attributes:
    ----------
    root_dir : Path
        The root directory where data-related artifacts will be stored.
    source_URL : str
        The URL from which the dataset will be downloaded.
    local_data_file : Path
        The path to the local file where the downloaded dataset will be saved.
    """
    
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    A configuration class for data validation settings.

    This class holds configuration details needed for validating a dataset, 
    such as paths to data directories, labels, and the validation status file, 
    as well as the schema for the data.

    Attributes:
    ----------
    root_dir : Path
        The root directory for the project.
    dataset_folder : Path
        The path to the folder containing the dataset.
    dataset_labels : Path
        The path to the CSV file containing dataset labels.
    STATUS_FILE : Path
        The path to the file where validation status will be recorded.
    all_schema : list
        A list of expected column names in the dataset for schema validation.
    """
    
    root_dir: Path
    dataset_folder: Path
    dataset_labels: Path
    STATUS_FILE: Path
    all_schema: list


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    A configuration class for data transformation settings.

    This class holds configuration details required for the data transformation 
    process, including paths for the original and transformed datasets, as well 
    as settings for image processing such as the median filter size.

    Attributes:
    ----------
    root_dir : Path
        The root directory for the project.
    dataset_folder : Path
        The path to the folder containing the original dataset.
    transformed_dataset : Path
        The path where the transformed dataset will be saved.
    dataset_labels_src : Path
        The path to the source CSV file containing the original dataset labels.
    dataset_labels : Path
        The path to the CSV file where the transformed dataset labels will be saved.
    median_filter_size : int
        The size of the median filter to be applied for noise reduction.
    """
    
    root_dir: Path
    dataset_folder: Path
    transformed_dataset: Path
    dataset_labels_src: Path
    dataset_labels: Path
    median_filter_size: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    A configuration class for model training settings.

    This class holds configuration details required for training a machine 
    learning model, including paths for saving models and figures, as well as 
    parameters related to the training process.

    Attributes:
    ----------
    root_dir : Path
        The root directory for the project.
    models : Path
        The path where the trained model files will be saved.
    figures : Path
        The path where training and validation figures (e.g., loss and accuracy plots) 
        will be saved.
    dataset_folder : Path
        The path to the folder containing the dataset used for training.
    dataset_labels : Path
        The path to the CSV file containing the dataset labels.
    model_params : dict
        A dictionary containing model training parameters such as batch size, learning 
        rate, number of epochs, and the number of classes.
    """
    
    root_dir: Path
    models: Path
    figures: Path
    dataset_folder: Path
    dataset_labels: Path
    model_params: dict
