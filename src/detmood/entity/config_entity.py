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
    Configuration for data transformation and preparation.

    This configuration defines paths and parameters needed for dataset transformation, 
    including source and destination directories, label paths, and transformation parameters.

    Attributes:
        root_dir (Path): Root directory containing the data.
        dataset_folder (Path): Directory where the original dataset is stored.
        transformed_dataset (Path): Directory where transformed dataset will be saved.
        dataset_labels_src (Path): Path to the source labels for the dataset.
        dataset_labels (Path): Path to the transformed labels file.
        params (dict): Parameters for data transformations and processing.
        dataset_val_status (bool): Flag indicating if data is valid.
    """
    
    root_dir: Path
    dataset_folder: Path
    transformed_dataset: Path
    dataset_labels_src: Path
    dataset_labels: Path
    params: dict
    dataset_val_status: bool


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
