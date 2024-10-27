from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    dataset_folder: Path
    dataset_labels: Path
    STATUS_FILE: Path
    all_schema: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_folder: Path
    transformed_dataset: Path
    dataset_labels_src: Path
    dataset_labels: Path
    median_filter_size: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    models: Path
    figures: Path
    dataset_folder: Path
    dataset_labels: Path
    model_params: dict
