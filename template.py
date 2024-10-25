import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "detmood"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/constant/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/stage_01_data_ingestion.py",
    f"src/{project_name}/pipelines/stage_02_data_validation.py",
    f"src/{project_name}/pipelines/stage_03_data_transformation.py",
    f"src/{project_name}/pipelines/stage_04_model_trainer.py",
    f"src/{project_name}/pipelines/stage_04_model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "research/trials.ipynb",
    "main.py",
    "requirements.txt",
    "setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file {filename}")
    else:
        logging.info(f"{filename} is already created")
