import os
import pandas as pd
from src.detmood.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_dataset(self)-> bool:
        try:
            validation_status = None

            data_df = pd.read_csv(self.config.dataset_labels)
            all_cols = list(data_df.columns)

            for col in self.config.all_schema:
                if col not in all_cols:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                
                    return validation_status
                
                elif validation_status == None:
                    validation_status = True

            for img in os.listdir(self.config.dataset_folder):
                if img not in list(data_df[all_cols[0]]):
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                
                    return validation_status
                
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                        
            return validation_status
        
        except Exception as e:
            raise e
