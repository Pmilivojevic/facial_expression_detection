from src.detmood.config.configuration import ConfigurationMananger
from src.detmood.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationMananger()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
