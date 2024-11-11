from src.detmood.config.configuration import ConfigurationManager
from src.detmood.components.model_trainer import ModelTrainer

class ModelTrainerTrainingPipeline:
    """
    A class to manage the model training process for the training pipeline.

    This class is responsible for orchestrating the steps involved in training 
    a machine learning model, including data loading, model training, 
    evaluation, and saving the trained model.
    """
    
    def __init__(self):
        """
        Initializes the ModelTrainerTrainingPipeline instance.

        Currently, no parameters are required, but this method can be 
        extended in the future to include configuration or other dependencies.
        """
        
        pass
    
    def main(self, dataset, splits):
        """
        The main entry point for the model training pipeline.

        This method performs the following steps:
        1. Loads configuration settings using the ConfigurationManager.
        2. Initializes the ModelTrainer class with the loaded configuration.
        3. Executes the training process, which includes training the model 
           on the prepared dataset and evaluating its performance.
        """
        
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config, dataset=dataset, splits=splits)
        model_trainer.train()
