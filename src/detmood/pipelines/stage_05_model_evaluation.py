from src.detmood.config.configuration import ConfigurationManager
from src.detmood.components.model_evaluation import ModelEvaluation

class ModelEvaluationTrainingPipeline:
    """
    Orchestrates the model evaluation process, including configuration setup 
    and invoking the evaluation process for trained models.

    This class is responsible for managing the evaluation pipeline, which:
    - Reads configuration settings.
    - Initializes the model evaluation process.
    - Executes the evaluation of models on the test dataset.
    """
    
    def __init__(self):
        """
        Initializes the ModelEvaluationTrainingPipeline class.

        Currently, no attributes are initialized in the constructor, 
        as the pipeline is designed to handle setup dynamically in the `main` method.
        """
        
        pass
    
    def main(self):
        """
        Main method for executing the model evaluation pipeline.

        This method performs the following steps:
        1. Reads the configuration for model evaluation using the `ConfigurationManager`.
        2. Initializes the `ModelEvaluation` component with the loaded configuration.
        3. Invokes the `test` method from `ModelEvaluation` to evaluate all models 
           on the test dataset and generate performance reports.

        Side Effects:
            - Saves evaluation reports (classification reports and confusion matrices) 
              to the configured output directory.
        """
        
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.test()