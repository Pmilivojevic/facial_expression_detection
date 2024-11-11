from src.detmood.config.configuration import ConfigurationManager
from src.detmood.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    """
    A pipeline to orchestrate the data transformation and preparation process.

    This class initializes the configuration and transformation settings,
    performs data preprocessing, and prepares the dataset splits for training.

    Methods:
        main: Executes the data transformation and returns the prepared dataset and splits.
    """
    
    def __init__(self):
        """
        Initializes the DataTransformationTrainingPipeline class.
        
        Sets up any necessary attributes or configurations required for the data transformation
        pipeline.
        """
        
        pass
    
    def main(self):
        """
        Main method to execute the data transformation pipeline.

        This method retrieves configuration settings, initializes the DataTransformation class,
        applies the transformation process, and returns the transformed dataset and fold splits
        for model training.

        Returns:
            Tuple[CustomImageDataset, Iterator]: The dataset after transformations and the splits
            for cross-validation.
        """
        
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        dataset, splits = data_transformation.transformation_compose()
        
        return dataset, splits
