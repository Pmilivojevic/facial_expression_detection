from src.detmood.constant import MOOD_DICT
from src.detmood.constant.dataset_preparation import CustomImageDataset
from src.detmood.entity.config_entity import ModelEvaluationConfig
from src.detmood.utils.main_utils import save_json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelEvaluation:
    """
    Handles the evaluation of trained models on a test dataset.

    Attributes:
        config (ModelEvaluationConfig): Configuration object containing paths and parameters for
                                        evaluation.
    """
    
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation class.

        Args:
            config (ModelEvaluationConfig): Configuration object containing paths and parameters
                                            for evaluation.
        """
        
        self.config = config
    
    def prepare_dataset(self):
        """
        Prepares the test dataset for evaluation by applying preprocessing transforms.

        The preprocessing includes resizing, normalizing, and converting images to tensors.
        
        Returns:
            DataLoader: A PyTorch DataLoader object for the test dataset, which can be iterated
                        over in batches.
        """
        
        test_transform = transforms.Compose([
            transforms.Resize((
                self.config.model_params.img_in_size,
                self.config.model_params.img_in_size
            )),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = CustomImageDataset(
            self.config.test_labels,
            self.config.test_data_path,
            1,
            'test',
            test_transform
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=self.config.model_params.batch_size,
            shuffle=False
        )
        
        return test_loader
    
    def model_preparation(self, model_path, device):
        """
        Loads a trained model, prepares it for evaluation, and sets it to evaluation mode.

        The model's classifier head is adjusted to match the number of output classes.

        Args:
            model_path (str): Path to the trained model file.
            device (str): Device to use for computation ('cpu' or 'cuda').

        Returns:
            torch.nn.Module: The prepared PyTorch model ready for evaluation.
        """
        
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        model.classifier[1] = nn.Sequential(
            nn.Linear(
                in_features=1280,
                out_features=512
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=512,
                out_features=self.config.model_params.num_classes
            )
        )
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model.to(device)
    
    def test(self):
        """
        Evaluates all trained models in the models directory on the test dataset.

        This method performs the following steps:
        - Prepares the test dataset.
        - Iterates through all models in the `models_path` directory.
        - For each model, generates predictions on the test dataset.
        - Calculates and saves evaluation metrics (classification report and confusion matrix).

        The results for each model are saved in the `stats` directory as:
        - JSON files containing precision, recall, F1-scores, and other metrics.
        - PNG images of confusion matrices.
        """
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for model_name in tqdm(os.listdir(self.config.models_path)):
            model_path = os.path.join(self.config.models_path, model_name)
            model = self.model_preparation(model_path, device)
            
            test_loader = self.prepare_dataset()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            report = classification_report(
                all_labels,
                all_preds,
                target_names=[f'{mood}' for mood in MOOD_DICT.keys()],
                output_dict=True
            )
            
            save_json(
                path=os.path.join(
                    self.config.stats,
                    f"{str.split(model_name, '.')[0]}_report.json"
                ),
                data=report
            )
            
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=list(MOOD_DICT.keys()),
                yticklabels=list(MOOD_DICT.keys())
            )
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f"Confusion Matrix for {str.split(model_name, '.')[0]}")
            plt.savefig(os.path.join(
                self.config.stats,
                f"cm_{str.split(model_name, '.')[0]}.png"
            ))
