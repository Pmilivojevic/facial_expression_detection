from src.detmood.constant.dataset_preparation import CustomImageDataset
from src.detmood.entity.config_entity import ModelTrainerConfig
from src.detmood.utils.main_utils import save_json
from src.detmood.constant import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

class ModelTrainer:
    """
    A class for training image classification models with k-fold cross-validation.

    This class handles dataset preparation, model initialization, training, validation,
    saving the best-performing models, and plotting training/validation metrics.
    
    Attributes:
        config (ModelTrainerConfig): Configuration object with training parameters, file paths, and model parameters.
    """
    
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer with the specified configuration.

        Args:
            config (ModelTrainerConfig): Configuration object with paths and model parameters.
        """
        
        self.config = config
    
    def dataset_folds_preparation(self):
        """
        Prepares the dataset with image transformations and initializes stratified k-fold splits.

        Returns:
            tuple: A CustomImageDataset instance and a StratifiedKFold object.
        """
        
        transform = transforms.Compose([
            transforms.Resize((
                self.config.model_params.img_in_size,
                self.config.model_params.img_in_size
            )),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(translate=(0.1, 0.1), degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = CustomImageDataset(
            self.config.dataset_labels,
            self.config.dataset_folder,
            self.config.model_params.data_aug_size,
            transform=transform
        )
        
        skf = StratifiedKFold(
            n_splits=self.config.model_params.num_folds,
            shuffle=True,
            random_state=42
        )
        
        return dataset, skf
    
    def model_preparation(self, device):
        """
        Prepares the EfficientNet model for training by modifying the classifier layer to match the 
        number of output classes.

        Returns:
            nn.Module: The modified EfficientNet model.
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
        
        return model.to(device)
    
    def validation(
            self,
            device,
            fold,
            model,
            criterion,
            val_loader,
            val_losses,
            val_accuracies,
            epoch,
            best_val_loss
        ):
        """
        Performs the validation step, computes loss and accuracy, and saves the model if validation
        loss improves. Generates a confusion matrix plot.

        Args:
            device (str): Device to perform validation on ('cuda' or 'cpu').
            fold (int): Current fold number in cross-validation.
            model (nn.Module): The model being validated.
            criterion (nn.Module): Loss function used during validation.
            val_loader (DataLoader): DataLoader for the validation dataset.
            val_losses (list): List tracking validation loss per epoch.
            val_accuracies (list): List tracking validation accuracy per epoch.
            epoch (int): Current epoch within the training cycle.
            best_val_loss (float): Minimum validation loss recorded across epochs.

        Returns:
            tuple: 
                - Updated lists of validation loss and accuracy per epoch.
                - Average validation loss for the current epoch.
                - Validation accuracy for the current epoch.
                - Lists of predictions and true labels.
        """
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            print('Validation process...')
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
            model_path = os.path.join(self.config.models, f'efficientnet_fold_{fold + 1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Saved Best Model for Fold {fold + 1} at Epoch {epoch + 1}')
            
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=range(self.config.model_params.num_classes),
                yticklabels=range(self.config.model_params.num_classes)
            )
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix for Fold {fold + 1}')
            plt.savefig(os.path.join(self.config.figures, f'cm_fold_{fold + 1}.png'))
        
        return val_losses, val_accuracies, avg_val_loss, val_accuracy, all_preds, all_labels
    
    def train_plot(self, range, train_matric, val_matric, train_label, val_label, fold):
        """
        Plots the training and validation metrics (loss or accuracy) over epochs for the given fold.

        Args:
            range (iterable): Range of epochs to plot along the x-axis.
            train_matric (list): Metric values for training data.
            val_matric (list): Metric values for validation data.
            train_label (str): Label for the training metric plot (e.g., 'Train Loss').
            val_label (str): Label for the validation metric plot (e.g., 'Validation Loss').
            fold (int): Current fold number for cross-validation.
        """
        
        plt.figure(figsize=(12, 6))
        plt.plot(range, train_matric, label=train_label)
        plt.plot(range, val_matric, label=val_label)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Train/validation Loss for Fold {fold + 1}')
        plt.legend()
        plt.savefig(os.path.join(self.config.figures, f'Train_Val_{str.split(train_label)[-1]}_Fold_{fold + 1}.png'))
    
    def train(self):
        """
        Orchestrates the training process across k-fold cross-validation.

        This method:
        - Prepares data folds
        - Trains and validates the model on each fold
        - Logs training and validation metrics
        - Saves the best-performing models and generates plots of training and validation metrics

        Returns:
            None
        """
        
        gc.collect()
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: ', device)
        
        dataset, skf = self.dataset_folds_preparation()
        
        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(dataset.balanced_frame, dataset.balanced_frame['label']))):
            print(f'Fold {fold + 1}/{self.config.model_params.num_folds}')
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.model_params.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.model_params.batch_size,
                shuffle=False
            )
            
            model = self.model_preparation(device)
            
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.model_params.lr)
            
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            best_val_loss = float('inf')
            
            for epoch in tqdm(range(self.config.model_params.num_epochs)):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0
                
                for images, labels in tqdm(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    accuracy = 100*((predicted == labels).sum().item()/labels.size(0))
                    sys.stdout.write("train_loss:%.4f - train_accuracy:%.4f" %(loss.item(), accuracy))
                    sys.stdout.flush()
                    
                avg_train_loss = running_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                train_accuracy = 100 * correct_train / total_train
                train_accuracies.append(train_accuracy)
                
                val_losses, val_accuracies, avg_val_loss, val_accuracy, all_preds, all_labels = self.validation(
                    device,
                    fold,
                    model,
                    criterion,
                    val_loader,
                    val_losses,
                    val_accuracies,
                    epoch,
                    best_val_loss
                )
                
                print(f'Epoch [{epoch+1}/{self.config.model_params.num_epochs}], '
                      f'Loss: {avg_train_loss:.4f}, '
                      f'Validation Loss: {avg_val_loss:.4f}, '
                      f'Train Accuracy: {train_accuracy:.2f}%, '
                      f'Validation Accuracy: {val_accuracy:.2f}%')
                
            report = classification_report(
                all_labels,
                all_preds,
                target_names=[f'{mood}' for mood in MOOD_DICT.keys()],
                output_dict=True
            )
            
            save_json(
                path=os.path.join(self.config.figures, f'metrics_report_fold{fold}.json'),
                data=report
            )
            
            epochs_range = range(1, self.config.model_params.num_epochs + 1)
            
            self.train_plot(
                epochs_range,
                train_losses,
                val_losses,
                'Train Loss',
                'Validation Loss',
                fold
            )
            
            self.train_plot(
                epochs_range,
                train_accuracies,
                val_accuracies,
                'Train Accuracy',
                'Validation Accuracy',
                fold
            )
            
            print(f'Finished fold {fold + 1}/{self.config.model_params.num_folds}\n')
        
        print('Training completed.')
