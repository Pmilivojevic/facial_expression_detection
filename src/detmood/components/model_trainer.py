from src.detmood.constant.dataset_preparation import CustomImageDataset
from src.detmood.entity.config_entity import ModelTrainerConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

class ModelTrainer:
    """
    A class to handle model training with k-fold cross-validation.

    This class is responsible for preparing dataset folds, training the model with each fold, 
    validating the model, saving the best model checkpoint, and plotting metrics over epochs.

    Attributes:
        config (ModelTrainerConfig): Configuration object with training parameters and paths.
    """
    
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer with the given configuration.

        Args:
            config (ModelTrainerConfig): Configuration object with paths and model parameters.
        """
        
        self.config = config
    
    def dataset_folds_preparation(self):
        """
        Prepares the dataset and stratified k-folds for cross-validation.

        Applies image transformations and initializes a StratifiedKFold splitter.

        Returns:
            tuple: A dataset (CustomImageDataset) instance and a StratifiedKFold object.
        """
        
        transform = transforms.Compose([
            transforms.Resize((
                self.config.model_params.img_in_size,
                self.config.model_params.img_in_size
            )),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = CustomImageDataset(
            self.config.dataset_labels,
            self.config.dataset_folder,
            transform=transform
        )
        
        skf = StratifiedKFold(
            n_splits=self.config.model_params.num_folds,
            shuffle=True,
            random_state=42
        )
        
        return dataset, skf
    
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
        Evaluates the model on the validation set.

        Runs a validation loop to calculate validation loss and accuracy, saves the model 
        checkpoint if the validation loss improves, and plots the confusion matrix.

        Args:
            device (str): Device to run the validation on ('cuda' or 'cpu').
            fold (int): Current fold number in k-fold cross-validation.
            model (nn.Module): The neural network model.
            criterion (nn.Module): Loss function for evaluation.
            val_loader (DataLoader): DataLoader for validation dataset.
            val_losses (list): List to store validation loss per epoch.
            val_accuracies (list): List to store validation accuracy per epoch.
            epoch (int): Current epoch number.
            best_val_loss (float): Lowest recorded validation loss for the current fold.

        Returns:
            tuple: Updated validation loss and accuracy lists, average validation loss, 
            and validation accuracy for the epoch.
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
        
        return val_losses, val_accuracies, avg_val_loss, val_accuracy
    
    def train_plot(self, range, train_matric, val_matric, train_label, val_label, fold):
        """
        Plots training and validation metrics over epochs for a specific fold.

        Saves the plot for either loss or accuracy.

        Args:
            range (iterable): Epochs range for the plot x-axis.
            train_matric (list): Training metric values per epoch.
            val_matric (list): Validation metric values per epoch.
            train_label (str): Label for training metric (e.g., 'Train Loss').
            val_label (str): Label for validation metric (e.g., 'Validation Loss').
            fold (int): Current fold number in k-fold cross-validation.
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
        Main training function to train the model using k-fold cross-validation.

        For each fold, trains the model across epochs, tracks loss and accuracy, 
        validates and saves the best model checkpoint, and plots metrics.

        Returns:
            None
        """
        
        gc.collect()
        torch.cuda.empty_cache()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: ', device)
        
        dataset, skf = self.dataset_folds_preparation()
        
        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(dataset.data_frame, dataset.data_frame['label']))):
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
            model.to(device)
            
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
                
                val_losses, val_accuracies, avg_val_loss, val_accuracy = self.validation(
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
