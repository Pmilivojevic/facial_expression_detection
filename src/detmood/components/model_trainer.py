from src.detmood.constant.dataset_preparation import CustomImageDataset
from src.detmood.entity.config_entity import ModelTrainerConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: ', device)
        
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
                    
                avg_train_loss = running_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                train_accuracy = 100 * correct_train / total_train
                train_accuracies.append(train_accuracy)
                
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
                
                print(f'Epoch [{epoch+1}/{self.config.model_params.num_epochs}], '
                        f'Loss: {avg_train_loss:.4f}, '
                        f'Validation Loss: {avg_val_loss:.4f}, '
                        f'Train Accuracy: {train_accuracy:.2f}%, '
                        f'Validation Accuracy: {val_accuracy:.2f}%')
                
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
            
            epochs_range = range(1, self.config.model_params.num_epochs + 1)
            
            plt.figure(figsize=(12, 6))
            plt.plot(epochs_range, train_losses, label='Train Loss')
            plt.plot(epochs_range, val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Train/validation Loss for Fold {fold + 1}')
            plt.legend()
            plt.savefig(os.path.join(self.config.figures, f'train_val_lossfold_{fold + 1}.png'))
            
            plt.figure(figsize=(12, 6))
            plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
            plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Train/Validation Accuracy for Fold {fold + 1}')
            plt.legend()
            plt.savefig(os.path.join(self.config.figures, f'train_val_accuracy_fold_{fold + 1}.png'))
            
            print(f'Finished fold {fold + 1}/{self.config.model_params.num_folds}\n')
        
        print('Training completed.')
