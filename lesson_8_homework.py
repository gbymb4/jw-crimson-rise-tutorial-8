# -*- coding: utf-8 -*-
"""
PyTorch Homework Assignment: Datasets and DataLoaders
====================================================

OBJECTIVE: Practice implementing custom datasets and dataloaders in PyTorch

INSTRUCTIONS:
1. Choose an image classification dataset from Kaggle (e.g., Cats vs Dogs, CIFAR-10, Flowers, etc.)
2. Download and organize the dataset
3. Complete the TODOs below to implement:
   - Custom Dataset class
   - Data transforms
   - DataLoader setup
4. Run the training and evaluate results

The neural network, training loop, and evaluation code are already implemented.
Your job is to focus on the data loading pipeline!

USEFUL DOCUMENTATION:
- PyTorch Dataset Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Dataset API Reference: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
- DataLoader API Reference: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- Transforms Documentation: https://pytorch.org/vision/stable/transforms.html
- PIL Image Documentation: https://pillow.readthedocs.io/en/stable/reference/Image.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CustomImageDataset(Dataset):
    """
    TODO: Implement a custom dataset class for loading images and labels
    
    This class should:
    1. Initialize with data directory path, transform, and optional label mapping
    2. Load image paths and corresponding labels
    3. Implement __len__ to return dataset size
    4. Implement __getitem__ to return (image, label) pairs
    
    DOCUMENTATION REFERENCES:
    - Dataset base class: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    - PIL Image operations: https://pillow.readthedocs.io/en/stable/reference/Image.html
    - OS path operations: https://docs.python.org/3/library/os.html#os.walk
    
    HINTS:
    - Use os.walk() or os.listdir() to find image files
    - Common image extensions: .jpg, .jpeg, .png, .bmp
    - Use PIL.Image.open() to load images
    - Convert labels to integers (0, 1, 2, ...)
    - Apply transforms if provided
    
    PSEUDOCODE:
    def __init__(self, data_dir, transform=None):
        # Store parameters
        # Scan directory for images and labels
        # Create lists: self.image_paths, self.labels
        # Create label mapping: self.class_to_idx
    
    def __len__(self):
        # Return number of samples
    
    def __getitem__(self, idx):
        # Load image at index idx
        # Get corresponding label
        # Apply transforms if any
        # Return (image_tensor, label)
    """
    
    def __init__(self, data_dir, transform=None):
        # TODO: Initialize the dataset
        # Your code here...
        pass
    
    def __len__(self):
        # TODO: Return the size of the dataset
        # Your code here...
        pass
    
    def __getitem__(self, idx):
        # TODO: Load and return image and label at given index
        # Your code here...
        pass

def get_transforms():
    """
    TODO: Define data transforms for training and validation
    
    Create two transform pipelines:
    1. train_transform: Should include data augmentation
    2. val_transform: Should only include normalization and resizing
    
    DOCUMENTATION REFERENCES:
    - Transforms Guide: https://pytorch.org/vision/stable/transforms.html
    - Transform Examples: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
    
    HINTS:
    - Common transforms: Resize, CenterCrop, RandomHorizontalFlip, RandomRotation
    - Always end with ToTensor() and Normalize()
    - ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Or calculate your own dataset statistics
    
    PSEUDOCODE:
    train_transform = transforms.Compose([
        # Add data augmentation transforms here
        transforms.ToTensor(),
        transforms.Normalize(mean=[...], std=[...])
    ])
    
    val_transform = transforms.Compose([
        # Add basic transforms here (resize, crop)
        transforms.ToTensor(),
        transforms.Normalize(mean=[...], std=[...])
    ])
    """
    
    # TODO: Implement training transforms with augmentation
    train_transform = None  # Your code here...
    
    # TODO: Implement validation transforms (no augmentation)
    val_transform = None   # Your code here...
    
    return train_transform, val_transform

def create_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    TODO: Create training and validation dataloaders
    
    Steps:
    1. Get transforms using get_transforms()
    2. Create dataset instances for train and validation
    3. Create DataLoader instances with appropriate parameters
    
    DOCUMENTATION REFERENCES:
    - DataLoader Guide: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    - DataLoader Parameters: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
    
    HINTS:
    - Use shuffle=True for training, shuffle=False for validation
    - num_workers > 0 can speed up data loading
    - Consider pin_memory=True if using GPU
    
    PSEUDOCODE:
    train_transform, val_transform = get_transforms()
    
    train_dataset = CustomImageDataset(train_dir, train_transform)
    val_dataset = CustomImageDataset(val_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, ...)
    val_loader = DataLoader(val_dataset, ...)
    """
    
    # TODO: Get transforms
    train_transform, val_transform = get_transforms()
    
    # TODO: Create dataset instances
    train_dataset = None  # Your code here...
    val_dataset = None    # Your code here...
    
    # TODO: Create dataloaders
    train_loader = None   # Your code here...
    val_loader = None     # Your code here...
    
    return train_loader, val_loader, train_dataset.class_to_idx

# ============================================================================
# PROVIDED CODE - Neural Network, Training, and Evaluation
# ============================================================================

class SimpleClassifier(nn.Module):
    """Simple CNN for image classification"""
    
    def __init__(self, num_classes):
        super(SimpleClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training function"""
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4
    
    # TODO: Update these paths to your dataset
    TRAIN_DIR = "path/to/your/train/directory"  # Update this!
    VAL_DIR = "path/to/your/validation/directory"  # Update this!
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    print("Creating dataloaders...")
    try:
        train_loader, val_loader, class_to_idx = create_dataloaders(
            TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS
        )
        print(f"Classes found: {class_to_idx}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Make sure you've implemented the TODO sections and updated the data paths!")
        return
    
    # Initialize model
    num_classes = len(class_to_idx)
    model = SimpleClassifier(num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    print("\nTraining completed!")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    _, final_acc, all_preds, all_targets = validate(model, val_loader, criterion, device)
    print(f'Final Validation Accuracy: {final_acc:.2f}%')
    
    # Classification report
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved as 'trained_model.pth'")

if __name__ == "__main__":
    main()

"""
SUBMISSION CHECKLIST:
=====================
□ Downloaded dataset from Kaggle
□ Organized data into train/validation folders
□ Implemented CustomImageDataset class
□ Implemented get_transforms() function  
□ Implemented create_dataloaders() function
□ Updated TRAIN_DIR and VAL_DIR paths
□ Successfully ran training without errors
□ Achieved reasonable validation accuracy (>60%)
□ Included training plots in submission
□ Answered reflection questions (see below)

REFLECTION QUESTIONS:
====================
1. What dataset did you choose and why?
2. What data augmentation techniques did you use and why?
3. How did your model perform? What was the final validation accuracy?
4. What challenges did you face with the dataset/dataloader implementation?
5. How might you improve the model performance further?

BONUS CHALLENGES:
================
□ Implement data augmentation that's specific to your dataset
□ Add class balancing if your dataset is imbalanced  
□ Implement custom collate_fn for the DataLoader
□ Add data visualization functions to explore your dataset
□ Implement cross-validation instead of a single train/val split
"""