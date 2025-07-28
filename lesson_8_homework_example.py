# -*- coding: utf-8 -*-
"""
PyTorch Dataset and DataLoader Example with Synthetic Data
==========================================================

This example demonstrates how to implement custom Dataset and DataLoader classes
using synthetic data. The synthetic data consists of noise with simple geometric
shapes (circles, squares, triangles) overlaid on top.

This serves as a reference implementation for the homework assignment.

DOCUMENTATION REFERENCES:
- Dataset Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Dataset API: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
- DataLoader API: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- Transforms: https://pytorch.org/vision/stable/transforms.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
from sklearn.metrics import classification_report
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SyntheticShapeDataset(Dataset):
    """
    Custom dataset that generates synthetic images with geometric shapes.
    
    Creates images with noise background and overlays one of three shapes:
    - Class 0: Circle
    - Class 1: Square  
    - Class 2: Triangle
    """
    
    def __init__(self, num_samples=1000, img_size=(64, 64), transform=None):
        """
        Initialize the synthetic dataset.
        
        Args:
            num_samples (int): Number of samples to generate
            img_size (tuple): Size of generated images (height, width)
            transform: Optional transform to be applied to images
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        
        # Define class names and mapping
        self.classes = ['circle', 'square', 'triangle']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Pre-generate labels for consistency
        self.labels = [random.randint(0, 2) for _ in range(self.num_samples)]
        
        print(f"Created synthetic dataset with {num_samples} samples")
        print(f"Classes: {self.classes}")
        print(f"Image size: {img_size}")
    
    def __len__(self):
        """Return the total number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate and return a sample at the given index.
        
        Args:
            idx (int): Index of the sample to generate
            
        Returns:
            tuple: (image, label) where image is a PIL Image and label is an integer
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        # Get the label for this index
        label = self.labels[idx]
        
        # Generate the synthetic image
        image = self._generate_image(label)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _generate_image(self, shape_class):
        """
        Generate a synthetic image with the specified shape.
        
        Args:
            shape_class (int): Class of shape to draw (0=circle, 1=square, 2=triangle)
            
        Returns:
            PIL.Image: Generated image
        """
        # Create base noise image
        width, height = self.img_size
        
        # Generate random noise background (RGB)
        noise_array = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(noise_array, 'RGB')
        
        # Create drawing context
        draw = ImageDraw.Draw(image)
        
        # Define shape parameters
        center_x, center_y = width // 2, height // 2
        shape_size = min(width, height) // 4
        
        # Choose color for the shape (brighter than noise)
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        color = random.choice(colors)
        
        # Add some randomness to position and size
        offset_x = random.randint(-shape_size//2, shape_size//2)
        offset_y = random.randint(-shape_size//2, shape_size//2)
        size_variation = random.randint(-shape_size//4, shape_size//4)
        
        actual_center_x = center_x + offset_x
        actual_center_y = center_y + offset_y
        actual_size = shape_size + size_variation
        
        # Draw the appropriate shape
        if shape_class == 0:  # Circle
            bbox = [
                actual_center_x - actual_size,
                actual_center_y - actual_size,
                actual_center_x + actual_size,
                actual_center_y + actual_size
            ]
            draw.ellipse(bbox, fill=color, outline='black', width=2)
            
        elif shape_class == 1:  # Square
            bbox = [
                actual_center_x - actual_size,
                actual_center_y - actual_size,
                actual_center_x + actual_size,
                actual_center_y + actual_size
            ]
            draw.rectangle(bbox, fill=color, outline='black', width=2)
            
        elif shape_class == 2:  # Triangle
            points = [
                (actual_center_x, actual_center_y - actual_size),  # Top
                (actual_center_x - actual_size, actual_center_y + actual_size),  # Bottom left
                (actual_center_x + actual_size, actual_center_y + actual_size)   # Bottom right
            ]
            draw.polygon(points, fill=color, outline='black')
        
        return image
    
    def visualize_samples(self, num_samples=9):
        """Visualize some samples from the dataset."""
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('Sample Synthetic Images', fontsize=16)
        
        indices = random.sample(range(len(self)), num_samples)
        
        for i, idx in enumerate(indices):
            row, col = i // 3, i % 3
            
            # Get sample without transforms for visualization
            temp_transform = self.transform
            self.transform = None
            image, label = self[idx]
            self.transform = temp_transform
            
            axes[row, col].imshow(image)
            axes[row, col].set_title(f'Class: {self.classes[label]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

def get_transforms():
    """
    Define data transforms for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform

def create_dataloaders(train_samples=800, val_samples=200, batch_size=32, num_workers=2):
    """
    Create training and validation dataloaders with synthetic data.
    
    Args:
        train_samples (int): Number of training samples
        val_samples (int): Number of validation samples
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, class_to_idx)
    """
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = SyntheticShapeDataset(
        num_samples=train_samples,
        img_size=(64, 64),
        transform=train_transform
    )
    
    val_dataset = SyntheticShapeDataset(
        num_samples=val_samples,
        img_size=(64, 64),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx

def demonstrate_dataloader():
    """Demonstrate the dataloader functionality."""
    print("Creating dataloaders...")
    train_loader, val_loader, class_to_idx = create_dataloaders(
        train_samples=100, val_samples=50, batch_size=8
    )
    
    print(f"Class to index mapping: {class_to_idx}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Get a batch and show info
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch shape: {batch_images.shape}")
    print(f"Label shape: {batch_labels.shape}")
    print(f"Image data range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
    print(f"Labels in batch: {batch_labels.tolist()}")
    
    # Visualize a batch
    visualize_batch(batch_images, batch_labels, class_to_idx)

def visualize_batch(images, labels, class_to_idx):
    """Visualize a batch of images."""
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Denormalize images for visualization
    images = images * 0.5 + 0.5  # Convert from [-1, 1] to [0, 1]
    images = torch.clamp(images, 0, 1)
    
    batch_size = images.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(batch_size):
        # Convert tensor to numpy and transpose for matplotlib
        img = images[i].permute(1, 2, 0).numpy()
        
        axes[i].imshow(img)
        axes[i].set_title(f'Class: {idx_to_class[labels[i].item()]}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Simple CNN for demonstration (same as homework)
class SimpleClassifier(nn.Module):
    """Simple CNN for shape classification"""
    
    def __init__(self, num_classes=3):
        super(SimpleClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def quick_training_demo():
    """Run a quick training demonstration."""
    print("\n" + "="*60)
    print("QUICK TRAINING DEMONSTRATION")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        train_samples=500, val_samples=100, batch_size=16
    )
    
    # Initialize model
    model = SimpleClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training (just 3 epochs)
    print("Training for 3 epochs...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1} - Accuracy: {epoch_acc:.2f}%')
    
    # Quick validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    final_acc = 100 * correct / total
    print(f'Final Validation Accuracy: {final_acc:.2f}%')
    print("Training demonstration completed!")

def main():
    """Main function demonstrating the synthetic dataset."""
    print("PyTorch Synthetic Dataset Example")
    print("="*50)
    
    # Create a sample dataset for visualization
    print("\n1. Creating sample dataset for visualization...")
    sample_dataset = SyntheticShapeDataset(num_samples=20, img_size=(64, 64))
    
    # Visualize some samples
    print("2. Visualizing sample images...")
    sample_dataset.visualize_samples()
    
    # Demonstrate dataloader functionality  
    print("\n3. Demonstrating DataLoader functionality...")
    demonstrate_dataloader()
    
    # Run quick training demo
    print("\n4. Running quick training demonstration...")
    quick_training_demo()
    
    print("\n" + "="*50)
    print("Example completed! This demonstrates:")
    print("- Custom Dataset implementation")
    print("- DataLoader creation and usage") 
    print("- Data transforms and augmentation")
    print("- Integration with PyTorch training loop")
    print("\nUse this as reference for your homework assignment!")

if __name__ == "__main__":
    main()

"""
KEY IMPLEMENTATION DETAILS:
==========================

1. SyntheticShapeDataset Class:
   - Inherits from torch.utils.data.Dataset
   - Implements __init__, __len__, and __getitem__ methods
   - Generates synthetic images with geometric shapes on noise background
   - Handles transforms properly

2. Data Generation:
   - Creates RGB noise as background
   - Overlays geometric shapes (circle, square, triangle) 
   - Adds randomness in position, size, and color
   - Uses PIL for image creation and drawing

3. Transforms:
   - Separate pipelines for training (with augmentation) and validation
   - Proper normalization and tensor conversion
   - Common augmentations: rotation, flip, color jitter

4. DataLoader:
   - Proper configuration with batch_size, shuffle, num_workers
   - Pin memory for GPU acceleration when available
   - Demonstrates batch processing

5. Integration:
   - Shows how custom dataset works with standard PyTorch training loop
   - Proper device handling and model integration
   - Visualization utilities for debugging

This example provides a complete working reference that students can
study before implementing their own dataset for the homework assignment.
"""