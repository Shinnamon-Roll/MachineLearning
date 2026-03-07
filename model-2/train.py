import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import copy
import time
import ssl
import urllib.request
import json

# Bypass SSL verification for downloading pretrained weights
ssl._create_default_https_context = ssl._create_unverified_context

from model_mobilenet import CustomMobileNetV2
from data_loader import get_dataloaders

def train_binary_model(data_dir, batch_size=32, num_epochs=15, learning_rate=0.0001):
    # Check if GPU/MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load DataLoaders
    # Note: data_loader.py handles the actual image loading and augmentation
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

    # Initialize Model for 2 Classes (Salmon, Trout)
    model = CustomMobileNetV2(num_classes=2, pretrained=True)
    
    # Freeze 40% of layers using the class method
    model.freeze_percentage(freeze_percent=0.4)
    
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Store training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, paths in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Show current image being processed (optional, for progress)
                if phase == 'train':
                     print(f"Training on: {os.path.basename(paths[0])} ...    ", end='\r')

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # Use .float() instead of .double() for MPS compatibility
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save the model
    save_path = "mobilenet_v2_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save history to JSON for Dashboard
    # We save to a different file for Model 2
    history_path = "../dashboard/public/data/training_history_model2.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    # Path to Image folder (same as Model 1)
    # Adjust this path if necessary based on where you run the script
    DATA_DIR = "/Users/shinnamon/Documents/Project/MachineLearning/Image/"
    
    if os.path.exists(DATA_DIR):
        train_binary_model(DATA_DIR)
    else:
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please check the path or move your Image folder.")
