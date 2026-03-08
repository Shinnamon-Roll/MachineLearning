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
from focal_loss import FocalLoss

def train_binary_model(data_dir, batch_size=32, num_epochs=15, learning_rate=0.0001):
    # Check if GPU/MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load DataLoaders
    # Note: data_loader.py handles the actual image loading and augmentation
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

    # Calculate Class Weights to handle imbalance
    # Count samples in training set
    salmon_count = len([label for label in train_loader.dataset.labels if label == 0])
    trout_count = len([label for label in train_loader.dataset.labels if label == 1])
    total_count = salmon_count + trout_count
    
    print(f"Training Data Balance: Salmon={salmon_count}, Trout={trout_count}")
    
    # Calculate alpha for Focal Loss (alpha is weight for class 1 - Trout)
    # If balanced, alpha = 0.5. If Trout is rare, alpha > 0.5 to focus on it.
    if total_count > 0:
        alpha = salmon_count / total_count # Weight for Trout (Class 1)
        print(f"Using Focal Loss Alpha (for Trout): {alpha:.4f}")
    else:
        alpha = 0.5
        print("Warning: No samples found. Using default alpha 0.5")

    # Initialize Model for 2 Classes (Salmon, Trout)
    model = CustomMobileNetV2(num_classes=2, pretrained=True)
    
    # Freeze 40% of layers using the class method
    model.freeze_percentage(freeze_percent=0.4)
    
    model = model.to(device)

    # Loss and Optimizer
    # Use Focal Loss instead of CrossEntropyLoss
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Learning Rate Scheduler
    # Increased patience for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Store training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    start_time = time.time()

    # Phase 1: Train with frozen layers
    print("\n--- Phase 1: Training with frozen layers ---")
    
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
                # Step the scheduler based on training loss or validation loss
                # Here we step after validation phase actually, but keeping logic consistent
                pass
            
            # Step scheduler after validation phase
            if phase == 'val':
                scheduler.step(epoch_loss)

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
                # Save best model to disk immediately so we can use it even if training stops early
                torch.save(model.state_dict(), "mobilenet_v2_best.pth")
                print(f"Saved new best model with Acc: {best_acc:.4f}")

        print()
    
    # Phase 2: Fine-tuning (Unfreeze more layers)
    print("\n--- Phase 2: Fine-tuning (Unfreezing more layers) ---")
    
    # Unfreeze all layers (or just more layers)
    model.freeze_percentage(freeze_percent=0.0) # Unfreeze all
    
    # Re-initialize optimizer with lower learning rate for fine-tuning
    finetune_lr = learning_rate * 0.1
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
    
    # Fine-tune for additional epochs (e.g., 10 more epochs)
    finetune_epochs = 15 
    total_epochs = num_epochs + finetune_epochs
    
    for epoch in range(num_epochs, total_epochs):
        print(f'Epoch {epoch}/{total_epochs - 1}')
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

                if phase == 'train':
                     print(f"Fine-tuning on: {os.path.basename(paths[0])} ...    ", end='\r')

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

            epoch_loss = running_loss / dataset_sizes[phase]
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

    # --- Test Phase ---
    print("\nEvaluating on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    # Calculate Model Stats
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate inference time (approximate per image using last batch)
    start_time = time.time()
    with torch.no_grad():
        _ = model(inputs)
    end_time = time.time()
    inference_ms = (end_time - start_time) * 1000 / inputs.size(0)
    
    # Calculate size
    model_path = 'mobilenet_v2_best.pth'
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0

    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save metrics to JSON
    metrics = {
        "model_name": "MobileNetV2",
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1_score": test_f1,
        "confusion_matrix": conf_matrix,
        "classes": ["Salmon", "Trout"],
        "test_samples": len(all_labels),
        "params": f"{total_params/1000000:.1f}M",
        "inference": f"{inference_ms:.0f}ms",
        "size": f"{model_size_mb:.1f}MB",
        "split_counts": {
            "train": dataset_sizes['train'],
            "val": dataset_sizes['val'],
            "test": len(test_loader.dataset)
        }
    }
    
    metrics_path = "../dashboard/public/data/metrics_model2.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Path to Image folder (same as Model 1)
    # Adjust this path if necessary based on where you run the script
    DATA_DIR = "/Users/shinnamon/Documents/Project/MachineLearning/Image/"
    
    if os.path.exists(DATA_DIR):
        train_binary_model(DATA_DIR)
    else:
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please check the path or move your Image folder.")
