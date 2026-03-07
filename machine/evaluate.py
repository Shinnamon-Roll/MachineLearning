import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from model import ImprovedDenseNet121
from data_loader import get_dataloaders

def evaluate_model(model_path, data_dir, batch_size=32):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DataLoaders (We only need test/val loader here)
    _, test_loader = get_dataloaders(data_dir, batch_size)
    
    # Initialize Model
    model = ImprovedDenseNet121(num_classes=2, pretrained=False)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Print progress
            print(f"Processed batch {i+1}/{len(test_loader)}", end='\r')

    print("\nEvaluation complete.")

    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    classes = test_loader.dataset.classes
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    DATA_DIR = "/Users/shinnamon/Documents/Project/MachineLearning/Image/"
    MODEL_PATH = "salmon_trout_binary_model.pth"
    
    evaluate_model(MODEL_PATH, DATA_DIR)
