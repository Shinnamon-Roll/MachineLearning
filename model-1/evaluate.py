import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from model import ImprovedDenseNet121
from data_loader import get_dataloaders

def evaluate_model(model_path, data_dir, batch_size=32):
    # Check if GPU is available
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load DataLoaders (We need all loaders to count splits)
    # Note: get_dataloaders returns (train_loader, val_loader, test_loader)
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)

    # Calculate split counts
    split_counts = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
        "test": len(test_loader.dataset)
    }
    
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    classes = test_loader.dataset.classes
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Calculate Model Stats
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate inference time
    import time
    start_time = time.time()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    end_time = time.time()
    avg_inference_ms = ((end_time - start_time) / 10) * 1000

    # Calculate model size
    model_size_mb = 0
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Save metrics to JSON for dashboard
    metrics_data = {
        "model_name": "ImprovedDenseNet121",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "test_samples": len(all_labels),
        "params": f"{total_params/1000000:.1f}M",
        "inference": f"{avg_inference_ms:.0f}ms",
        "size": f"{model_size_mb:.1f}MB",
        "split_counts": split_counts
    }
    
    json_path = "/Users/shinnamon/Documents/Project/MachineLearning/dashboard/public/data/metrics.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to {json_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('/Users/shinnamon/Documents/Project/MachineLearning/dashboard/public/data/confusion_matrix.png') # Save to public/data too
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png' and dashboard folder")

if __name__ == "__main__":
    DATA_DIR = "/Users/shinnamon/Documents/Project/MachineLearning/Image/"
    MODEL_PATH = "model-1/salmon_trout_binary_model.pth"
    
    evaluate_model(MODEL_PATH, DATA_DIR)
