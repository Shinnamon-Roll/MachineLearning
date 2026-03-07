import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import argparse
import numpy as np
import cv2

# Ensure we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_mobilenet import CustomMobileNetV2

# Define classes
CLASSES = ['Salmon', 'Trout']

class CLAHETransform:
    """
    Custom transform to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Must match data_loader.py implementation exactly.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return Image.fromarray(final_img)
        return img

def predict(image_path, model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Initialize Model
    model = CustomMobileNetV2(num_classes=2, pretrained=False)
    
    # Load Weights
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}
        
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        return {"error": f"Failed to load model weights: {str(e)}"}
        
    model.to(device)
    model.eval()
    
    # Preprocessing (Must match validation/test transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
        
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        class_idx = predicted.item()
        class_name = CLASSES[class_idx]
        confidence_score = confidence.item() * 100
        
    return {
        "class": class_name,
        "confidence": confidence_score,
        "probabilities": {
            "Salmon": probs[0][0].item() * 100,
            "Trout": probs[0][1].item() * 100
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Salmon/Trout Model 2')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default=os.path.join(current_dir, 'mobilenet_v2_best.pth'), help='Path to model weights')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        exit(1)

    result = predict(args.image_path, args.model_path)
    print(json.dumps(result))
