import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ImprovedDenseNet121

# Define classes for Binary Classification
CLASSES = [
    "Salmon",
    "Trout"
]

def load_model(model_path, num_classes):
    # Load model structure
    model = ImprovedDenseNet121(num_classes=num_classes, pretrained=False)
    
    # Load weights
    try:
        # Support for MPS/CUDA/CPU
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
        
    model.eval()
    return model

def predict_pipeline(image_path, model_path):
    """
    Inference Pipeline:
    Binary Classification: Salmon vs Trout
    """
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Binary Classification ---
    model = load_model(model_path, num_classes=len(CLASSES))
    if model is None:
        return {"error": "Failed to load model"}
        
    model.to(device)

    # Transform Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        class_idx = predicted.item()
        class_name = CLASSES[class_idx]
        confidence_score = confidence.item() * 100

    # print(f"Prediction: {class_name} ({confidence_score:.2f}%)")
    
    result = {
        "class": class_name,
        "confidence": confidence_score,
        "probabilities": {
            "Salmon": probs[0][0].item() * 100,
            "Trout": probs[0][1].item() * 100
        }
    }
    
    return result

if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description='Inference for Salmon/Trout Model 1')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    # Default model path assumes running from project root or model-1 dir, adjust as needed
    default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salmon_trout_binary_model.pth")
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to model weights')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        exit(1)

    # Suppress print statements from predict_pipeline by redirecting stdout temporarily or modifying function
    # But since predict_pipeline has print(), we should modify it to NOT print if we want clean JSON output.
    # For now, let's modify predict_pipeline to remove the print statement.
    
    result = predict_pipeline(args.image_path, args.model_path)
    print(json.dumps(result))
