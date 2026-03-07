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

    print(f"Prediction: {class_name} ({confidence_score:.2f}%)")
    
    result = {
        "classification": class_name,
        "confidence": confidence_score
    }
    
    return result

if __name__ == "__main__":
    # Test
    MODEL_PATH = "salmon_trout_binary_model.pth"
    # Replace with an actual image path for testing
    TEST_IMAGE = "../Image/Salmon!/Salmon Test/some_image.jpg" 
    
    # if os.path.exists(TEST_IMAGE):
    #     predict_pipeline(TEST_IMAGE, MODEL_PATH)
