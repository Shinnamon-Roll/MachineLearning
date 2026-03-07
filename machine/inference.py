import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ImprovedDenseNet121

# Define classes for Stage 1 (Binary)
STAGE1_CLASSES = [
    "Salmon",
    "Trout"
]

# Define classes for Stage 2 (Freshness) - Assuming these are the target classes
STAGE2_CLASSES = [
    "Pink Orange",
    "Bright Orange",
    "Red Orange"
]

def load_model(model_path, num_classes):
    # Load model structure
    model = ImprovedDenseNet121(num_classes=num_classes, pretrained=False)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
        
    model.eval()
    return model

def predict_pipeline(image_path, stage1_model_path, stage2_model_path=None):
    """
    Inference Pipeline:
    1. Binary Classification: Salmon vs Trout
    2. (Optional) Freshness Grading: If Salmon, grade freshness
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stage 1: Binary Classification ---
    stage1_model = load_model(stage1_model_path, num_classes=len(STAGE1_CLASSES))
    if stage1_model is None:
        return {"error": "Failed to load Stage 1 model"}
        
    stage1_model.to(device)

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

    # Predict Stage 1
    with torch.no_grad():
        outputs1 = stage1_model(image_tensor)
        # Apply softmax to get probabilities
        probs1 = torch.nn.functional.softmax(outputs1, dim=1)
        confidence1, predicted1 = torch.max(probs1, 1)
        
        class1_idx = predicted1.item()
        class1_name = STAGE1_CLASSES[class1_idx]
        confidence_score = confidence1.item() * 100

    print(f"Stage 1 Prediction: {class1_name} ({confidence_score:.2f}%)")
    
    result = {
        "classification": class1_name,
        "confidence": confidence_score,
        "freshness": None
    }

    # --- Stage 2: Freshness Grading (Only if Salmon and model provided) ---
    if class1_name == "Salmon" and stage2_model_path:
        print("Detected Salmon. Proceeding to Freshness Grading...")
        stage2_model = load_model(stage2_model_path, num_classes=len(STAGE2_CLASSES))
        
        if stage2_model:
            stage2_model.to(device)
            with torch.no_grad():
                outputs2 = stage2_model(image_tensor)
                probs2 = torch.nn.functional.softmax(outputs2, dim=1)
                _, predicted2 = torch.max(probs2, 1)
                class2_idx = predicted2.item()
                freshness_grade = STAGE2_CLASSES[class2_idx]
            
            print(f"Stage 2 Prediction (Freshness): {freshness_grade}")
            result["freshness"] = freshness_grade
        else:
            print("Stage 2 model not found or failed to load.")

    return result

if __name__ == "__main__":
    # Example usage
    # result = predict_pipeline(
    #     image_path="path/to/test_image.jpg",
    #     stage1_model_path="salmon_trout_binary_model.pth"
    # )
    # print(result)
    pass
