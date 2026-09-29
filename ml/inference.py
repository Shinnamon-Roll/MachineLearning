"""Called by the dashboard API. Must print exactly one JSON line to stdout."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image

from config import CLASSES, WEIGHTS_DIR, get_device
from data import eval_transform
from models import MODELS, build_model


def predict(image_path, model_name, weights_path):
    device = get_device()
    if not Path(weights_path).exists():
        return {"error": f"Model file not found at {weights_path}"}
    try:
        model = build_model(model_name, pretrained=False)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        return {"error": f"Failed to load model weights: {e}"}
    model.to(device).eval()

    try:
        image = Image.open(image_path).convert("RGB")
        tensor = eval_transform()(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    idx = int(probs.argmax())
    return {
        "class": CLASSES[idx],
        "confidence": probs[idx].item() * 100,
        "probabilities": {c: probs[i].item() * 100 for i, c in enumerate(CLASSES)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salmon/Trout inference")
    parser.add_argument("image_path")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--model_path", help="default: ml/weights/<model>.pth")
    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)
    weights = args.model_path or WEIGHTS_DIR / f"{args.model}.pth"
    print(json.dumps(predict(args.image_path, args.model, weights)))
