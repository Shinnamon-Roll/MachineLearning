import argparse
import json
import time

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import CLASSES, CONFIG, WEIGHTS_DIR, get_device
from data import get_dataloaders
from models import MODELS, build_model


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        preds.extend(outputs.argmax(1).cpu().tolist())
        labels.extend(targets.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    _, recall_per_class, _, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "recall_per_class": dict(zip(CLASSES, recall_per_class.tolist())),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "test_samples": len(labels),
    }


@torch.no_grad()
def measure_latency(model, device):
    """Same protocol for every model: batch 1, warm-up, then mean over N synced runs."""
    cfg = CONFIG["latency"]
    model.eval()
    x = torch.randn(cfg["batch_size"], 3, CONFIG["image_size"], CONFIG["image_size"], device=device)
    for _ in range(cfg["warmup"]):
        model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(cfg["runs"]):
        model(x)
    _sync(device)
    return (time.perf_counter() - start) * 1000 / cfg["runs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate saved weights on the test split")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--weights", help="default: ml/weights/<model>.pth")
    args = parser.parse_args()

    device = get_device()
    model = build_model(args.model, pretrained=False)
    weights = args.weights or WEIGHTS_DIR / f"{args.model}.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)

    _, _, test_loader = get_dataloaders(seed=0)
    result = evaluate(model, test_loader, device)
    result["latency_ms"] = measure_latency(model, device)
    print(json.dumps(result, indent=2))
