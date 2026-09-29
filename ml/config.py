"""Single source of truth for every experiment setting.

Both models are trained with exactly these values; only the backbone differs.
"""
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO_DIR = ML_DIR.parent
DATA_DIR = REPO_DIR / "Image"
SPLITS_PATH = ML_DIR / "splits.json"
WEIGHTS_DIR = ML_DIR / "weights"
DASHBOARD_DATA_DIR = REPO_DIR / "dashboard" / "public" / "data"

CLASSES = ["Salmon", "Trout"]  # index 0 = Salmon, 1 = Trout
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

CONFIG = {
    "seeds": [0, 1, 2],
    "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 2,
    "phase1_epochs": 15,        # backbone frozen, head only
    "phase2_epochs": 15,        # everything unfrozen
    "lr": 1e-4,
    "finetune_lr_factor": 0.1,  # phase 2 lr = lr * factor
    "scheduler": {"type": "ReduceLROnPlateau", "monitor": "val_loss", "factor": 0.1, "patience": 5},
    "loss": {"type": "FocalLoss", "gamma": 2.0, "alpha": "train class ratio"},
    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "latency": {"warmup": 10, "runs": 50, "batch_size": 1},
}


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
