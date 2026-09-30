import hashlib
import json
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CLASSES, CONFIG, DATA_DIR, IMAGE_EXTENSIONS, REPO_DIR, SPLITS_PATH


class CLAHETransform:
    """CLAHE on the L channel (LAB) to enhance texture such as fat lines."""

    def __init__(self, clip_limit, tile_grid_size):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            return img
        l, a, b = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        merged = cv2.merge((clahe.apply(l), a, b))
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def _clahe():
    return CLAHETransform(CONFIG["clahe"]["clip_limit"], CONFIG["clahe"]["tile_grid_size"])


def _normalize():
    return transforms.Normalize(CONFIG["normalize_mean"], CONFIG["normalize_std"])


def train_transform():
    size = CONFIG["image_size"]
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        _clahe(),
        transforms.ToTensor(),
        _normalize(),
    ])


def eval_transform():
    """Used for val, test and inference, so preprocessing always matches."""
    size = CONFIG["image_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        _clahe(),
        transforms.ToTensor(),
        _normalize(),
    ])


def make_splits(data_dir=DATA_DIR, out_path=SPLITS_PATH):
    """Stratified split, written once and shared by every model and seed.

    Byte-identical images are grouped so a duplicate can never sit in two splits.
    """
    cfg = CONFIG["split"]
    rng = random.Random(cfg["seed"])
    splits = {"train": [], "val": [], "test": []}
    duplicates = 0
    unreadable = []

    for label, cls in enumerate(CLASSES):
        groups = {}
        for path in sorted((data_dir / cls).iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                unreadable.append(str(path.relative_to(REPO_DIR)))
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(path)
        duplicates += sum(len(g) - 1 for g in groups.values())

        group_list = list(groups.values())
        rng.shuffle(group_list)
        n = len(group_list)
        n_train = round(n * cfg["train"])
        n_val = round(n * cfg["val"])
        parts = {
            "train": group_list[:n_train],
            "val": group_list[n_train:n_train + n_val],
            "test": group_list[n_train + n_val:],
        }
        for split, split_groups in parts.items():
            for group in split_groups:
                for path in group:
                    splits[split].append([str(path.relative_to(REPO_DIR)), label])

    out_path.write_text(json.dumps({"config": cfg, "duplicates_grouped": duplicates,
                                    "unreadable_skipped": unreadable, **splits}, indent=2))
    return splits


def load_splits():
    if not SPLITS_PATH.exists():
        make_splits()
    return json.loads(SPLITS_PATH.read_text())


class SalmonTroutDataset(Dataset):
    def __init__(self, items, transform):
        self.paths = [REPO_DIR / p for p, _ in items]
        self.labels = [label for _, label in items]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(seed):
    splits = load_splits()
    generator = torch.Generator().manual_seed(seed)
    kwargs = {"batch_size": CONFIG["batch_size"], "num_workers": CONFIG["num_workers"],
              "worker_init_fn": _seed_worker}
    train_loader = DataLoader(SalmonTroutDataset(splits["train"], train_transform()),
                              shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(SalmonTroutDataset(splits["val"], eval_transform()), shuffle=False, **kwargs)
    test_loader = DataLoader(SalmonTroutDataset(splits["test"], eval_transform()), shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
