"""Train both models with the identical recipe from config.CONFIG.

    python ml/train.py --model all          # full run: 3 seeds per model
    python ml/train.py --model all --quick  # smoke test: 1+1 epochs, one seed
"""
import argparse
import copy
import json
import random
import shutil
import ssl
import statistics
import time

import numpy as np
import torch
import torch.optim as optim

from config import CONFIG, DASHBOARD_DATA_DIR, WEIGHTS_DIR, get_device
from data import get_dataloaders, load_splits
from evaluate import evaluate, measure_latency
from focal_loss import FocalLoss
from models import MODELS, build_model, set_backbone_trainable

# Bypass SSL verification for downloading pretrained weights
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += inputs.size(0)
    return total_loss / count, correct / count


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    sched = CONFIG["scheduler"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched["factor"], patience=sched["patience"])
    return optimizer, scheduler


def train_one(name, seed, cfg, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(seed)

    labels = train_loader.dataset.labels
    alpha = labels.count(0) / len(labels)  # weight for class 1 (Trout)
    criterion = FocalLoss(alpha=alpha, gamma=CONFIG["loss"]["gamma"])

    model = build_model(name, pretrained=True).to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = {"val_acc": -1.0, "state": None, "epoch": -1}

    phases = [
        ("phase1 (backbone frozen)", False, cfg["phase1_epochs"], CONFIG["lr"]),
        ("phase2 (fine-tune all)", True, cfg["phase2_epochs"], CONFIG["lr"] * CONFIG["finetune_lr_factor"]),
    ]
    epoch = 0
    for phase_name, backbone_trainable, epochs, lr in phases:
        set_backbone_trainable(model, backbone_trainable)
        optimizer, scheduler = make_optimizer(model, lr)
        print(f"[{name} seed={seed}] {phase_name}: {epochs} epochs, lr={lr:g}")
        for _ in range(epochs):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
                history[key].append(value)
            print(f"  epoch {epoch:2d}  train {train_loss:.4f}/{train_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}")
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            epoch += 1

    model.load_state_dict(best["state"])
    WEIGHTS_DIR.mkdir(exist_ok=True)
    weights_path = WEIGHTS_DIR / f"{name}_seed{seed}.pth"
    torch.save(model.state_dict(), weights_path)

    result = evaluate(model, test_loader, device)
    result.update({
        "seed": seed,
        "val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "latency_ms": measure_latency(model, device),
        "size_mb": weights_path.stat().st_size / (1024 * 1024),
        "params": sum(p.numel() for p in model.parameters()),
    })
    print(f"[{name} seed={seed}] best epoch {best['epoch']}  val_acc {best['val_acc']:.4f}  "
          f"test_acc {result['accuracy']:.4f}  f1 {result['f1_score']:.4f}")
    return result, history


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def train_model(name, cfg, device):
    runs = [train_one(name, seed, cfg, device) for seed in cfg["seeds"]]
    results = [r for r, _ in runs]
    selected = max(range(len(runs)), key=lambda i: results[i]["val_acc"])
    chosen, history = runs[selected]
    shutil.copy(WEIGHTS_DIR / f"{name}_seed{chosen['seed']}.pth", WEIGHTS_DIR / f"{name}.pth")

    splits = load_splits()
    metrics = {"model_name": MODELS[name]["display_name"]}
    for key in ("accuracy", "precision", "recall", "f1_score"):
        metrics[key], metrics[f"{key}_std"] = mean_std([r[key] for r in results])
    latency, _ = mean_std([r["latency_ms"] for r in results])
    metrics.update({
        "confusion_matrix": chosen["confusion_matrix"],
        "recall_per_class": chosen["recall_per_class"],
        "classes": list(chosen["recall_per_class"]),
        "test_samples": chosen["test_samples"],
        "params": f"{chosen['params'] / 1e6:.1f}M",
        "inference": f"{latency:.1f}ms",
        "size": f"{chosen['size_mb']:.1f}MB",
        "split_counts": {k: len(splits[k]) for k in ("train", "val", "test")},
        "selected_seed": chosen["seed"],
        "seeds": [{k: r[k] for k in ("seed", "val_acc", "best_epoch", "accuracy", "precision",
                                      "recall", "f1_score", "latency_ms")} for r in results],
        "device": str(device),
        "config": cfg,
    })

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DASHBOARD_DATA_DIR / MODELS[name]["metrics_file"]).write_text(json.dumps(metrics, indent=2))
    (DASHBOARD_DATA_DIR / MODELS[name]["history_file"]).write_text(json.dumps(history, indent=2))
    print(f"[{name}] test acc {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}  "
          f"f1 {metrics['f1_score']:.4f} ± {metrics['f1_score_std']:.4f}  (n={len(results)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="1+1 epochs, first seed only (smoke test)")
    args = parser.parse_args()

    cfg = copy.deepcopy(CONFIG)
    if args.quick:
        cfg.update(phase1_epochs=1, phase2_epochs=1, seeds=cfg["seeds"][:1])

    device = get_device()
    print(f"device: {device}")
    start = time.time()
    for name in (list(MODELS) if args.model == "all" else [args.model]):
        train_model(name, cfg, device)
    print(f"done in {(time.time() - start) / 60:.1f} min")
