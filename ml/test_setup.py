"""Smoke checks: python ml/test_setup.py"""
import torch

from config import CLASSES
from data import load_splits
from models import MODELS, build_model, set_backbone_trainable


def test_models():
    for name in MODELS:
        model = build_model(name, pretrained=False).eval()
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 2), name

        set_backbone_trainable(model, False)
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable and all(id(p) not in backbone_ids for p in trainable), name

        set_backbone_trainable(model, True)
        assert all(p.requires_grad for p in model.parameters()), name


def test_splits():
    splits = load_splits()
    sets = {k: {p for p, _ in splits[k]} for k in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    for k in sets:
        labels = [label for _, label in splits[k]]
        share = labels.count(0) / len(labels)
        assert 0.4 < share < 0.6, (k, share)
        print(f"{k}: {len(labels)} images, {CLASSES[0]} share {share:.2f}")
    print(f"duplicates grouped: {splits['duplicates_grouped']}, unreadable skipped: {len(splits['unreadable_skipped'])}")


if __name__ == "__main__":
    test_models()
    test_splits()
    print("all checks passed")
