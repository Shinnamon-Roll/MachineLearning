import torch
import torch.nn as nn
import torchvision.models as models


class ImprovedDenseNet121(nn.Module):
    """DenseNet121 backbone + GAP + Dropout(0.2) + Linear head."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.densenet121(weights=weights).features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, num_classes)

    @property
    def backbone(self):
        return self.features

    def forward(self, x):
        # torchvision's DenseNet applies ReLU after the last norm layer before pooling
        out = torch.relu(self.features(x))
        out = torch.flatten(self.global_avg_pool(out), 1)
        return self.classifier(self.dropout(out))


class CustomMobileNetV2(nn.Module):
    """MobileNetV2 backbone + its default Dropout(0.2) + Linear head."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.base_model = models.mobilenet_v2(weights=weights)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    @property
    def backbone(self):
        return self.base_model.features

    def forward(self, x):
        return self.base_model(x)


# Everything the rest of the code needs to know about each model.
MODELS = {
    "densenet": {
        "cls": ImprovedDenseNet121,
        "display_name": "DenseNet121",
        "metrics_file": "metrics.json",
        "history_file": "training_history.json",
    },
    "mobilenet": {
        "cls": CustomMobileNetV2,
        "display_name": "MobileNetV2",
        "metrics_file": "metrics_model2.json",
        "history_file": "training_history_model2.json",
    },
}


def build_model(name, pretrained=True, num_classes=2):
    return MODELS[name]["cls"](num_classes=num_classes, pretrained=pretrained)


def set_backbone_trainable(model, trainable):
    """Phase 1 freezes the whole backbone (head only); phase 2 unfreezes everything."""
    for p in model.parameters():
        p.requires_grad = True
    if not trainable:
        for p in model.backbone.parameters():
            p.requires_grad = False
