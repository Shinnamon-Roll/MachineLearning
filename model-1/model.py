import torch
import torch.nn as nn
import torchvision.models as models

class ImprovedDenseNet121(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedDenseNet121, self).__init__()
        
        # Base model: Use DenseNet121
        # Implement Transfer Learning starting from ImageNet weights
        original_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Feature extractor (features layer)
        self.features = original_model.features
        
        # Modifications (Improved DenseNet121): 
        # Add a Global Average Pooling (GAP) layer to reduce dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regularization: Add a Dropout layer with a rate of 0.2 to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        
        # Classifier
        # DenseNet121 features output channels is 1024
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.features(x)
        
        # Apply GAP
        out = self.global_avg_pool(features)
        
        # Flatten
        out = torch.flatten(out, 1)
        
        # Apply Dropout
        out = self.dropout(out)
        
        # Classification
        out = self.classifier(out)
        
        return out
