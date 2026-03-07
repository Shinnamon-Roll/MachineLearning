import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

class CustomMobileNetV2(nn.Module):
    """
    Custom MobileNetV2 architecture for binary classification (Salmon vs. Trout).
    Adapted for transfer learning with partial layer freezing.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CustomMobileNetV2, self).__init__()
        
        # 1. Base Architecture: Load MobileNetV2 with default ImageNet weights
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.base_model = models.mobilenet_v2(weights=weights)
        
        # 2. Custom Classifier Head
        # The default classifier is a Sequential block:
        # (0): Dropout(p=0.2, inplace=False)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)
        
        # We keep the dropout layer but replace the final linear layer
        in_features = self.base_model.classifier[1].in_features
        
        # Replace the final classification layer with a new one for our specific num_classes
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        """
        return self.base_model(x)
    
    def freeze_percentage(self, freeze_percent=0.4):
        """
        Freezes the first `freeze_percent` of the model's parameters.
        This matches the behavior of the DenseNet pipeline for fair comparison.
        
        Args:
            freeze_percent (float): The percentage of parameters to freeze (0.0 to 1.0).
                                    Default is 0.4 (40%).
        """
        # Get all parameters as a list
        parameters = list(self.parameters())
        num_params = len(parameters)
        
        # Calculate how many parameters to freeze
        num_frozen = int(freeze_percent * num_params)
        
        print(f"Total parameters layers: {num_params}")
        print(f"Freezing first {num_frozen} layers ({freeze_percent*100:.1f}%)")
        
        # Freeze the first num_frozen parameters
        for i, param in enumerate(parameters):
            if i < num_frozen:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # Verify which parts are frozen/trainable
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

if __name__ == "__main__":
    # Simple test to verify the model structure
    model = CustomMobileNetV2(num_classes=2)
    print(model)
    model.freeze_percentage(0.4)
