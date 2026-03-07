import torch
from model import ImprovedDenseNet121
from train import freeze_layers

def test_setup():
    print("Testing model instantiation...")
    # Use pretrained=False to avoid SSL errors during test and speed up
    model = ImprovedDenseNet121(num_classes=6, pretrained=False)
    print("Model instantiated.")
    
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 6)
    
    print("Testing layer freezing...")
    freeze_layers(model, freeze_percent=0.4)
    
    # Check if some parameters are frozen
    params = list(model.parameters())
    
    # DenseNet has many parameters. 
    # Check first few are frozen and last few are not.
    print(f"First param requires_grad: {params[0].requires_grad}")
    print(f"Last param requires_grad: {params[-1].requires_grad}")
    
    assert params[0].requires_grad == False
    assert params[-1].requires_grad == True
    print("Layer freezing test passed.")

if __name__ == "__main__":
    test_setup()
