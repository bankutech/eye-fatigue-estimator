import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class FatigueModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FatigueModel, self).__init__()
        
        # Load pre-trained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)
            
        # The original mobilenet classifier is a Sequential block with Dropout and Linear layer.
        # We replace the classifier with a custom regression head.
        in_features = self.backbone.last_channel
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1) # Output a single value (fatigue score 0-100)
        )
        
    def forward(self, x):
        # We use a scaled sigmoid activation to bound the score to [0, 100]
        # and prevent gradient death from clamp
        x = self.backbone(x)
        return torch.sigmoid(x) * 100.0

if __name__ == "__main__":
    # Test the model with dummy data
    model = FatigueModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output value:", output.item())
