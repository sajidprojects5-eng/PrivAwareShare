
import torch
import torch.nn as nn
import torchvision.models as models

class PrivNetCNN(nn.Module):
    """ResNet18-based embedding extractor for faces."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        modules = list(backbone.children())[:-1]  # remove final FC
        self.feature_extractor = nn.Sequential(*modules)
        self.fc = nn.Linear(backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        # x: (B,3,224,224)
        feats = self.feature_extractor(x)  # (B,C,1,1)
        feats = feats.view(feats.size(0), -1)
        emb = self.fc(feats)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb
