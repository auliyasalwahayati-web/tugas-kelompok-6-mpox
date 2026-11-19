import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetB0(nn.Module):

    def __init__(self, output=6, pretrained=True):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = efficientnet_b0(
            weights=weights
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            output
        )

    def forward(self, x):

        x = self.model(x)

        return torch.log_softmax(x, dim=1)