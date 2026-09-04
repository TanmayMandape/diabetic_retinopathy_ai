import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_model(num_classes=5):

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    num_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        num_features,
        num_classes
    )

    return model
