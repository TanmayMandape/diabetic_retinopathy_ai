import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_binary_model(pretrained=True):
    """
    Stage 1:
    Classifies fundus images as:
    0 = Non-referable
    1 = Referable
    """

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        2
    )

    return model


def create_severity_model(pretrained=True):
    """
    Stage 2:
    Classifies referable DR into:
    0 = Grade 2
    1 = Grade 3
    2 = Grade 4
    """

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        3
    )

    return model