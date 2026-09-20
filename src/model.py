import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes=5, pretrained=True):

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    # Get the number of input features
    in_features = model.classifier[1].in_features

    # Replace ImageNet classifier with our 5-class DR classifier
    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model