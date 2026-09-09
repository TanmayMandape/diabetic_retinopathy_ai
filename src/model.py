import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes=5, pretrained=True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


def get_gradcam_target_layer(model):
    """
    Last convolutional feature block of EfficientNet-B0 -- the
    standard Grad-CAM target for this architecture in torchvision.
    """
    return model.features[-1]
