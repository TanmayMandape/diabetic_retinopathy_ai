import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model import get_gradcam_target_layer
from src.transforms import val_transform


def generate_gradcam(model, image, target_layer=None):
    """
    Generate Grad-CAM visualization for a retinal image.

    image:
        Preprocessed RGB numpy array in [0, 255] uint8,
        e.g. output of src.preprocessing.load_image()
    """

    if target_layer is None:
        target_layer = get_gradcam_target_layer(model)

    # Prepare input tensor
    input_tensor = val_transform(image).unsqueeze(0)

    # Move input to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get the model's predicted class
    with __import__("torch").no_grad():
        outputs = model(input_tensor)
        predicted_class = outputs.argmax(dim=1).item()

    # Tell Grad-CAM which class we want to explain
    targets = [ClassifierOutputTarget(predicted_class)]

    # Create Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    # Generate CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Convert original image to float [0, 1]
    rgb_image = image.astype("float32") / 255.0

    # Overlay heatmap on original image
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization