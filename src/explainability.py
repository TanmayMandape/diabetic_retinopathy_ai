"""
Explainability Module: Grad-CAM and Retinal Vessel Enhancement
SIH26038 — Explainable AI-Based Diabetic Retinopathy Screening
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import frangi


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN models.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)

        output = self.model(image_tensor)
        probabilities = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]
        score.backward()

        # Global-average-pool the gradients
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input image resolution
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()

        # Normalize to [0,1]
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, output.detach(), probabilities.detach()

    def close(self):
        if self.forward_handle:
            self.forward_handle.remove()
        if self.backward_handle:
            self.backward_handle.remove()


def enhance_vessels(image_np: np.ndarray) -> np.ndarray:
    """
    Retinal Vessel Enhancement using CLAHE and Frangi filter on green channel.
    Returns normalized vessel response map (float32 [0.0, 1.0]).
    """
    # Retinal vessels are particularly visible in the green channel
    green = image_np[:, :, 1]
    green_u8 = green.astype(np.uint8) if green.dtype != np.uint8 else green

    # CLAHE contrast normalization
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(green_u8).astype(np.float32) / 255.0

    # Vessels are dark ridges on the retinal background
    vessel_response = frangi(
        enhanced,
        sigmas=range(1, 5),
        black_ridges=True
    )

    # Normalize to 0-1
    vmin = vessel_response.min()
    vmax = vessel_response.max()

    if vmax > vmin:
        vessel_response = (vessel_response - vmin) / (vmax - vmin)
    else:
        vessel_response = np.zeros_like(vessel_response)

    return vessel_response
