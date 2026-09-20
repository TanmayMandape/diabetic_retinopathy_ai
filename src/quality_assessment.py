"""
Quality Assessment Module
SIH26038 — Explainable AI-Based Diabetic Retinopathy Screening
"""

import cv2
import numpy as np

FOCUS_THRESHOLD = 10.0


def calculate_focus_variance(image_np: np.ndarray) -> float:
    """
    Calculate Laplacian variance as a focus/sharpness score.
    Higher value -> sharper image
    Lower value -> more blurred image
    """
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def assess_image_quality(image_np: np.ndarray, threshold: float = FOCUS_THRESHOLD) -> dict:
    """
    Pass/Reject quality assessment.
    """
    focus_score = calculate_focus_variance(image_np)
    passed = focus_score >= threshold
    return {
        "focus_score": focus_score,
        "threshold": threshold,
        "passed": passed,
        "quality_label": "Good" if passed else "Low Sharpness",
        "status": "PASS" if passed else "REJECT"
    }
