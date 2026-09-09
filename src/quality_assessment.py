import cv2
import numpy as np

# NOTE: these thresholds are initial heuristic values, not
# medically validated cutoffs. Tune them against your own dataset
# before relying on them for anything real.
SHARPNESS_MIN = 50
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 220
BLACK_RATIO_MAX = 0.70


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


def calculate_black_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    black_pixels = np.sum(gray < 10)
    total_pixels = gray.size
    return black_pixels / total_pixels


def assess_quality(image):
    sharpness = calculate_sharpness(image)
    brightness = calculate_brightness(image)
    black_ratio = calculate_black_ratio(image)

    issues = []

    if sharpness < SHARPNESS_MIN:
        issues.append("Image may be blurry")

    if brightness < BRIGHTNESS_MIN:
        issues.append("Image may be too dark")

    if brightness > BRIGHTNESS_MAX:
        issues.append("Image may be overexposed")

    if black_ratio > BLACK_RATIO_MAX:
        issues.append("Insufficient retinal area")

    quality_ok = len(issues) == 0

    return {
        "quality": "GOOD" if quality_ok else "POOR",
        "sharpness": sharpness,
        "brightness": brightness,
        "black_ratio": black_ratio,
        "issues": issues,
    }
