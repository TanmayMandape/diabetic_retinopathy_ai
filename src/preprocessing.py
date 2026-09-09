import os
import cv2
import numpy as np


def resolve_image_path(image_dir, id_code):
    """
    APTOS-style CSVs store id_code WITHOUT the file extension.
    This function works whether id_code already has an extension
    or not, so dataset.py never throws FileNotFoundError over a
    missing '.png'.
    """
    id_code = str(id_code)
    if os.path.splitext(id_code)[1]:  # already has an extension
        candidate = os.path.join(image_dir, id_code)
        if os.path.exists(candidate):
            return candidate

    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(image_dir, id_code + ext)
        if os.path.exists(candidate):
            return candidate

    # fall back to .png even if it doesn't exist yet, so the
    # caller gets a sensible error message instead of a silent None
    return os.path.join(image_dir, id_code + ".png")


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # guard against a degenerate crop (e.g. an all-black image)
    if w == 0 or h == 0:
        return image

    return image[y:y + h, x:x + w]


def load_image(image_path, image_size=224):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_black_borders(image)
    image = cv2.resize(image, (image_size, image_size))

    return image
