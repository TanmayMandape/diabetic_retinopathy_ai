import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, image_size=224):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = crop_black_borders(image)

    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    return image


def crop_black_borders(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    return image[y:y+h, x:x+w]
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
image_path = BASE_DIR / "data" / "train_images" / "000c1434d8d7.png"

image = load_image(str(image_path))
plt.imshow(image)
plt.axis("off")
plt.show()