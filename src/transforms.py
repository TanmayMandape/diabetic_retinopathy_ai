from PIL import ImageOps
from torchvision import transforms


def pad_to_square(image):
    """
    Pad an image to a square while preserving
    the original aspect ratio.
    """

    width, height = image.size
    max_side = max(width, height)

    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2
    pad_right = max_side - width - pad_left
    pad_bottom = max_side - height - pad_top

    return ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(0, 0, 0)
    )


train_transform = transforms.Compose([
    transforms.Resize(512),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.Lambda(pad_to_square),

    transforms.Resize((512, 512)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize(512),

    transforms.Lambda(pad_to_square),

    transforms.Resize((512, 512)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])