import torch
import torch.nn.functional as F

from src.config import MODEL_PATH, CLASS_NAMES
from src.model import create_model
from src.transforms import val_transform
from src.preprocessing import load_image
from src.gradcam import generate_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def get_model():
    """Lazy-load so importing this module doesn't require the
    weights file to exist yet (useful for the Flask app's startup)."""
    global _model
    if _model is None:
        _model = create_model(num_classes=5)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        _model.load_state_dict(state_dict)
        _model.to(device)
        _model.eval()
    return _model


def predict_image(image_path, with_gradcam=True):
    model = get_model()

    # Same crop+resize pipeline used during training -- avoids
    # train/serve skew.
    image = load_image(image_path)

    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, prediction = probabilities.max(dim=1)

    result = {
        "class_id": prediction.item(),
        "prediction": CLASS_NAMES[prediction.item()],
        "confidence": confidence.item(),
        "probabilities": probabilities.cpu().numpy()[0].tolist(),
    }

    if with_gradcam:
        result["gradcam_image"] = generate_gradcam(model, image)

    return result
