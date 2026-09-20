"""
Inference & Report Generation Engine
SIH26038 — Explainable AI-Based Diabetic Retinopathy Screening
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

from src.transforms import val_transform, pad_to_square
from src.quality_assessment import assess_image_quality
from src.explainability import GradCAM, enhance_vessels

STAGE_A_PATH = r"D:\diabetic\models\efficientnet_b0_best.pth"
STAGE_B_PATH = r"D:\diabetic\models\efficientnet_b0_stage_b_best.pth"

GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

_STAGE_A_MODEL = None
_STAGE_B_MODEL = None
_DEVICE = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def load_models():
    """
    Load locked Stage A and Stage B models into memory.
    """
    global _STAGE_A_MODEL, _STAGE_B_MODEL
    device = get_device()

    if _STAGE_A_MODEL is None:
        model_a = efficientnet_b0(weights=None)
        in_features = model_a.classifier[1].in_features
        model_a.classifier[1] = nn.Linear(in_features, 5)

        ckpt_a = torch.load(STAGE_A_PATH, map_location=device)
        state_dict_a = ckpt_a.get("model_state_dict", ckpt_a)
        model_a.load_state_dict(state_dict_a)
        model_a = model_a.to(device)
        model_a.eval()
        _STAGE_A_MODEL = model_a

    if _STAGE_B_MODEL is None:
        model_b = efficientnet_b0(weights=None)
        in_features = model_b.classifier[1].in_features
        model_b.classifier[1] = nn.Linear(in_features, 3)

        ckpt_b = torch.load(STAGE_B_PATH, map_location=device)
        state_dict_b = ckpt_b.get("model_state_dict", ckpt_b)
        model_b.load_state_dict(state_dict_b)
        model_b = model_b.to(device)
        model_b.eval()
        _STAGE_B_MODEL = model_b

    return _STAGE_A_MODEL, _STAGE_B_MODEL


def crop_to_circle(img_np: np.ndarray, bg_color=(11, 29, 26), border_color=(16, 185, 129)) -> np.ndarray:
    """
    Clips RGB image into a circle with card background and subtle border ring.
    """
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1]) - 3

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    output = np.full_like(img_np, bg_color, dtype=np.uint8)
    output[mask] = img_np[mask]

    # Draw smooth border ring
    cv2.circle(output, center, radius, border_color, 2, cv2.LINE_AA)
    return output


def run_pipeline(pil_image: Image.Image) -> dict:
    """
    Runs locked screening pipeline:
    1. Preprocessing & Quality Assessment
    2. Stage A inference
    3. Routing (G0/G1 -> non-referable; G2/G3/G4 -> Stage B)
    4. Grad-CAM on active model
    5. Retinal Vessel Enhancement (Frangi)
    6. Returns structured report results and generated images.
    """
    device = get_device()
    stage_a, stage_b = load_models()

    raw_rgb = pil_image.convert("RGB")

    # 1. Quality Assessment
    quality_res = assess_image_quality(np.array(raw_rgb))

    # 2. Exact Preprocessing — pass raw_rgb directly to val_transform
    input_tensor = val_transform(raw_rgb).unsqueeze(0).to(device)

    # Display image setup (square 512x512 matching transform structure)
    sq_img = pad_to_square(raw_rgb).resize((512, 512), Image.BILINEAR)
    orig_np = np.array(sq_img)

    # 3. Stage A Inference
    with torch.no_grad():
        out_a = stage_a(input_tensor)
        probs_a = F.softmax(out_a, dim=1)
        pred_a = probs_a.argmax(dim=1).item()
        conf_a = float(probs_a[0, pred_a].item())

    # 4. Routing Logic
    if pred_a < 2:
        # Non-referable
        referable = False
        final_grade = pred_a
        final_conf = conf_a
        stage_b_pred = None
        stage_b_conf = None
        active_model = stage_a
        target_class = pred_a

        grade_str = GRADE_LABELS[final_grade]
        explanation = (
            f"Stage A classified this image as {grade_str} and therefore non-referable. "
            f"Stage B was not required. Grad-CAM highlights regions that influenced "
            f"the Stage A prediction."
        )
    else:
        # Referable -> Stage B
        referable = True
        with torch.no_grad():
            out_b = stage_b(input_tensor)
            probs_b = F.softmax(out_b, dim=1)
            class_b = probs_b.argmax(dim=1).item()
            conf_b = float(probs_b[0, class_b].item())

        stage_b_pred = class_b + 2
        stage_b_conf = conf_b
        final_grade = stage_b_pred
        final_conf = stage_b_conf
        active_model = stage_b
        target_class = class_b

        grade_str = GRADE_LABELS[final_grade]
        conf_pct = f"{final_conf * 100:.2f}"
        explanation = (
            f"Stage A classified this image as referable. Stage B then estimated "
            f"{grade_str} with {conf_pct}% confidence. Grad-CAM highlights regions "
            f"that influenced the final severity prediction."
        )

    # 5. Grad-CAM on active model
    target_layer = active_model.features[-1]
    gcam = GradCAM(active_model, target_layer)
    cam, _, _ = gcam.generate(input_tensor, target_class=target_class)
    gcam.close()

    # Blend Grad-CAM with display image
    heatmap_u8 = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    gradcam_blend = (0.55 * orig_np + 0.45 * heatmap_rgb).astype(np.uint8)

    # 6. Retinal Vessel Enhancement
    vessel_map = enhance_vessels(orig_np)
    vessel_u8 = (vessel_map * 255).astype(np.uint8)
    vessel_rgb = cv2.cvtColor(vessel_u8, cv2.COLOR_GRAY2RGB)

    # 7. Circular clipped images
    circ_original = crop_to_circle(orig_np)
    circ_gradcam = crop_to_circle(gradcam_blend)
    circ_vessel = crop_to_circle(vessel_rgb)

    return {
        "image_quality": quality_res["quality_label"],
        "focus_score": quality_res["focus_score"],
        "final_grade": final_grade,
        "prediction_label": GRADE_LABELS[final_grade],
        "confidence": final_conf,
        "confidence_pct": f"{final_conf * 100:.2f}%",
        "referable": referable,
        "referral_str": "YES" if referable else "NO",
        "stage_a_pred": pred_a,
        "stage_a_conf": conf_a,
        "stage_b_pred": stage_b_pred,
        "stage_b_conf": stage_b_conf,
        "explanation": explanation,
        "original_np": orig_np,
        "gradcam_np": gradcam_blend,
        "vessel_np": vessel_rgb,
        "circ_original_np": circ_original,
        "circ_gradcam_np": circ_gradcam,
        "circ_vessel_np": circ_vessel
    }


def generate_static_report_image(results: dict, filename: str = "report.png") -> str:
    """
    Renders and exports a static PNG report under D:\\diabetic\\outputs\\unseen_demo\\
    """
    output_dir = r"D:\diabetic\outputs\unseen_demo"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    # Canvas dimensions
    W, H = 840, 960
    bg_color = (7, 22, 19)        # #071613
    card_color = (11, 29, 26)     # #0B1D1A
    border_color = (19, 61, 53)   # #133D35
    accent_color = (16, 185, 129)  # #10B981
    text_color = (240, 245, 243)
    muted_text = (150, 180, 170)
    amber_text = (245, 158, 11)

    canvas = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Card rectangle
    card_x1, card_y1, card_x2, card_y2 = 40, 40, W - 40, H - 40
    draw.rectangle([card_x1, card_y1, card_x2, card_y2], fill=card_color, outline=border_color, width=2)
    # Accent line on left
    draw.rectangle([card_x1, card_y1, card_x1 + 6, card_y2], fill=accent_color)

    # Fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_header = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 16)
        font_body = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = font_header = font_bold = font_body = font_small = ImageFont.load_default()

    # Title & Subheader
    draw.text((70, 70), "SIH26038 — Explainable DR Screening Report", font=font_title, fill=text_color)
    draw.text((70, 105), "Screening result", font=font_header, fill=muted_text)

    # Metrics Row
    draw.text((70, 150), "Image quality", font=font_small, fill=muted_text)
    draw.text((70, 170), results["image_quality"], font=font_bold, fill=text_color)

    draw.text((320, 150), "Prediction", font=font_small, fill=muted_text)
    draw.text((320, 170), results["prediction_label"], font=font_bold, fill=text_color)

    # Confidence Row
    draw.text((70, 215), "Confidence", font=font_small, fill=muted_text)
    draw.text((W - 140, 215), results["confidence_pct"], font=font_bold, fill=accent_color)

    # Progress bar
    bar_x1, bar_y1, bar_x2, bar_y2 = 70, 240, W - 70, 252
    draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=(20, 45, 40), outline=border_color)
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * (results["confidence"]))
    draw.rectangle([bar_x1, bar_y1, fill_x2, bar_y2], fill=accent_color)

    # 3 Image Panels
    panel_y = 280
    panel_w = 210
    gap = 25
    x_starts = [70, 70 + panel_w + gap, 70 + 2 * (panel_w + gap)]

    titles = ["Original", "AI Attention (Grad-CAM)", "Retinal Vessel Enhancement"]
    imgs_np = [results["circ_original_np"], results["circ_gradcam_np"], results["circ_vessel_np"]]

    for i in range(3):
        x = x_starts[i]
        draw.text((x, panel_y), titles[i], font=font_small, fill=muted_text)
        img_pil = Image.fromarray(imgs_np[i]).resize((panel_w, panel_w), Image.BILINEAR)
        canvas.paste(img_pil, (x, panel_y + 25))

    # Referral Row
    ref_y = panel_y + panel_w + 45
    ref_color = (239, 68, 68) if results["referable"] else accent_color
    draw.text((70, ref_y), f"Referral: {results['referral_str']}", font=font_title, fill=ref_color)

    # AI Explanation Section
    exp_y = ref_y + 45
    draw.text((70, exp_y), "AI Explanation", font=font_bold, fill=text_color)

    # Multiline wrap for explanation
    exp_text = results["explanation"]
    words = exp_text.split()
    lines = []
    curr_line = ""
    for w in words:
        test_line = curr_line + (" " if curr_line else "") + w
        if len(test_line) * 7.5 > (W - 140):
            lines.append(curr_line)
            curr_line = w
        else:
            curr_line = test_line
    if curr_line:
        lines.append(curr_line)

    line_y = exp_y + 25
    for l in lines:
        draw.text((70, line_y), l, font=font_body, fill=muted_text)
        line_y += 20

    # Disclaimer Section
    disc_y = H - 85
    draw.text((70, disc_y), "Not a medical diagnosis.", font=font_small, fill=amber_text)
    draw.text((70, disc_y + 18), "Ophthalmic evaluation recommended.", font=font_small, fill=amber_text)

    canvas.save(out_path, "PNG")
    return out_path
