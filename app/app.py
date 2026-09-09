import os
import sys
import uuid

import cv2
from flask import Flask, render_template, request

# allow `from src...` imports when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import UPLOAD_DIR, GRADCAM_DIR
from src.predict import predict_image
from src.quality_assessment import assess_quality

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    file_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1] or ".png"
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    file.save(upload_path)

    image_bgr = cv2.imread(upload_path)
    if image_bgr is None:
        return render_template("index.html", error="Could not read that image file")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- Quality gate: this is the piece the original draft never
    # actually wired into the Flask route ---
    quality_result = assess_quality(image_rgb)
    if quality_result["quality"] == "POOR":
        return render_template(
            "index.html",
            quality_issue=quality_result,
            uploaded_image=f"/uploads/{file_id}{ext}",
        )

    # --- Prediction + Grad-CAM ---
    result = predict_image(upload_path, with_gradcam=True)

    gradcam_filename = f"{file_id}_gradcam.png"
    gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor(result["gradcam_image"], cv2.COLOR_RGB2BGR))

    report = {
        "image_quality": quality_result,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"] * 100, 1),
        "class_id": result["class_id"],
    }

    return render_template(
        "index.html",
        result=report,
        uploaded_image=f"/uploads/{file_id}{ext}",
        gradcam_image=f"/static/gradcam/{gradcam_filename}",
    )


# serve uploaded originals (they live outside /static)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
