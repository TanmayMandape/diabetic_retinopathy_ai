# Diabetic Retinopathy AI — MVP (fixed version)

This is your original ChatGPT roadmap, corrected so the pieces actually
connect to each other. What was fixed vs. the original draft:

1. **Filename mismatch** — APTOS's `train.csv` stores `id_code` *without*
   `.png`. `src/preprocessing.py::resolve_image_path()` now handles both
   cases so you don't get `FileNotFoundError`.
2. **Dead crop code** — `crop_black_borders()` is now actually called by
   both training (`dataset.py`) and inference (`predict.py`), so train
   and serve use identical preprocessing.
3. **Quality gate not wired in** — `app.py`'s `/predict` route now calls
   `assess_quality()` and shows a "please re-upload" screen for poor
   images, instead of silently predicting on bad photos.
4. **Grad-CAM target layer resolved** — `model.features[-1]`, the
   standard EfficientNet-B0 conv block, set in `src/model.py`.
5. **Path inconsistency** — every script imports paths from
   `src/config.py` instead of hardcoding `../models/...` vs
   `models/...`.

## Step 0 — Get the dataset

Download the APTOS 2019 Blindness Detection dataset from Kaggle:
https://www.kaggle.com/c/aptos2019-blindness-detection/data

Unzip it so you have:
```
data/train.csv
data/train_images/*.png
```
(Both already-provided empty folders in this project.)

## Step 1 — Install dependencies

```bash
cd diabetic-retinopathy-ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Check PyTorch:
```bash
python3 -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```
CUDA `False` is fine — training just runs slower on CPU. With ~3,600
APTOS images and CPU-only, expect roughly 15–30+ min per epoch on a
typical laptop; consider a smaller `num_epochs` first to sanity-check.

## Step 2 — Verify your CSV matches the code's assumptions

```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/train.csv'); print(df.head()); print(df['id_code'].iloc[0])"
```
Confirm the columns are `id_code` and `diagnosis`. If your `id_code`
column *does* include `.png`, `resolve_image_path()` still handles it —
no changes needed either way.

## Step 3 — Train the model

```bash
python3 -m src.train
```
This runs the full pipeline: stratified train/val/test split, class
weighting for the imbalanced classes, 10 epochs of EfficientNet-B0
fine-tuning, and a final classification report + macro F1 on the held-
out test set. The best checkpoint (by validation loss) is saved to
`models/dr_model.pth`.

To change epoch count or batch size, edit the call at the bottom of
`src/train.py`, e.g.:
```python
main(num_epochs=5, batch_size=8)
```

## Step 4 — Sanity-check a single prediction (optional, before the web app)

```bash
python3 -c "
from src.predict import predict_image
r = predict_image('data/train_images/<some_id_code>.png', with_gradcam=False)
print(r['prediction'], r['confidence'])
"
```

## Step 5 — Run the web app

```bash
python3 app/app.py
```
Open http://127.0.0.1:5000 in your browser, upload a fundus image, and
you'll get: image-quality gate → prediction → confidence → Grad-CAM
heatmap → "not a medical diagnosis" disclaimer.

## Notes / known limitations

- The quality-assessment thresholds (sharpness, brightness, black-area
  ratio) are heuristic starting points, not clinically validated
  cutoffs — tune them against your own held-out images.
- Uploaded files accumulate in `app/uploads/` and
  `app/static/gradcam/` with no automatic cleanup — fine for a demo,
  add a cron/cleanup job before anything long-running.
- This is a screening aid, not a diagnostic device. It hasn't been
  through clinical validation or regulatory review, so treat all
  outputs as a prototype signal, not medical advice.
