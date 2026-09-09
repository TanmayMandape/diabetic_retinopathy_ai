# Diabetic Retinopathy AI — MVP

An AI-based screening prototype for **Diabetic Retinopathy (DR)** classification from retinal fundus images.

The project uses **EfficientNet-B0 transfer learning** to classify fundus images into five diabetic retinopathy severity levels and provides an image-quality check, prediction confidence, and Grad-CAM visualization through a Flask web interface.

> **Important:** This is an MVP/research prototype and is **not a clinically validated diagnostic system or a substitute for professional medical advice.**

---

## Project Pipeline

```text
Fundus Image
     │
     ▼
Image Quality Assessment
     │
     ├── Poor Quality ──► Re-upload Image
     │
     ▼
Preprocessing
     │
     ▼
EfficientNet-B0
     │
     ▼
5-Class Prediction
     │
     ├── Prediction
     ├── Confidence
     └── Grad-CAM
```

---

## Step 0 — Get the Dataset

Download the **APTOS 2019 Blindness Detection** dataset from Kaggle:

https://www.kaggle.com/c/aptos2019-blindness-detection/data

Unzip it so the project contains:

```text
data/
├── train.csv
└── train_images/
    ├── *.png
    └── ...
```

The project expects the APTOS training CSV to contain:

```text
id_code
diagnosis
```

The `id_code` values in the original APTOS `train.csv` do not include `.png`. The project's `resolve_image_path()` function handles both cases, with or without the extension.

---

## Step 1 — Install Dependencies

From the project root:

```bash
cd diabetic-retinopathy-ai
python3 -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Check the PyTorch installation:

```bash
python3 -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

If CUDA returns `False`, the project can still run on CPU, although training will be significantly slower.

With approximately 3,600 APTOS images, CPU-only training may take roughly 15–30+ minutes per epoch depending on the hardware.

For an initial sanity check, consider reducing the number of epochs before running the full training process.

---

## Step 2 — Verify the Dataset

Check the CSV:

```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/train.csv'); print(df.head()); print(df['id_code'].iloc[0])"
```

Confirm that the CSV contains:

```text
id_code
diagnosis
```

The code supports `id_code` values both with and without the `.png` extension.

---

## Step 3 — Train the Model

Run:

```bash
python3 -m src.train
```

The training pipeline performs:

* Stratified train/validation/test splitting
* Image preprocessing
* Class-imbalance handling using class-weighted loss
* EfficientNet-B0 transfer learning
* Model validation
* Best-checkpoint selection
* Final test-set evaluation
* Classification report
* Macro F1 calculation

The best checkpoint is saved to:

```text
models/dr_model.pth
```

### Changing Training Parameters

The default training configuration can be modified in `src/train.py`.

For example:

```python
main(num_epochs=5, batch_size=8)
```

---

## Step 4 — Sanity-Check a Single Prediction

Before running the web application, a single image can be tested directly:

```bash
python3 -c "
from src.predict import predict_image
r = predict_image('data/train_images/<some_id_code>.png', with_gradcam=False)
print(r['prediction'], r['confidence'])
"
```

This provides a quick check that the trained model can load successfully and generate a prediction.

---

## Step 5 — Run the Web App

Start the Flask application:

```bash
python3 app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

Upload a retinal fundus image.

The application performs:

1. Image-quality assessment
2. Image preprocessing
3. Diabetic retinopathy prediction
4. Confidence calculation
5. Grad-CAM visualization
6. Medical disclaimer display

Poor-quality images are rejected by the quality gate with a request to upload a better image.

---

# Model Evaluation

The model was trained for **10 epochs** on the APTOS 2019 dataset and evaluated on a held-out test set.

## Dataset Split

| Dataset    |    Images |
| ---------- | --------: |
| Total      | **3,662** |
| Training   | **2,563** |
| Validation |   **549** |
| Test       |   **550** |

The split corresponds to approximately:

* **70% training**
* **15% validation**
* **15% test**

using a random state of `42`.

---

## Test-Set Results

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **80.00%** |
| Macro Precision |   **0.64** |
| Macro Recall    |   **0.66** |
| Macro F1        | **0.6483** |
| Weighted F1     |   **0.80** |

### Per-Class Results

| Class                              | Precision | Recall |       F1 |
| ---------------------------------- | --------: | -----: | -------: |
| No Diabetic Retinopathy            |      0.97 |   0.97 | **0.97** |
| Mild Diabetic Retinopathy          |      0.55 |   0.61 | **0.58** |
| Moderate Diabetic Retinopathy      |      0.76 |   0.71 | **0.73** |
| Severe Diabetic Retinopathy        |      0.37 |   0.48 | **0.42** |
| Proliferative Diabetic Retinopathy |      0.57 |   0.52 | **0.55** |

These results correspond to the specific reported training run and may vary with changes to the dataset, preprocessing, configuration, random seed, or training process.

---

# Training Configuration

The evaluated model used the following configuration:

| Parameter          | Configuration                |
| ------------------ | ---------------------------- |
| Architecture       | EfficientNet-B0              |
| Pretrained Weights | ImageNet                     |
| Input Size         | 224 × 224                    |
| Epochs             | 10                           |
| Batch Size         | 16                           |
| Learning Rate      | 1e-4                         |
| Optimizer          | AdamW                        |
| Weight Decay       | 1e-4                         |
| Loss               | Class-Weighted Cross Entropy |
| Dataset Split      | 70% / 15% / 15%              |
| Random State       | 42                           |

The best model checkpoint is selected according to **validation loss** and saved as:

```text
models/dr_model.pth
```

---

# Training Progress

During the reported training run:

* Training accuracy increased from **58.84% in Epoch 1** to **91.07% in Epoch 10**.
* The lowest validation loss was **0.8863 at Epoch 3**.
* After Epoch 3, validation loss generally increased while training accuracy continued to improve.

This behavior indicates **some overfitting during the later training epochs**.

The checkpointing mechanism helps mitigate this by retaining the model corresponding to the **best validation loss**, rather than automatically using the final epoch.

---

# What the Current MVP Demonstrates

The current implementation combines:

* Retinal fundus image preprocessing
* Image-quality assessment
* Five-class diabetic retinopathy classification
* EfficientNet-B0 transfer learning
* ImageNet pretrained weights
* Class-imbalance handling
* Stratified dataset splitting
* Validation-based model checkpointing
* Held-out test-set evaluation
* Prediction confidence
* Grad-CAM visualization
* Flask-based web interface

---

# Important Limitations

### 1. Clinical Validation

The model has **not undergone clinical validation or regulatory review**.

The reported metrics are based on the APTOS dataset and a specific training configuration. They should not be interpreted as evidence of clinical performance.

### 2. Image Quality Thresholds

The image-quality assessment uses heuristic thresholds for:

* Sharpness
* Brightness
* Black-area ratio

These thresholds are starting points and have not been clinically validated.

### 3. Class Performance

Performance is not uniform across the five classes. In particular, the **Severe Diabetic Retinopathy** class has substantially lower precision and F1 than the No-DR and Moderate-DR classes.

The macro F1 of **0.6483** provides a more informative view of this class imbalance than accuracy alone.

### 4. Data and Training Variability

Results may change depending on:

* Dataset split
* Random seed
* Image preprocessing
* Hyperparameters
* Hardware
* Training duration
* Model configuration

### 5. Uploaded Files

Uploaded files currently accumulate in:

```text
app/uploads/
```

and Grad-CAM outputs accumulate in:

```text
app/static/gradcam/
```

There is currently no automatic cleanup mechanism. For a long-running deployment, a cleanup or retention mechanism should be implemented.

---

# Disclaimer

This project is intended for **research, educational, and demonstration purposes only**.

It is a screening prototype and **not a medical diagnostic device**. Model predictions should not be used to diagnose, treat, or make clinical decisions about a patient.

---

## Model Checkpoint

The best-performing checkpoint from the evaluated training run is:

```text
models/dr_model.pth
```

The checkpoint is selected using validation loss rather than simply using the final training epoch.
