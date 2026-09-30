Explainable AI-Based Diabetic Retinopathy Screening
Smart India Hackathon 2026 — SIH26038 | MathWorks
An explainable AI pipeline for diabetic retinopathy (DR) screening and referral triage from retinal fundus photographs.
The system is designed around a practical screening question:
> **Does this patient require specialist attention?**
Rather than relying only on five-class DR classification, the system uses a two-stage architecture that separates referable-disease detection from severity estimation.
---
Overview
```text
Fundus Image
     │
     ▼
Image Quality Assessment
     │
     ▼
Retinal Vessel Enhancement
     │
     ▼
Stage A — Referral Gate
     │
     ├── Grade 0/1 → Non-Referable
     │
     └── Grade 2/3/4 → Referable
                         │
                         ▼
                  Stage B — Severity
                         │
                         ▼
                      Grad-CAM
                         │
                         ▼
                  Referral Report
```
The final output combines predicted severity, referral status, confidence, and visual model attention into a clinician-reviewable report.
---
What We Built
1. Image Quality Assessment
A Laplacian-variance based focus metric identifies potentially low-quality or blurred retinal images before inference.
Current threshold: `10.0` — provisional demonstration threshold, not clinically calibrated.
2. Retinal Vessel Enhancement
A classical image-processing pipeline provides an additional structural representation:
```text
Green Channel → CLAHE → Frangi Vesselness
```
3. Two-Stage DR Classification
Stage A — Referral Gate
EfficientNet-B0 based five-class model:
```text
Grade 0 / 1 → Non-Referable
Grade 2 / 3 / 4 → Referable
```
Stage B — Severity Classification
A specialised EfficientNet-B0 model operates only on referable cases:
```text
Grade 2 → Moderate DR
Grade 3 → Severe DR
Grade 4 → Proliferative DR
```
4. Explainability
Integrated Grad-CAM visualizes image regions influencing model predictions.
Grad-CAM is treated as model attention, not validated lesion localization.
5. Automated Reporting
The system generates a visual screening report containing:
Image quality
Predicted DR grade
Confidence
Referral status
Grad-CAM visualization
Retinal vessel enhancement
AI explanation
A Streamlit interface provides the interactive demonstration.
---
Results
Evaluation was performed on a 733-image stratified validation split from the APTOS 2019 labelled training data.
Metric	Result
Referable DR Sensitivity	90.94%
Referable DR Specificity	94.48%
Five-Class Accuracy	82.95%
Stage B Accuracy	79.53%
The referral gate meets the SIH26038 target of >90% sensitivity and >85% specificity on this internal validation split.
> **Important:** These are internal validation results from a single stratified split. They do not establish clinical performance or generalisation to real-world Indian clinical populations.
---
Dataset
APTOS 2019 Blindness Detection
3,662 labelled training images
5 ICDR severity classes
Stratified 80/20 split
2,929 training images
733 validation images
Validation distribution
Grade	Clinical interpretation	Images
0	No DR	361
1	Mild NPDR	74
2	Moderate NPDR	200
3	Severe NPDR	39
4	Proliferative DR	59
APTOS provides image-level severity labels but does not provide pixel-level lesion annotations. Therefore, supervised lesion segmentation is not claimed in this implementation.
---
Architecture
Baseline
The initial baseline used an ImageNet-pretrained EfficientNet-B0 for five-class DR classification.
The baseline achieved 82.95% validation accuracy, but showed substantial confusion toward Grade 2, particularly for minority severe classes.
This motivated the two-stage referral-oriented architecture.
Two-Stage Design
```text
                    DR Fundus Image
                           │
                           ▼
                  Image Quality Check
                           │
                           ▼
                  ┌─────────────────┐
                  │     Stage A     │
                  │ Referral Gate   │
                  └────────┬────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
          Non-Referable           Referable
            Grade 0/1              Grade 2/3/4
                                      │
                                      ▼
                              ┌───────────────┐
                              │    Stage B    │
                              │   Severity    │
                              └───────┬───────┘
                                      │
                                      ▼
                                  Grad-CAM
                                      │
                                      ▼
                               Referral Report
```
The purpose is to make referral triage the primary screening decision, while severity estimation remains a secondary task.
---
Preprocessing
The same geometry-preserving preprocessing is used for model inference and Grad-CAM rendering:
```text
Raw Fundus Image
       ↓
Resize shortest side → 512
       ↓
Pad to square
       ↓
Resize → 512 × 512
       ↓
ImageNet Normalisation
       ↓
Model Input
```
Padding preserves the original aspect ratio and avoids direct anisotropic resizing of retinal images.
---
Model Configuration
Parameter	Stage A	Stage B
Architecture	EfficientNet-B0	EfficientNet-B0
Pretraining	ImageNet	ImageNet
Input	512 × 512	512 × 512
Classes	5	3
Optimiser	AdamW	AdamW
Learning Rate	1e-4	1e-4
Weight Decay	1e-4	1e-4
Epochs	10	10
Selected Checkpoint	Epoch 6	Epoch 7
---
Explainability
Grad-CAM is applied to the final spatial feature layer of EfficientNet-B0.
The resulting heatmap represents:
> **Regions of the image that influenced the model prediction.**
The project deliberately does not describe Grad-CAM as lesion detection.
Without pixel-level lesion ground truth, the highlighted regions cannot be claimed to correspond specifically to microaneurysms, hemorrhages, or exudates.
---
Retinal Structure Analysis
Vessel Enhancement — Implemented
```text
RGB Fundus
    ↓
Green Channel
    ↓
CLAHE
    ↓
Frangi Vesselness
    ↓
Vessel Response Map
```
Optic Disc Localisation — Prototype
A classical Hough-based detector was investigated.
Initial testing produced detections on 3/5 images, with one confirmed false positive. The detector was not sufficiently stable and was therefore excluded from the runnable inference pipeline.
Not Currently Implemented
Microaneurysm detection
Exudate segmentation
Hemorrhage classification
Neovascularisation detection
Fovea localisation
---
Implementation Status
Implemented and Verified
[x] Dataset audit
[x] Stratified train/validation split
[x] Geometry-preserving preprocessing
[x] Image quality assessment
[x] Retinal vessel enhancement
[x] Stage A referral classification
[x] Stage B severity classification
[x] Grad-CAM explainability
[x] Referral logic
[x] Automated report generation
[x] Streamlit demonstration
[x] End-to-end inference pipeline
Prototype / Future Work
[ ] Robust optic-disc localisation
[ ] Lesion-level segmentation
[ ] Fovea localisation
[ ] MATLAB / ONNX deployment validation
[ ] Simulink deployment simulation
[ ] External clinical validation
[ ] Prospective Indian clinical evaluation
---
Limitations
This is a research prototype, not a clinical diagnostic system.
Current limitations include:
Single internal validation split
No external dataset validation
No prospective clinical validation
Grade 3 and Grade 4 severity classification remains challenging
Quality threshold is not clinically calibrated
Grad-CAM has not been quantitatively validated against lesion masks
MATLAB / ONNX bridge is not yet validated
Simulink deployment simulation is not yet implemented
---
Intended Use
The system is intended as a screening and referral decision-support tool.
It is not intended to:
Replace an ophthalmologist
Provide autonomous diagnosis
Determine treatment
Serve as the sole basis for clinical decisions
Clinical deployment would require external validation, prospective evaluation, calibration, safety assessment, and appropriate regulatory approval.
---
Technology Stack
Python 3.11
PyTorch
EfficientNet-B0
OpenCV
torchvision
PIL
Grad-CAM
CLAHE
Frangi Vesselness Filtering
Streamlit
CUDA
NVIDIA RTX 3050 6GB
---
Project Structure
```text
diabetic/
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── quality_assessment.py
│   ├── explainability.py
│   └── predict.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/
│   ├── efficientnet_b0_best.pth
│   └── efficientnet_b0_stage_b_best.pth
│
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```
---
Run the Demo
```bash
git clone <repository-url>
cd diabetic

python -m venv .venv
```
Windows
```powershell
.venv\Scripts	ctivate
pip install -r requirements.txt
python -m streamlit run app.py
```
The Streamlit application launches the interactive retinal-screening demonstration.
---
Future Direction
The next stage is to move from a classification prototype toward a clinically evaluated screening workflow:
```text
Better Clinical Data
        ↓
External Validation
        ↓
Lesion-Level Evidence
        ↓
Calibration
        ↓
Prospective Pilot
        ↓
Clinical Deployment
```
The long-term objective is not autonomous diagnosis, but scalable, explainable screening and referral support in settings where specialist capacity is limited.
---
References
APTOS 2019 Blindness Detection — Kaggle
Tan, M. & Le, Q. V. — EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019
Selvaraju, R. R. et al. — Grad-CAM: Visual Explanations from Deep Networks, ICCV 2017
Frangi, A. F. et al. — Multiscale Vessel Enhancement Filtering, MICCAI 1998
International Council of Ophthalmology — ICDR Diabetic Retinopathy Severity Scale
---
Project
Smart India Hackathon 2026 — SIH26038
Explainable AI-Based Diabetic Retinopathy Screening
> Research prototype — screening and referral decision support, not a clinical diagnostic system.