# SmartClaim AI

**Intelligent Vehicle Damage Detection & Insurance Claim Estimator**

Upload a photo of a damaged car, enter a few vehicle and policy details, and SmartClaim AI
detects the damage, identifies the affected parts and produces an itemised repair estimate
with the amount the insurer pays and the amount the customer pays, following IRDAI
motor own-damage rules.

> Mini project — B.E. Information Technology, A.P. Shah Institute of Technology, Thane (AY 2026-27)
>
> **Team:** Sunny Gupta (23104136), Pranav Ghodke (23104115), Aditya Chavan (23104090)
> **Guide:** Ms. Snehal Mali · **HoD:** Dr. Kiran Deshpande

---

## How it works

```
photo ──► YOLOv8n-cls classifier ──► front / rear + normal / breakage / crushed
      ──► YOLO11 parts segmenter ──► 21 car parts            ┐
      ──► YOLO11 damage segmenter ─► 8 damage types         ├─► damage → part attribution
                                                             ┘
          ──► repair plan (low / high) ──► pricing ──► IRDAI depreciation, deductible,
                                                       total-loss check ──► claim estimate
```

1. **Classify** the photo into one of 6 classes (`F_Breakage`, `F_Crushed`, `F_Normal`,
   `R_Breakage`, `R_Crushed`, `R_Normal`).
2. **Segment** car parts and damage regions; each damage mask is assigned to the part it
   overlaps most (at least 20% overlap).
3. **Plan repairs**: every damage type maps to a low and high action (paint / repair /
   replace). Large dents (>30% of the part) are replaced in the high plan, and the high plan
   also adds parts typically damaged behind the panels for the classified severity.
4. **Price** each part: base price × car-segment multiplier, labour hours × city-tier rate,
   paint per panel, plus 18% GST.
5. **Settle** the claim with IRDAI rules: material/age-based depreciation (plastic 50%,
   fibreglass 30%, glass nil, metal 0–50% by age, 50% on paint material), compulsory
   deductible (₹1,000 up to 1500 cc, ₹2,000 above) plus voluntary deductible,
   zero-depreciation add-on, and constructive total loss when the bill exceeds 75% of IDV.

All prices and rules live in two editable files —
[`pricing.json`](car-estimator/estimator/pricing.json) and
[`damage_rules.json`](car-estimator/estimator/damage_rules.json) — so every rupee in an
estimate can be traced and updated without retraining. Prices are indicative and should be
validated against OEM price lists or workshop quotes before real use.

## Results

| Model | Metric | Score |
|---|---|---|
| Damage classifier (YOLOv8n-cls) | Validation accuracy (575 images) | **80.35%** |
| Car parts segmenter (YOLO11n-seg, 21 classes) | Mask mAP@0.5 | **0.81** |
| Damage segmenter (YOLO11n-seg, 8 classes) | Mask mAP@0.5 | 0.15 |

YOLOv8n-cls was selected after comparing it with a baseline CNN (48.70%), CNN with
regularisation (46.61%), EfficientNet-B0 (60.70%) and ResNet50 (73.91%).

## Project structure

```
Car-Detection/
├── car-estimator/                 # the application
│   ├── app.py                     # Gradio web app
│   ├── estimator/
│   │   ├── vision.py              # model loading, inference, damage → part attribution
│   │   ├── pricing.py             # repair plans, line-item pricing, claim settlement
│   │   ├── policy.py              # IRDAI depreciation, deductibles, total-loss threshold
│   │   ├── cli.py                 # command-line estimator
│   │   ├── pricing.json           # part prices, labour rates, segment multipliers, GST
│   │   └── damage_rules.json      # damage → action rules, severity scenarios
│   ├── tests/test_estimator.py    # unit tests for pricing and policy rules
│   └── training/                  # claims dataset analysis notebook + CSV
├── damage-prediction/training/
│   └── DamagePartDetection.ipynb  # classifier training and comparison (CNN, EfficientNet, ResNet50, YOLOv8)
├── damage-type/training/          # part and damage segmentation training (see its README)
└── report/                        # project report (docx/pdf), presentation (pptx/pdf), figures
```

## Setup

Requires Python 3.12. On Apple Silicon the models run on the GPU (MPS); CUDA and CPU also work.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics gradio pytest
# for the training notebooks:
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn xgboost statsmodels huggingface_hub jupyter
```

### Model weights and datasets

Datasets and trained weights are **not committed** (see `.gitignore`). The app expects:

| Weights | Produced by |
|---|---|
| `runs/classify/yolo_runs/car_damage_cls/weights/best.pt` | `damage-prediction/training/DamagePartDetection.ipynb` |
| `runs/segment/car_parts/weights/best.pt` | `damage-type/training/train_seg.py` |
| `runs/segment/car_damage/weights/best.pt` | `damage-type/training/train_seg.py` |

- **Classifier:** unzip the front/rear damage dataset (`archive.zip`) into
  `damage-prediction/training/` so it contains `dataset/<class>/…jpg`, then run the notebook.
  It builds `yolo_dataset/` (75% train / 25% validation) and trains YOLOv8n-cls.
- **Segmentation:** follow [`damage-type/training/README.md`](damage-type/training/README.md)
  to download the DrBimmer car parts and damage dataset and train both models.

## Usage

### Web app

```bash
cd car-estimator
python app.py
```

Open <http://127.0.0.1:7860>, upload a photo (or pick a sample), set the car segment, city
tier, vehicle age, engine cc, IDV, zero-dep and voluntary deductible, then click
**Estimate claim**. The app shows the detected damage overlaid on the photo and a low/high
estimate with a part-by-part breakdown.

### Command line

```bash
cd car-estimator
python -m estimator.cli photo.jpg --segment sedan --city-tier tier1 \
    --vehicle-age 3 --engine-cc 1200 --idv 600000
```

Options: `--zero-dep`, `--voluntary-deductible N`, `--json` for machine-readable output,
and `--cls-weights / --parts-weights / --damage-weights` to use other models.

### Tests

```bash
cd car-estimator
python -m pytest -q
```

## Example

Sedan, tier-1 city, 3 years old, 1200 cc, IDV ₹6,00,000, photo classified as front-crushed:

| | Repair bill | Insurer pays | Customer pays |
|---|---:|---:|---:|
| Low estimate (visible damage only) | ₹4,460 | ₹2,903 | ₹1,558 |
| High estimate (+ typical hidden damage) | ₹70,304 | ₹54,808 | ₹15,496 |

## Tech stack

Python · PyTorch · Ultralytics YOLOv8 / YOLO11 · Gradio · scikit-learn · pandas · Jupyter

## Disclaimer

Estimates are indicative. Depreciation follows the published IRDAI schedule, but a
surveyor's inspection decides the final claim.
