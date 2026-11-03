# MorphoAgroNet

### A Geometry-Guided Multi-Scale Deep Learning Framework for Fine-Grained Plant Disease Recognition

## 1. Overview

**MorphoAgroNet** is a geometry-guided deep learning framework designed for fine-grained plant disease classification under field conditions.

The framework integrates lesion morphology analysis, multi-scale feature extraction, geometry-guided attention, hierarchical feature fusion, and explainable AI to improve disease recognition.

It also incorporates environment-adaptive robust learning to study model performance under different image conditions.


## 3. Key Components

- **Field-Aware Preprocessing:** Image enhancement using CIELAB color conversion, CLAHE, bilateral filtering, and normalization.
- **Lesion Morphology Estimation:** Generates lesion pseudo-masks and extracts geometric descriptors.
- **Multi-Scale Lesion Geometry Encoding (MLGE):** Captures lesion patterns using multiple convolutional scales.
- **Geometry-Guided Attention (GGAB):** Incorporates lesion geometry into spatial feature attention.
- **Hierarchical Multi-Scale Fusion (HMF):** Combines features from different backbone stages.
- **Fine-Grained Semantic Separation (FGSS):** Uses prototype-based learning to improve class discrimination.
- **Environment-Adaptive Robust Learning (EARL):** Studies robustness under simulated environmental image variations.
- **Explainable AI:** Supports Grad-CAM, attention visualization, and lesion-mask visualization.

## 4. Repository Structure

```text
MorphoAgroNet/
│
├── configs/
│   ├── default.yaml
│   ├── ablations.yaml
│   ├── plantdoc.yaml
│   ├── plantvillage.yaml
│   └── pathology2020.yaml
│
├── morphoagronet/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   ├── train_baselines.py
│   ├── run_ablations.py
│   ├── run_robustness.py
│   ├── run_transfer.py
│   ├── run_statistics.py
│   ├── profile_model.py
│   └── evaluate_localization.py
│
├── tests/
├── README.md
├── HOW_TO_RUN.txt
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## 5. Datasets

The framework supports the following plant disease datasets:

| Dataset | Purpose |
|---|---|
| PlantVillage | Main plant disease classification |
| PlantDoc | Field-oriented disease classification |
| Plant Pathology 2020 | Fine-grained disease classification |
| PlantSeg | Lesion localization evaluation |
| Plant Pathology 2021 | External evaluation |

Dataset files must be organized according to the folder structures and configuration requirements of the project.

## 6. Installation

### Clone the Repository

```bash
git clone https://github.com/drbrkou-sys/MorphoAgroNet.git
cd MorphoAgroNet
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 7. Execution

Run the following commands from the project root directory.

### Step 1: Prepare Dataset

```bash
python scripts/prepare_data.py --help
```

### Step 2: Train MorphoAgroNet

```bash
python scripts/train.py --help
```

### Step 3: Evaluate the Model

```bash
python scripts/evaluate.py --help
```

### Step 4: Generate Predictions

```bash
python scripts/predict.py --help
```

### Step 5: Generate Explainability Visualizations

```bash
python scripts/explain.py --help
```

Use the corresponding configuration files and command-line arguments for the required dataset and experiment. Refer to `HOW_TO_RUN.txt` for execution details.

## 6. Evaluation

The framework provides the following evaluation capabilities:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Macro ROC-AUC
- Intersection over Union (IoU)
- Dice Score
- Pointing Game
- Energy Pointing
- Boundary F1-score

Additional experiment scripts support baseline comparison, ablation studies, robustness evaluation, transfer evaluation, statistical analysis, and model profiling.

## 7. Requirements

- Python 3.10 or compatible environment
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- PyYAML
- Matplotlib
- Seaborn
- Pillow
- tqdm

Install the dependencies using `requirements.txt`.

## 8. License

This project is released under the **MIT License**. See the `LICENSE` file for details.

## 9. Repository

**GitHub:** https://github.com/drbrkou-sys/MorphoAgroNet

```bash
git clone https://github.com/drbrkou-sys/MorphoAgroNet.git
```

## 10. Citation

If you use this framework in your research, please cite the repository.

A formal citation and DOI can be added after the Zenodo record is published.

## 11. Contact

For questions, research collaboration, or technical discussions, please contact the project authors through the GitHub repository.
