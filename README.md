
# MorphoAgroNet

**Geometry-guided multi-scale lesion morphology for fine-grained plant disease recognition.**

Research software implementing the primary pipeline in the manuscript *A Geometry-Guided Deep Architecture Integrating Multi-Scale Lesion Morphology for Robust Fine-Grained Plant Disease Classification Under Field Conditions*.

## What is implemented

- Field-aware color/luminance preprocessing, constrained contrast enhancement and edge-preserving filtering.
- Annotation-free CIELAB lesion pseudo-label generation; weakly supervised convolutional soft-lesion estimator.
- Stop-gradient OpenCV contours, curvature statistics, lesion area, irregularity, asymmetry, spot density; differentiable three-scale CNN encoders (MLGE) with adaptive scale weighting.
- ImageNet ResNet-50 hierarchical feature taps, residual geometry-conditioned spatial attention (GGAB), adaptive hierarchical morphology fusion (HMF).
- Prototype EMA, compactness and margin losses, hard-negative separation (FGSS).
- Field-corruption training, paired embedding and symmetric KL consistency (EARL).
- Grad-CAM, aggregated geometry attention, integrated explanations, lesion mask agreement and independent-mask validation.
- Three primary dataset readers, 70/10/20 stratified manifests, seven standard classifiers and backbone-only comparison, ablations, robustness severity evaluation, source-only/10%-target adaptation, repeated-seed statistics, model profiling, classification inference.

## System requirements

Python 3.11 recommended; GPU with CUDA is strongly recommended for full datasets. CPU inference and smoke tests are supported but full experiments are expensive. PyTorch pretrained checkpoints may need network access on first use.

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt

python -m pip install -e . --no-deps

python -m pytest -q
```

For CUDA-enabled PyTorch, use a wheel/installation command appropriate to your driver and CUDA environment from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). pip install requirements.txt does not by itself guarantee CUDA support.

## Datasets

- PlantVillage: [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) . **Main paper experiment:** the 15 pepper/potato/tomato categories; other classes are filtered. The manuscript's 20,638-image subset requires checking the dataset version. Dataset layout: raw/plantvillage/\<class>/\<image>.jpg or nested train/class folders.

- PlantDoc: [https://github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) . A class-directory layout is required. If your download is annotation-only/object-detection structured, first arrange disease classification images into label-named subdirectories; do not use bounding-box labels as disease-class folders. Two-image spider-mite class excluded by the manuscript.

- Plant Pathology 2020 FGVC7: [https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data) . Place train.csv and images/*.jpg together. Four class indicators are read from train.csv; official unlabeled test is excluded.

- PlantSeg masks: prepare an independent CSV with image,mask,class_id for evaluate_localization.py. Each class_id is the class index of the model checkpoint, **not** the external PlantSeg integer code.

- Plant Pathology 2021: this work includes a generic external-image evaluation route; label-space correspondence must be specified manually before quantitative comparison.

**Important data integrity:** The supplied datasets are not bundled or automatically downloaded. File hashes and exact duplicates are checked before splitting; near-duplicate detection and crop/plant-group stratification are *not* automated. Review image overlap and source provenance before publication. The recreated split is a reproducible stratified split, not guaranteed byte-for-byte identical to the manuscript's reported class-wise counts.

## Quick start

```bash
python scripts/prepare_data.py --root data/raw/plantvillage --kind plantvillage --output data/manifests/plantvillage.csv

python scripts/prepare_data.py --root data/raw/plantdoc --kind plantdoc --output data/manifests/plantdoc.csv

python scripts/prepare_data.py --root data/raw/pathology2020 --kind pathology2020 --output data/manifests/pathology2020.csv

python scripts/train.py --manifest data/manifests/plantvillage.csv --epochs 100 --batch 32 --output results/plantvillage/seed42

python scripts/evaluate.py --checkpoint results/plantvillage/seed42/best.pt --manifest data/manifests/plantvillage.csv

python scripts/predict.py --checkpoint results/plantvillage/seed42/best.pt --image examples/leaf.jpg

python scripts/explain.py --checkpoint results/plantvillage/seed42/best.pt --image examples/leaf.jpg
```

To run without downloading pretrained weights, append --no-pretrained to training (not recommended for publication). To reduce physical batch size, use --batch 8 --grad-accum 4 (effective batch size 32); note batch normalization still sees microbatches of eight.

## Experiments

```bash
python scripts/train_baselines.py --manifest data/manifests/plantvillage.csv

python scripts/run_ablations.py --manifest data/manifests/plantvillage.csv

python scripts/run_robustness.py --checkpoint results/plantvillage/seed42/best.pt --manifest data/manifests/plantvillage.csv

python scripts/profile_model.py --checkpoint results/plantvillage/seed42/best.pt

python scripts/evaluate_localization.py --checkpoint results/plantvillage/seed42/best.pt --pairs data/plantseg_pairs.csv

python scripts/run_transfer.py --checkpoint results/plantvillage/seed42/best.pt --target-manifest data/manifests/plantdoc.csv --source-kind plantvillage --target-kind plantdoc --adapt
```

The model uses the source pretrained classifier for source-only transfer, restricts logits to semantically matched classes and optionally fine-tunes on 10% of the target training pool. Confirm canonical label mappings (in morphoagronet/evaluation/transfer.py) for the actual dataset release. Transfer adaptation is a research baseline and **does not** implement an independent validation-driven checkpoint selection.

To compare paired run scores, write two files containing arrays, e.g. model_f1.json and baseline_f1.json:

```bash
python scripts/run_statistics.py --model-runs model_f1.json --baseline-runs baseline_f1.json
```

For image-level bootstrap and McNemar analysis, also pass --predictions-a path/to/predictions.json --predictions-b path/to/predictions.json. Maintain identical test sample order across models.

## Caveats / faithful versus selected details

The model follows the manuscript's functional descriptions, but some equations were rendered as non-extractable Word objects. The repository explicitly selects values where the document does not establish a recoverable number. In particular: geometry encoder kernels 3/5/7, feature projection width 256, FGSS coefficient 0.1 and EARL coefficient 0.2 are **implementation defaults, not verified manuscript settings**. The manuscript fixes the weak lesion coefficient 0.20 and area penalty 0.10. The comparison CNNs and architecture ablations are implemented; complete manuscript Table 12 controlled-EARL baseline retraining can be reproduced by adding matched perturbation to baseline training and is not automated in the current runner. The code has smoke/unit tests but **was not trained on the full public datasets here**, and no reported manuscript accuracy is claimed to be reproduced. No pretrained or fitted model checkpoints are included.

The FAP boundary objective is exposed as geometry_preservation_loss; the current fixed OpenCV preprocessing cannot receive gradients, so that term is diagnostic rather than an optimizable component. The lesion pseudo-masks are generated online for simplicity, rather than once and cached. This uses the same deterministic unsupervised rule but incurs extra training overhead. The morphologically exact features use detached binary masks while the geometry CNN branch processes the differentiable soft map.

The pseudo-mask branch is intrinsically noisy on complex images and should be assessed using external masks; its masks are not ground-truth annotations. Inference profiling includes CPU contour processing and device synchronization costs. Robustness corruptions are deterministic but not a validated real-world sensor model. Do not interpret synthetic corruptions as proof of field readiness.

## Saved outputs

Each training run saves best.pt, history.json, metrics.json, and predictions.json. Data, pretrained weights, checkpoints and generated results are excluded from the source ZIP.

## Verification

```bash
python -m pytest -q

python -m compileall -q morphoagronet scripts
```

## Responsible use

This software is for controlled research benchmarking. Predicted diagnoses should not be treated as validated agronomic advice or field-deployment evidence without
