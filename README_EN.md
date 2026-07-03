<div align="center">

<img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>

# MammoDiffusion

**Conditional generation of synthetic mammography images with diffusion models**
to support breast cancer classification.

[Versione italiana](README.md)

<!--
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RSNA%20Breast%20Cancer-pink?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
-->
</div>

---

## Table Of Contents

- [Project Description](#project-description)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [External Materials](#external-materials)
- [Installation](#installation)
- [Notebook Execution Order](#notebook-execution-order)
- [Notebook Descriptions](#notebook-descriptions)
- [Project Evolution](#project-evolution)
- [Team](#team)

---

## Project Description

**MammoDiffusion** focuses on building and training diffusion-based models for conditional generation of synthetic mammography images, with the goal of understanding whether synthetic data can improve the identification of positive cancer cases.

The project follows an incremental experimental workflow:

1. **Preprocessing and augmentation** of the original RSNA Breast Cancer Detection dataset
2. **Synthetic image generation** through two approaches: fine-tuning *Stable Diffusion 2.1* and training an LDM (Latent Diffusion Model) *from scratch*
3. **Training and evaluation** of ResNet-50 classifiers in multiple configurations, used as a computationally sustainable baseline and also assessed in terms of sustainability
4. **Post-delivery extension** with MaxViT-Tiny-512 classifiers, architecture comparisons, and further research questions on VAEs, from-scratch diffusion, and combined real/augmented/synthetic training data

---

## Research Questions

### RQ1 - Generation Quality

> *Can both a from-scratch diffusion architecture and a fine-tuned one generate realistic and sufficiently diverse mammography samples?*

- **Approaches:** fine-tuning of *Stable Diffusion 2.1*; Keras LDM trained from scratch
- **Evaluation metrics:** FID (Frechet Inception Distance), Inception Score (IS), Precision, Recall, Density, Coverage

### RQ2 - Impact On Classification

> *Does adding synthetic samples improve classifier performance compared with using only real data?*

- **Configurations:** Baseline (real data only), Real+Synth (real + synthetic), Full Synth (synthetic only)
- **Evaluation metrics:** AUC, F1, Accuracy, Precision, Recall on the real test set
- **Architecture:** ImageNet-pretrained ResNet-50, trained in two phases (head training + partial backbone fine-tuning)

### RQ2b - Does A True 512x512 Classifier Beat ResNet-50?

> *Does a hybrid transformer backbone with native 512x512 input (MaxViT-Tiny, `timm`) outperform ResNet-50 (224x224) under the same training configurations?*

- **Architecture:** MaxViT-Tiny-512 (`timm`, `in1k` pretrained), trained with `BCEWithLogitsLoss`/`BinaryFocalLoss` in PyTorch using the same Baseline/Real+Synth/Full Synth configurations as RQ2
- **Direct comparison:** `21_Confronto_ResNet50_vs_MaxViT512.ipynb`, evaluated on the same real test set

### RQ3 - Sustainability And Responsible AI

> *Is diffusion-based data augmentation actually worthwhile in terms of performance gain and computational sustainability, compared with traditional data augmentation?*

- **Comparison:** Real+Augmented classifier (traditional augmentation) vs Real+Synth classifier (diffusion-based augmentation)
- **Metrics:** AUC, F1, CO2 emissions and energy consumption tracked with `codecarbon`

### RQ4 - Can Combining All Data Sources Improve The Best Absolute Model?

> *Instead of choosing strictly between traditional augmentation and synthetic data, does a classifier trained on real + augmented + synthetic data achieve better performance on the real test set?*

- **Motivation:** mammography is a complex and high-impact domain; higher computational costs can be justified if the clinical performance gain is measurable
- **Future configuration:** Real+Augmented+Synthetic, compared against Baseline, Real+Augmented, Real+Synth and Full Synth

### RQ5 - Fine-Tuning The Stable Diffusion VAE

> *Does adapting the Stable Diffusion VAE to the mammography domain change generative quality and downstream metrics?*

- **Future experiment:** controlled fine-tuning or adaptation of the VAE on preprocessed mammography images
- **Metrics:** FID, IS, Precision/Recall/Density/Coverage, plus downstream impact on classifiers trained with generated data

### RQ6 - Pretrained VAE Inside The From-Scratch Diffuser

> *Does replacing or initializing the from-scratch diffuser VAE with a pretrained VAE improve stability, sample quality and classification usefulness?*

- **Future experiment:** a new notebook for a from-scratch LDM using a pretrained or domain-adapted Stable Diffusion VAE
- **Comparison:** current from-scratch LDM vs LDM with pretrained VAE vs fine-tuned Stable Diffusion

### RQ7 - From-Scratch Synthetic Data As Classification Signal

> *Do the images in `data/synthetic/fromscratch/` contain enough discriminative information to improve or partly replace fine-tuned synthetic data?*

- **Future experiment:** Real+FromScratchSynthetic and FromScratchSynthetic-only classifiers, evaluated on the same real test set
- **Goal:** distinguish visual realism, domain coverage and actual usefulness for classification

---

## Repository Structure

```text
MammoDiffusion/
|
|-- assets/                                         # Supporting assets
|   |-- logo_MammoDiffusion.png
|
|-- experiments/                                    # Model weights and experiment logs
|   |-- exp20260618_baseline_resnet50_fine_tuned_batch_size_16/
|   |-- exp20260617_real_synth_resnet50_fine_tuned_batch_size_16/
|   |-- exp20260619_full_synth_resnet50_fine_tuned_batch_size_16/
|   |-- exp_maxvit512_baseline/
|   |-- exp_maxvit512_real_synth_partial/
|   |-- ...
|
|-- notebooks/                                      # Workflow notebooks
|   |-- 01_Preprocessing_RSNA_512_gray_MLO.ipynb
|   |-- 02_Data_Augmentation_Trad.ipynb
|   |-- 03a_Finetuning_StableDiffusion2.1_baseline.ipynb
|   |-- 03b_Finetuning_StableDiffusion2.1_filtered.ipynb
|   |-- 04a_LDM_basic.ipynb
|   |-- 04b_LDM_extra1361.ipynb
|   |-- 04c_Confronto_FromScratch_vs_FineTuned.ipynb
|   |-- 05_Classificatore_Baseline_ResNet-50_FineTuned.ipynb
|   |-- 06_Classificatore_RealSynthetic_ResNet-50_FineTuned.ipynb
|   |-- 06b_Classificatore_RealSynthetic_ResNet-50_FineTuned_Full.ipynb
|   |-- 07_Val_Classificatori_RS_AllVSPart.ipynb
|   |-- 08_Classificatore_Synthetic_ResNet-50_FineTuned.ipynb
|   |-- 09_Test_Classificatori.ipynb
|   |-- 10_Classificatore_Real_Augmented_ResNet-50_FineTuned.ipynb
|   |-- 11_Classificatore_RealSyntheticPositive_ResNet-50_FineTuned.ipynb
|   |-- 12_Valutazione_Sostenibilita.ipynb
|   |-- 13_Classificatore_Baseline_MaxViT512_FineTuned.ipynb
|   |-- 14_Classificatore_RealSynthetic_MaxViT512_FineTuned.ipynb
|   |-- 14b_Classificatore_RealSynthetic_MaxViT512_FineTuned_Full.ipynb
|   |-- 15_Val_Classificatori_RS_AllVSPart_MaxViT512.ipynb
|   |-- 16_Classificatore_Synthetic_MaxViT512_FineTuned.ipynb
|   |-- 17_Test_Classificatori_MaxViT512.ipynb
|   |-- 18_Classificatore_Real_Augmented_MaxViT512_FineTuned.ipynb
|   |-- 19_Classificatore_RealSyntheticPositive_MaxViT512_FineTuned.ipynb
|   |-- 20_Valutazione_Sostenibilita_MaxViT512.ipynb
|   |-- 21_Confronto_ResNet50_vs_MaxViT512.ipynb
|   |-- eco_tracker.py
|   |-- maxvit_utils.py
|   |-- generative_evaluator.py
|
|-- results/                                        # Metrics, plots and reports
|   |-- 01_preprocessing/
|   |-- 02_data_augmentation/
|   |-- 03b_finetuning_filtered/
|   |-- 04_ldm_keras_v2/
|   |-- 04b_ldm_keras_v2_extra1361/
|   |-- 07_val_classificatori_allVSpart/
|   |-- 09_test_classificatori/
|   |-- test_trad_aug_vs_real_synth/
|   |-- ...
|
|-- README.md
|-- README_EN.md
|-- requirements.txt
|-- .gitignore
```

> **Note:** the `data/` directory is excluded from the repository because of its size. Original, preprocessed and synthetic datasets are stored in the team's shared Google Drive.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | RSNA Breast Cancer Detection |
| **Original source** | [Kaggle - RSNA Breast Cancer 512 PNGs](https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs) |
| **Format** | PNG, 512x512 px, grayscale |
| **View** | MLO (Mediolateral Oblique), the only view used in this project |
| **Selection** | 1 image per patient; positive patients use the cancerous breast, negative patients are selected randomly |
| **Label** | `cancer` (0 = healthy, 1 = malignant) |
| **Split** | train / validation / test, stratified by label |

### Data Locations

| Folder | Description | Location |
|---|---|---|
| `data/original/dataset/` | Original RSNA 512x512 PNG images | Team Google Drive |
| `data/processed/{train, val, test, metadata}` | Preprocessed images and split CSV files | Team Google Drive |
| `data/real_augmented/` | Positive images created with traditional augmentation | Produced by notebook `02` |
| `data/synthetic/fine_tuned/{negative, positive}` | Filtered synthetic images generated by fine-tuned SD2.1 | Produced by notebook `03b` |
| `data/synthetic/fromscratch/{negative, positive}` | Filtered synthetic images generated by the from-scratch diffuser | Produced by notebook `04b` |

---

## External Materials

The following files are not included in the repository because of size or distribution constraints:

| Resource | Reason for exclusion | How to obtain it |
|---|---|---|
| Original RSNA dataset | Too large (> 20 GB) | Download from Kaggle or the team Google Drive |
| Preprocessed dataset (`data/processed/`) | Too large | Team Google Drive, downloaded automatically by notebooks |
| Augmented data (`data/real_augmented/`) | Too large | Produced by notebook `02` or obtained from the team drive |
| Synthetic images (`data/synthetic/`) | Too large | Team Google Drive, downloaded automatically by classifier notebooks |
| Model weights (`.keras`, `.pt`) | Too large, ignored by git | Team Google Drive or local experiment outputs |
| Stable Diffusion 2.1 base model | About 5 GB | Team Google Drive or the configured download path |

---

## Installation

### 1. Clone The Repository

```bash
git clone https://github.com/EnzoFumagalli/MammoDiffusion.git
cd MammoDiffusion
```

### 2. Create A Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

| Group | Libraries |
|---|---|
| Data science | `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `pillow` |
| Visualization | `matplotlib`, `seaborn` |
| ResNet-50 classifiers | `tensorflow >= 2.15` |
| MaxViT-512 classifiers | `torch >= 2.0`, `timm >= 1.0` |
| Diffusion models | `torch >= 2.0`, `torchvision`, `torchmetrics` |
| Stable Diffusion | `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub` |
| Generative metrics | `torch-fidelity`, `prdc` |
| Sustainability | `codecarbon` |
| Utilities | `tqdm`, `gdown`, `tensorboard` |

### 4. Download The Dataset

The notebooks automatically handle Google Drive downloads when possible. Alternatively, manually download the `data/` folders from the team's shared drive and place them in the project root.

---

## Notebook Execution Order

The workflow is organized into operational pipelines. Notebooks with related prefixes (for example `03a`/`03b` or `06`/`06b`) are variants of the same experimental step.

```text
PIPELINE 1 - Data preparation
    01_Preprocessing_RSNA_512_gray_MLO
        -> 02_Data_Augmentation_Trad

PIPELINE 2 - Stable Diffusion 2.1 generation (RQ1, input for RQ2/RQ3)
    03a_Finetuning_StableDiffusion2.1_baseline
    03b_Finetuning_StableDiffusion2.1_filtered

PIPELINE 3 - From-scratch LDM generation (RQ1)
    04a_LDM_basic
    04b_LDM_extra1361
        -> 04c_Confronto_FromScratch_vs_FineTuned

PIPELINE 4 - ResNet-50 classifiers (RQ2)
    05_Classificatore_Baseline_ResNet-50_FineTuned
    06_Classificatore_RealSynthetic_ResNet-50_FineTuned
    06b_Classificatore_RealSynthetic_ResNet-50_FineTuned_Full
        -> 07_Val_Classificatori_RS_AllVSPart
    08_Classificatore_Synthetic_ResNet-50_FineTuned
        -> 09_Test_Classificatori

PIPELINE 5 - ResNet-50 sustainability comparison (RQ3)
    10_Classificatore_Real_Augmented_ResNet-50_FineTuned
    11_Classificatore_RealSyntheticPositive_ResNet-50_FineTuned
        -> 12_Valutazione_Sostenibilita

PIPELINE 4b - MaxViT-Tiny-512 classifiers (RQ2b)
    13_Classificatore_Baseline_MaxViT512_FineTuned
    14_Classificatore_RealSynthetic_MaxViT512_FineTuned
    14b_Classificatore_RealSynthetic_MaxViT512_FineTuned_Full
        -> 15_Val_Classificatori_RS_AllVSPart_MaxViT512
    16_Classificatore_Synthetic_MaxViT512_FineTuned
        -> 17_Test_Classificatori_MaxViT512
            -> 21_Confronto_ResNet50_vs_MaxViT512

PIPELINE 5b - MaxViT-Tiny-512 sustainability comparison (RQ3)
    18_Classificatore_Real_Augmented_MaxViT512_FineTuned
    19_Classificatore_RealSyntheticPositive_MaxViT512_FineTuned
        -> 20_Valutazione_Sostenibilita_MaxViT512
```

---

## Notebook Descriptions

### `01_Preprocessing_RSNA_512_gray_MLO.ipynb`

Prepares the raw RSNA dataset by filtering MLO views, selecting one image per patient, normalizing/padding images to 512x512 grayscale, orienting tissue consistently to the left, and creating stratified train/validation/test splits.

**Output:** `data/processed/` and preprocessing metrics in `results/01_preprocessing/`.

### `02_Data_Augmentation_Trad.ipynb`

Creates traditional positive-class augmentation using mild contrast, brightness and Gaussian noise changes. No flips are used because the preprocessing step already normalizes tissue orientation.

**Output:** `data/real_augmented/metadata.csv` and augmentation reports in `results/02_data_augmentation/`.

### `03a_Finetuning_StableDiffusion2.1_baseline.ipynb`

Fine-tunes Stable Diffusion 2.1 on MLO mammograms and generates raw samples at 50 inference steps, without quality filtering.

**Output:** baseline generated images and checkpoint-level validation metrics.

### `03b_Finetuning_StableDiffusion2.1_filtered.ipynb`

Main Stable Diffusion experiment. It reuses or fine-tunes SD2.1 checkpoints, evaluates them at 100 inference steps, selects the best checkpoint, generates raw samples, applies the adaptive filter, and evaluates the final filtered synthetic dataset.

**Output:** `data/synthetic/fine_tuned/{positive, negative}` and metrics/plots in `results/03b_finetuning_filtered/`.

### `04a_LDM_basic.ipynb`

Trains an initial Keras Latent Diffusion Model from scratch using real and augmented training data.

**Output:** LDM checkpoints and generative metrics in `results/04_ldm_keras_v2/`.

### `04b_LDM_extra1361.ipynb`

Extends the from-scratch LDM pipeline with additional raw generation, adaptive filtering, negative-class generation, more robust EcoTracker logging and checkpoint sweep evaluation.

**Output:** filtered from-scratch synthetic images and metrics in `results/04b_ldm_keras_v2_extra1361/`.

### `04c_Confronto_FromScratch_vs_FineTuned.ipynb`

Compares the from-scratch LDM (`04b`) against fine-tuned Stable Diffusion (`03b`) using already generated artifacts.

**Output:** comparative plots and tables in `results/04c_Confronto_FromScratch_vs_FineTuned/`.

### `05_Classificatore_Baseline_ResNet-50_FineTuned.ipynb`

Trains the baseline binary classifier on real preprocessed data only. The architecture is ImageNet-pretrained ResNet-50 with a custom classification head and two training phases.

**Output:** best model, training logs and validation metrics in `experiments/exp20260618_baseline_resnet50_fine_tuned_batch_size_16/`.

### `06_Classificatore_RealSynthetic_ResNet-50_FineTuned.ipynb`

Trains the ResNet-50 classifier on real + synthetic data with partial backbone fine-tuning.

**Output:** `experiments/exp20260617_real_synth_resnet50_fine_tuned_batch_size_16/`.

### `06b_Classificatore_RealSynthetic_ResNet-50_FineTuned_Full.ipynb`

Variant of `06` with full backbone fine-tuning, introduced to check whether previous performance limitations came from insufficient backbone unfreezing.

**Output:** `experiments/exp20260618_real_synth_resnet50_fine_tuned_all_layers/`.

### `07_Val_Classificatori_RS_AllVSPart.ipynb`

Compares partial vs full fine-tuning for the Real+Synth ResNet-50 setting on the validation set.

**Output:** validation comparison metrics and predictions in `results/07_val_classificatori_allVSpart/`.

### `08_Classificatore_Synthetic_ResNet-50_FineTuned.ipynb`

Trains a ResNet-50 classifier only on synthetic images, then validates it on real data. This tests whether the generated images contain usable discriminative signal.

**Output:** `experiments/exp20260619_full_synth_resnet50_fine_tuned_batch_size_16/`.

### `09_Test_Classificatori.ipynb`

Final ResNet-50 evaluation of the Baseline, Real+Synth and Full Synth configurations on the real test set.

**Output:** metrics, predictions and figures in `results/09_test_classificatori/`.

### `10_Classificatore_Real_Augmented_ResNet-50_FineTuned.ipynb`

Trains a ResNet-50 classifier on real data plus traditional augmentation. This is used as the traditional augmentation reference for sustainability analysis.

**Output:** `experiments/exp_trad_aug_resnet50/`.

### `11_Classificatore_RealSyntheticPositive_ResNet-50_FineTuned.ipynb`

Trains a ResNet-50 classifier on real data plus positive synthetic samples, used for the sustainability comparison against traditional augmentation.

**Output:** `experiments/exp_synth_pos_resnet50/`.

### `12_Valutazione_Sostenibilita.ipynb`

Compares Real+Augmented and Real+SyntheticPositive in terms of test performance and environmental cost, using EcoTracker/codecarbon logs.

**Output:** comparative metrics, predictions and sustainability plots.

### MaxViT-Tiny-512 Classifiers (`13`-`21`)

Notebooks `13`-`21` reproduce the ResNet-50 classifier experiments with **MaxViT-Tiny-512** (`timm`, PyTorch, native 512x512 input). They use the same datasets and comparable training logic: two-phase training, validation-derived Youden thresholds and real test-set evaluation. Shared MaxViT dataset/training/Grad-CAM logic is implemented in `notebooks/maxvit_utils.py`.

### `13_Classificatore_Baseline_MaxViT512_FineTuned.ipynb`

MaxViT equivalent of `05`, trained on real data only.

**Output:** `experiments/exp_maxvit512_baseline/`.

### `14_Classificatore_RealSynthetic_MaxViT512_FineTuned.ipynb` / `14b_..._Full.ipynb`

MaxViT equivalents of `06` and `06b`, trained on real + synthetic data with partial or full backbone fine-tuning.

**Output:** `experiments/exp_maxvit512_real_synth_partial/` and `experiments/exp_maxvit512_real_synth_full/`.

### `15_Val_Classificatori_RS_AllVSPart_MaxViT512.ipynb`

Validation comparison between partial and full MaxViT fine-tuning for the Real+Synth setting.

**Output:** `results/15_val_classificatori_maxvit512_allVSpart/`.

### `16_Classificatore_Synthetic_MaxViT512_FineTuned.ipynb`

MaxViT equivalent of `08`, trained only on synthetic images.

**Output:** `experiments/exp_maxvit512_full_synth/`.

### `17_Test_Classificatori_MaxViT512.ipynb`

Final MaxViT evaluation of Baseline, Real+Synth and Full Synth on the real test set.

**Output:** `results/17_test_classificatori_maxvit512/`.

### `18_Classificatore_Real_Augmented_MaxViT512_FineTuned.ipynb` / `19_Classificatore_RealSyntheticPositive_MaxViT512_FineTuned.ipynb`

MaxViT equivalents of `10` and `11`, used for the MaxViT sustainability comparison.

**Output:** `experiments/exp_maxvit512_real_augmented/` and `experiments/exp_maxvit512_synth_pos/`.

### `20_Valutazione_Sostenibilita_MaxViT512.ipynb`

MaxViT equivalent of `12`, comparing Real+Augmented vs Real+SyntheticPositive on the real test set while reusing the data-generation sustainability logs.

**Output:** `results/test_maxvit512_trad_aug_vs_synth_pos/`.

### `21_Confronto_ResNet50_vs_MaxViT512.ipynb`

Direct architecture comparison. It loads metrics from `09` (ResNet-50) and `17` (MaxViT-512), computes metric deltas and summarizes whether MaxViT improves over ResNet under each configuration.

**Output:** `results/21_confronto_resnet50_vs_maxvit512/`.

---

## Project Evolution

The original project used ResNet-50 as the main classifier because it provided a practical compromise between computational cost, training simplicity and comparability across configurations. This was appropriate for the initial delivery and helped isolate the contribution of synthetic data. However, it remains a baseline: ResNet-50 operates at 224x224, while the preprocessed mammograms are available at 512x512.

The next development step therefore asks a clearer question: does the benefit of synthetic data change when the downstream classifier can directly exploit 512x512 resolution? Notebooks `13`-`21` replicate the ResNet pipeline with MaxViT-Tiny-512 while keeping splits, seeds, Youden thresholds and configurations as comparable as possible.

The long-term goal is not only to determine which single technique is "better" in isolation. Mammography is difficult, imbalanced and high-impact: if a more expensive combination produces robust gains in AUC, recall and F1 for the positive class, the additional cost may be justified. Future work will therefore include:

- a **Real+Augmented+Synthetic** classifier, aiming for the best absolute model rather than a strict choice between traditional and diffusion-based augmentation;
- **fine-tuning the Stable Diffusion VAE**, to test whether a mammography-adapted latent representation improves FID/PRDC and downstream classifier performance;
- a new **from-scratch LDM with a pretrained VAE**, to separate the contribution of the VAE from that of the diffusion U-Net;
- classifiers trained with `data/synthetic/fromscratch/`, to measure whether from-scratch samples are useful not only visually but also as discriminative training signal.

This extension keeps the same guiding principle: every generative improvement must be evaluated both through image-quality metrics and through its real effect on classifiers, using real validation/test sets and documented comparisons.

---

## Team

| Name | GitHub |
|---|---|
| Enzo Fumagalli | [@EnzoFumagalli](https://github.com/EnzoFumagalli) |
| Federico Marras | [@MarrasFederico](https://github.com/MarrasFederico) |
| Alexandro Sanna | [@AlexandroSanna](https://github.com/AlexandroSanna) |
| Samuele Nonnis | [@SamueleNonnis](https://github.com/SamueleNonnis) |

---

## Acknowledgements

This project was developed by Enzo Fumagalli, Federico Marras, Alexandro Sanna and Samuele Nonnis for the Deep Learning course, academic year 2026, within the Applied Computer Science and Data Analytics (IADA) degree program at the University of Cagliari.
