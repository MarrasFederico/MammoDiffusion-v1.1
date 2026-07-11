<div align="center">

<img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>

# MammoDiffusion v2

**Conditional generation of synthetic mammograms and evaluation of their usefulness for breast cancer classification.**

[Versione italiana](README.md)

</div>

## Goal

MammoDiffusion compares fine-tuned diffusion models and latent diffusion models trained from scratch on the RSNA Breast Cancer Detection dataset. Synthetic images are assessed not only through FID, Inception Score and PRDC, but also through their effect on classifiers evaluated exclusively on real validation and test data.

The original delivery used ResNet-50 as a sustainable and reproducible baseline. Version 2 extends the study to MaxViT-Tiny-512, MammoFM and RAD-DINO, introduces adapted VAEs, LoRA and a stronger from-scratch U-Net, and searches for the best absolute system even when it requires an expensive combination of real, augmented and synthetic data.

## Research Questions

1. Can from-scratch diffusers and fine-tuned Stable Diffusion 2.1 produce realistic and sufficiently diverse mammograms?
2. Do synthetic data improve AUC, F1 and positive-class recall over real-only training?
3. Does a native 512x512 classifier clarify the value of synthetic data compared with the 224x224 ResNet-50 baseline?
4. Does fine-tuning the Stable Diffusion VAE change FID, IS, PRDC and downstream performance?
5. Does placing the Stable Diffusion VAE inside the from-scratch diffuser improve over a VAE trained from zero?
6. Does the v3 U-Net with v-prediction and Min-SNR improve over previous from-scratch versions?
7. Can LoRA approach full fine-tuning quality at a lower computational cost?
8. Does Real+Augmented+Synthetic outperform either augmentation strategy alone?
9. Which synthetic source is most useful for each classifier under matched splits, seeds, thresholds and budgets?

## Structure

```text
MammoDiffusion/
|-- assets/
|-- data/                              # local, excluded from Git
|-- experiments/
|   |-- diffusers/                     # caches and checkpoints excluded from Git
|   `-- classifiers/
|       |-- resnet50/
|       |-- maxvit512/
|       |-- mammofm/
|       `-- raddino/
|-- notebooks/
|   |-- 1_preprocessing/
|   |-- 2_diffusers/
|   |-- 3_classifiers/
|   |-- 4_comparisons_and_test/
|   |-- pretrained_model/              # one local SD 2.1 base copy
|   `-- utility/
|-- results/
|   |-- preprocessing/
|   |-- diffusers/
|   |-- classifiers/
|   |-- comparisons/
`-- old/                               # local flat-layout archive, excluded from Git
```

The trailing letter identifies a data recipe for the same classifier. The `z` suffix is reserved for the final comparison within a family. Notebooks, experiments and results share the same prefix, for example `02a` for MaxViT RealOnly and `02z` for the MaxViT comparison.

Every active notebook contains a bootstrap that locates the project root and exposes `notebooks/utility/`. Notebooks can therefore be launched from the repository root or any subdirectory under `notebooks/`.

## Notebooks

### Preprocessing

| Notebook | Purpose |
|---|---|
| `1_preprocessing/01_Preprocessing_RSNA_512_gray_MLO.ipynb` | RSNA preprocessing, MLO selection, splits and 512x512 grayscale images |
| `1_preprocessing/02_Data_Augmentation_Trad.ipynb` | Traditional augmentation and metadata generation |

### Diffusers

| ID | Notebook | Main experiment |
|---|---|---|
| 01 | `2_diffusers/01_SD21_Baseline_50steps.ipynb` | SD2.1 baseline |
| 02 | `2_diffusers/02_SD21_Filtered_100steps.ipynb` | SD2.1 fine-tuning, sampling and filtering |
| 03 | `2_diffusers/03_SD21_VAE_FineTuned.ipynb` | SD2.1 VAE fine-tuning |
| 04 | `2_diffusers/04_SD21_LoRA.ipynb` | U-Net LoRA fine-tuning |
| 05 | `2_diffusers/05_LDM_Basic_FromScratch.ipynb` | Basic from-scratch LDM |
| 06 | `2_diffusers/06_LDM_Extra1361_FromScratch.ipynb` | Custom-VAE LDM with balanced output |
| 07 | `2_diffusers/07_LDM_SDVAE_Extra1361.ipynb` | U-Net retrained on SD-VAE latents |
| 08 | `2_diffusers/08_LDM_v3_SDVAE_FromScratch.ipynb` | v3 U-Net, v-prediction, Min-SNR and SD-VAE |

Notebooks 07 and 08 apply the same workflow to both classes:

```text
generate -> filter -> validate (real validation) -> test (real test)
```

Final metrics are stored separately under `metrics/positive/` and `metrics/negative/`. Main positive outputs are also mirrored to flat paths for backward compatibility.

### Classifiers

| Family | Available notebooks |
|---|---|
| ResNet-50 (`01`) | `01a` RealOnly, `01b` RealSynth partial, `01c` RealSynth full, `01d` SyntheticOnly, `01e` RealAugmented, `01f` RealSynthPositive |
| MaxViT-512 (`02`) | `02a`-`02f` ResNet-equivalent recipes, `02i` RealAugSynth FromScratch, `02j` RealAugSynth FineTuned |
| MammoFM (`03`) | `03a` RealOnly, `03b` RealSynth FineTuned, `03c` RealSynth FromScratch, `03d` RealAugmented |
| RAD-DINO (`04`) | `04a` RealOnly, `04b` RealSynth |

Validation, test and comparison notebooks live in `4_comparisons_and_test/`. `00z` compares diffusers; `01z` and `02z` close the ResNet and MaxViT families; `03z` compares ResNet-50 with MaxViT-512.

## Synthetic Data

| Directory | Source |
|---|---|
| `data/synthetic/fine_tuned/` | Fine-tuned SD2.1 |
| `data/synthetic/fine_tuned_vaeft/` | SD2.1 with adapted VAE |
| `data/synthetic/fine_tuned_lora/` | SD2.1 LoRA |
| `data/synthetic/fromscratch/` | LDM with custom VAE |
| `data/synthetic/fromscratch_new/` | LDM with SD-VAE |
| `data/synthetic/fromscratch_v3/` | v3 LDM with SD-VAE |

Each final dataset contains `positive/` and `negative/`. `data/` is excluded from Git and must be prepared or restored locally.

## Comparison Strategy

To avoid selecting a synthetic source on the test set, the project follows two stages:

1. evaluate every candidate synthetic source with fixed recipes and budgets, using only the real validation set for ranking and threshold selection;
2. use the best from-scratch source and best fine-tuned source in the expensive `Real+Synth` and `Real+Augmented+Synthetic` recipes, followed by one final evaluation on the real test set.

The matrix will be applied to ResNet-50, MaxViT-512, MammoFM and RAD-DINO. Where a complete matrix is computationally prohibitive, source selection will remain shared and documented instead of choosing a different dataset after seeing each model's test result.

## Installation

```bash
git clone https://github.com/MarrasFederico/MammoDiffusion-v2.git
cd MammoDiffusion-v2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch/TensorFlow and CUDA must match the selected GPU. Notebook 08 explicitly configures RTX 5060 Ti/Blackwell and the `libdevice` path; notebook 04 selects its GPU before importing PyTorch. Restart the kernel after changing GPU assignments.

The shared base model is resolved from `notebooks/pretrained_model/stable-diffusion-2-1-base` or `MAMMODIFFUSION_SD21_BASE`. Modified VAEs and adapters remain inside their experiment directories.

## Gradio Demo

The local demo lives in `assets/mammodiffusion_gradio/`. It uses the current paths for experiments 02 and 06 and keeps temporary outputs out of Git.

## Reproducibility

- unchanged real-data splits across experiments;
- Youden thresholds computed on the validation set;
- real test set isolated from checkpoint, filter and synthetic-source selection;
- seeds and budgets recorded in notebooks and manifests;
- class-wise FID, IS and PRDC;
- heavy datasets, checkpoints and caches excluded from Git, with code and metrics versioned.

## Evolution

The immediate priority is completing notebook 04 LoRA and rerunning notebook 08 under the final structure. Their results will determine which synthetic datasets enter the complete downstream comparison. The project does not prefer the cheapest method by default: in mammography, a higher cost is acceptable when it produces a robust and reproducible gain, especially in positive-class recall.

## Team

| Name | GitHub |
|---|---|
| Enzo Fumagalli | [@EnzoFumagalli](https://github.com/EnzoFumagalli) |
| Federico Marras | [@MarrasFederico](https://github.com/MarrasFederico) |
| Alexandro Sanna | [@AlexandroSanna](https://github.com/AlexandroSanna) |
| Samuele Nonnis | [@SamueleNonnis](https://github.com/SamueleNonnis) |

Developed for the 2026 Deep Learning course in the Applied Computer Science and Data Analytics degree program at the University of Cagliari.
