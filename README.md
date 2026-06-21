<div align="center">

<img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>

# MammoDiffusion

**Generazione condizionata di immagini mammografiche sintetiche tramite modelli diffusivi**
per il miglioramento della classificazione del cancro al seno.
<!--
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RSNA%20Breast%20Cancer-pink?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
-->
</div>

---

## Indice

- [Descrizione del progetto](#descrizione-del-progetto)
- [Domande di ricerca](#domande-di-ricerca)
- [Struttura della repository](#struttura-della-repository)
- [Dataset](#dataset)
- [Materiali esterni](#materiali-esterni)
- [Installazione](#installazione)
- [Ordine di esecuzione dei notebook](#ordine-di-esecuzione-dei-notebook)
- [Descrizione dei notebook](#descrizione-dei-notebook)
- [Team](#team)

---

## Descrizione del progetto

**MammoDiffusion** verte sulla costruzione e sull’addestramento di un modello diffusivo ai fini della generazione condizionata di immagini sintetiche, con il fine di comprendere se possano realmente migliorare l’identificazione  di casi positivi.

Il progetto segue un flusso in tre fasi:

1. **Preprocessing e augmentation** dei dati originali del dataset RSNA Breast Cancer Detection
2. **Generazione di immagini sintetiche** tramite due approcci distinti: fine-tuning di *Stable Diffusion 2.1* e addestramento di un modello LDM (Latent Diffusion Model) *from scratch*
3. **Training e valutazione** di classificatori ResNet-50 in tre configurazioni differenti, confrontate anche in termini di sostenibilità computazionale

---

## Domande di ricerca

### D1 — Qualità della generazione

> *Un'architettura diffusiva costruita from Scratch e anche una fine tunata sono in grado di generare campioni realistici e sufficientemente vari?*

- **Approcci:** fine-tuning di *Stable Diffusion 2.1*; LDM Keras addestrato from scratch
- **Metriche di valutazione:** FID (Fréchet Inception Distance), Inception Score (IS), Precision, Recall, Density, Coverage

### D2 — Impatto sulla classificazione

> *L’aggiunta di campioni sintetici migliora le prestazioni dei classificatori rispetto all’utilizzo dei soli dati reali?*

- **Configurazioni:** Baseline (soli dati reali), Real+Synth (reali + sintetici), Full Synth (soli sintetici)
- **Metriche di valutazione:** AUC, F1, Accuracy, Precision, Recall sul test set reale
- **Architettura:** ResNet-50 con ImageNet, training in due fasi (head training + fine-tuning parziale del backbone)

### D3 — Sostenibilità e Responsible AI

> *La data augmentation realizzata tramite generazione è realmente conveniente, sia in termini di guadagno prestazionale che in termini di sostenibilità, rispetto alla data augmentation tradizionale?*

- **Confronto:** classificatore Real+Augmented (augmentation tradizionale) vs classificatore Real+Synth (augmentation diffusiva)
- **Metriche:** AUC, F1 + emissioni CO₂ e consumi energetici tracciati con `codecarbon`

---

## Struttura della repository

```
MammoDiffusion/
│
├── assets/                                         # Materiali di supporto
│   ├── logo_MammoDiffusion.png
│
├── data/                                           # Dataset (NON in git, su Google Drive)
│   ├── original/
│   │   └── dataset/                               # Immagini originali RSNA 512×512 PNG
│   ├── processed/                                  # Dataset preprocessato (MLO, 1 img/paziente)
│   │   └── metadata/                              # CSV di split (train/val/test)
│   ├── real_augmented/                             # Immagini positive con augmentation tradizionale
│   └── synthetic/
│       └── fine_tuned/                            # Immagini sintetiche generate da SD2.1
│           ├── positive/                           # 1361 immagini positive filtrate
│           └── negative/                           # 1361 immagini negative filtrate
│
├── experiments/                                    # Pesi dei modelli (NON in git, .keras in .gitignore)
│   ├── exp20260618_baseline_resnet50_fine_tuned_batch_size_16/
│   ├── exp20260617_real_synth_resnet50_fine_tuned_batch_size_16/
│   ├── exp20260619_full_synth_resnet50_fine_tuned_batch_size_16/
│   ├── exp_trad_aug_resnet50/
│   └── ...                                        # Esperimenti intermedi
│
├── notebooks/                                      # Notebook del flusso di lavoro
│   ├── 01_Preprocessing_RSNA_512_gray_MLO.ipynb
│   ├── 02_Data_Augmentation_Trad.ipynb
│   ├── 03a_Finetuning_StableDiffusion2.1_baseline.ipynb
│   ├── 03b_Finetuning_StableDiffusion2.1_filtered.ipynb
│   ├── 04a_LDM_basic.ipynb
│   ├── 04b_LDM_extra1361.ipynb
│   ├── 05_Classificatore_Baseline_ResNet-50_FineTuned.ipynb
│   ├── 06_Classificatore_RealSynthetic_ResNet-50_FineTuned.ipynb
│   ├── 06b_Classificatore_RealSynthetic_ResNet-50_FineTuned_Full.ipynb
│   ├── 07_Val_Classificatori_RS_AllVSPart.ipynb
│   ├── 08_Classificatore_Synthetic_ResNet-50_FineTuned.ipynb
│   ├── 09_Test_Classificatori.ipynb
│   ├── 10_Classificatore_Real_Augmented_ResNet-50_FineTuned.ipynb
|   ├── 11_Classificatore_RealSyntheticPositive_ResNet-50_FineTuned.ipynb
│   ├── 12_Valutazione_Sostenibilità.ipynb
│   ├── eco_tracker.py                              # Wrapper codecarbon per il tracciamento CO₂
│   └── generative_evaluator.py                    # Calcolo metriche FID, IS, Precision, Recall
│
├── results/                                        # Output e metriche (in git)
│   ├── 01_preprocessing/
│   ├── 02_data_augmentation/
│   ├── 03b_finetuning_filtered/
│   ├── 05_confronto_metriche_val_classificatori/
│   ├── 07_test_classificatori_allVSpart/
│   ├── 07_val_classificatori_allVSpart/
│   ├── 09_test_classificatori/
│   └── test_trad_aug_vs_real_synth/
│
├── README.md
├── requirements.txt
└── .gitignore
```

> **Nota:** La cartella `data/` è esclusa dalla repository (dimensioni > GB). Tutti i dataset (originale, preprocessato, sintetico) sono disponibili su **Google Drive** condiviso del team. I file `.keras` con i pesi dei modelli sono anch'essi esclusi da git.

---

## Dataset

| Proprietà | Dettaglio |
|---|---|
| **Nome** | RSNA Breast Cancer Detection |
| **Fonte originale** | [Kaggle — RSNA Breast Cancer 512 PNGs](https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs) |
| **Formato** | PNG, 512×512 px, grayscale |
| **Vista utilizzata** | MLO (Mediolateral Oblique) — unica vista considerata nel progetto |
| **Selezione** | 1 immagine per paziente (positivi: solo immagini del seno malato; negativi: selezione random) |
| **Label** | `cancer` (0 = sano, 1 = maligno) |
| **Split** | train / val / test (stratificato per label) |

### Dove si trovano i dati

| Cartella | Descrizione | Posizione |
|---|---|---|
| `data/original/dataset/` | Immagini originali RSNA (512×512 PNG) | Google Drive del team |
| `data/processed/` | Immagini preprocessate + CSV split | Google Drive del team |
| `data/real_augmented/` | Immagini positive con augmentation tradizionale | Prodotto dal notebook `02` |
| `data/synthetic/fine_tuned/` | Immagini sintetiche generate da SD2.1 (filtrate) | Prodotto dal notebook `03b` |
| `data/synthetic/fromscratch/` | Immagini sintetiche generate dal diffusore from scratch (filtrate) | Prodotto dal notebook `04b` |


---

## Materiali esterni

I seguenti file non sono inclusi nella repository per dimensioni o per policy di distribuzione:

| Risorsa | Motivo esclusione | Come ottenerla |
|---|---|---|
| Dataset RSNA originale | Troppo grande (> 20 GB) | Scaricare da Kaggle o da Google Drive del team |
| Dataset preprocessato (`data/processed/`) | Troppo grande | Google Drive del team — scaricato automaticamente dai notebook |
| Immagini sintetiche (`data/synthetic/`) | Troppo grande | Google Drive del team — scaricate automaticamente dai notebook classificatori |
| Pesi dei modelli (`.keras`) | Troppo grandi, in `.gitignore` | Google Drive del team — scaricati automaticamente dai notebook di valutazione |
| Modello base SD2.1 | ~5 GB, licenza HuggingFace | `stabilityai/stable-diffusion-2-1` su HuggingFace Hub |

---

## Installazione

### 1. Clona la repository

```bash
git clone https://github.com/team/MammoDiffusion.git
cd MammoDiffusion
```

### 2. Crea un ambiente virtuale

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

Le dipendenze principali includono:

| Gruppo | Librerie |
|---|---|
| Data science | `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `pillow` |
| Visualizzazione | `matplotlib`, `seaborn` |
| Deep learning (classificatori) | `tensorflow >= 2.10` |
| Deep learning (diffusione) | `torch >= 2.0`, `torchvision`, `torchmetrics` |
| Stable Diffusion | `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub` |
| Metriche generative | `torch-fidelity`, `prdc` |
| Sostenibilità | `codecarbon` |
| Utility | `tqdm`, `gdown`, `tensorboard` |

### 4. Scarica il dataset

I notebook gestiscono automaticamente il download da Google Drive. In alternativa, scaricare manualmente le cartelle `data/` dalla cartella condivisa del team e posizionarle nella root del progetto.

---

## Ordine di esecuzione dei notebook

Il flusso è organizzato in tre pipeline distinte. I notebook numerati con lo stesso prefisso (es. `03a` e `03b`, oppure `06` e `06b`) rappresentano varianti alternative dello stesso step, non passi sequenziali.

```
PIPELINE 1 — Preparazione dei dati
    01_Preprocessing_RSNA_512_gray_MLO
        └── 02_Data_Augmentation_Trad

PIPELINE 2 — Generazione con Stable Diffusion 2.1  (D1, input per D2 e D3)
    03a_Finetuning_StableDiffusion2.1_baseline      (variante: solo reali)
    03b_Finetuning_StableDiffusion2.1_filtered      (variante principale: reali + augmentati, filtro qualità)

PIPELINE 3 — LDM from scratch  (D1)
    04a_LDM_basic                                   (variante: genera solo classe positiva)
    04b_LDM_extra1361                               (variante: +1361 raw, classe negativa, logica ottimizzata)

PIPELINE 4 — Classificatori  (D2)
    05_Classificatore_Baseline_ResNet-50_FineTuned
    06_Classificatore_RealSynthetic_ResNet-50_FineTuned         (fine-tuning parziale)
    06b_Classificatore_RealSynthetic_ResNet-50_FineTuned_Full   (fine-tuning completo)
        └── 07_Val_Classificatori_RS_AllVSPart      (confronto parziale vs completo)
    08_Classificatore_Synthetic_ResNet-50_FineTuned
        └── 09_Test_Classificatori                  (valutazione finale delle 3 configurazioni)

PIPELINE 5 — Sostenibilità  (D3)
    10_Classificatore_Real_Augmented_ResNet-50_FineTuned
    11_Classificatore_RealSyntheticPositive_ResNet-50_FineTuned
        └── 12_Valutazione_Sostenibilità
```

---

## Descrizione dei notebook

---

### `01_Preprocessing_RSNA_512_gray_MLO.ipynb`

**Scopo:** Preparare il dataset grezzo RSNA per l'uso nel progetto.

**Input:**
- `data/original/dataset/` — immagini PNG 512×512 originali
- `data/original/dataset/train.csv` — metadati RSNA (patient_id, image_id, laterality, view, cancer)

**Trasformazioni applicate:**
- Filtro sulla vista `MLO`
- Selezione di 1 sola immagine per paziente (per i positivi: il seno canceroso; per i negativi: selezione casuale)
- Normalizzazione e padding a 512×512 grayscale
- Divisione in train / val / test con stratificazione per label

**Output:**
- `data/processed/` — immagini preprocessate organizzate per split e label
- `data/processed/metadata/train.csv`, `val.csv`, `test.csv`, `all_processed.csv`
- `results/01_preprocessing/` — metriche e tracciamento CO₂

---

### `02_Data_Augmentation_Trad.ipynb`

**Scopo:** Aumentare il numero di campioni positivi nel training set tramite tecniche di augmentation tradizionale.

**Input:** `data/processed/` (dataset preprocessato)

**Trasformazioni applicate** (solo sulle immagini positive del train set):
- Variazioni di contrasto e luminosità
- Aggiunta di rumore gaussiano
- Flip orizzontale

**Output:**
- `data/real_augmented/` — immagini originali + campioni positivi aumentati, con CSV aggiornato
- `results/02_data_augmentation/` — statistiche augmentation e CO₂

---

### `03a_Finetuning_StableDiffusion2.1_baseline.ipynb`

**Scopo:** Fine-tuning baseline di Stable Diffusion 2.1 su immagini mammografiche MLO, addestrato sull'intero dataset reale preprocessato.

**Input:**
- `data/processed/` — immagini reali
- Modello base `stabilityai/stable-diffusion-2-1` (scaricato da HuggingFace)

**Training:** fine-tuning DreamBooth-style con `diffusers` + `accelerate`; le immagini di staging vengono create temporaneamente e rimosse al termine

**Output:**
- `experiments/20260607_sd21_rsna_mlo_512/` — checkpoint del modello (su Google Drive)

---

### `03b_Finetuning_StableDiffusion2.1_filtered.ipynb`

**Scopo:** Variante principale dell'esperimento SD2.1 — fine-tuning con 100 inference step, selezione del checkpoint migliore su validation set, generazione e filtraggio qualitativo delle immagini sintetiche.

**Input:**
- `data/processed/` + `data/real_augmented/` — dati di training
- Modello base `stabilityai/stable-diffusion-2-1`

**Pipeline:**
1. Fine-tuning del modello con dati reali + augmentati
2. Valutazione dei checkpoint su validation set (FID, IS, Precision, Recall)
3. Generazione di 2722 immagini (1361 positive + 1361 negative) con il checkpoint migliore (`checkpoint-3000`)
4. Filtro qualità con maschera adattiva: vengono scartate le immagini con artefatti; rimangono 1361 positive e 1361 negative
5. Calcolo metriche finali su test set

**Output:**
- `data/synthetic/fine_tuned/positive/` — 1361 immagini positive sintetiche filtrate
- `data/synthetic/fine_tuned/negative/` — 1361 immagini negative sintetiche filtrate
- `results/03b_finetuning_filtered/` — metriche FID/IS per classe, report filtro, tracciamento CO₂

---

### `04a_LDM_basic.ipynb`

**Scopo:** Addestramento di un modello LDM (Latent Diffusion Model) from scratch con Keras, su soli dati reali preprocessati.

**Input:** `data/processed/`

**Architettura:** U-Net con encoder VAE, addestrata in locale su GPU

**Output:**
- `experiments/20260617_ldm_basic/` — pesi del modello, curve di training, metriche generative
- `results/04a_ldm_basic/` — FID, IS, immagini campione generate

---

### `04b_LDM_extra1361.ipynb`

**Scopo:** Variante dell'esperimento LDM from scratch (`04a`) che ne estende e raffina la fase di generazione e valutazione, mantenendo la stessa architettura (VAE + U-Net) e gli stessi dati reali di training. Rispetto alla versione *basic* genera 1361 immagini *raw* aggiuntive (4083 in totale) e ne seleziona 1361 tramite il filtro adattivo, per ottenere un sottoinsieme di qualità superiore. Introduce inoltre una revisione della logica delle funzioni matematiche della diffusione e del campionamento, un tracciamento della sostenibilità (EcoTracker) più robusto, l'estensione della generazione anche alla classe negativa e nuovi grafici per l'interpretazione dei risultati. La selezione del checkpoint migliore è gestita interamente via script Python, tramite uno *sweep* delle metriche FID / IS / PRDC sul validation set.

**Input:** `data/processed/` — immagini reali preprocessate

**Output:**
- `experiments/20260619_ldm_extra1361/` — pesi, checkpoint, metriche
- `results/04b_ldm_keras_v2_extra1361/` — metriche, grafici e confronto con `04a`

---

### `05_Classificatore_Baseline_ResNet-50_FineTuned.ipynb`

**Scopo:** Implementazione di un classificatore binario di riferimento (baseline), addestrato sui soli dati reali preprocessati. I risultati saranno fondamentali per comprendere la reale efficacia delle immagini sintetiche generate. 
Nel notebook sono presenti celle relative alla valutazione in Validation Set del classificatore, per consentire una comprensione iniziale delle capacità di generalizzazione e predittive del modello sviluppato.

**Input:** `data/processed/` (train + val)

**Architettura ResNet-50:**
- Backbone: ResNet-50 con pesi ImageNet, input 224×224
- Head: GlobalAveragePooling → Dense(256) → BatchNorm → LeakyReLU → Dropout(0.5) → Dense(1, sigmoid)
- **Fase 1 (head training):** backbone congelato, lr=1e-3, loss=binary_crossentropy, class_weight bilanciato
- **Fase 2 (fine-tuning):** strati profondi scongelati da `conv4_block5_1_conv`, lr=1e-5, loss=BinaryFocalCrossentropy(γ=2, α=0.75)

**Output:**
- `experiments/exp20260618_baseline_resnet50_fine_tuned_batch_size_16/` — modello migliore (`.keras`), log training, metriche validation

---

### `06_Classificatore_RealSynthetic_ResNet-50_FineTuned.ipynb`

**Scopo:** Classificatore addestrato su dati reali + dati sintetici, con fine-tuning parziale del backbone. Il dataset utilizzato per il training rimane comunque sbilanciato, ma con le seguenti percentuali: 55% casi negativi, 45% casi positivi. 
Nel notebook sono presenti celle relative alla valutazione in Validation Set del classificatore, per consentire una comprensione iniziale delle capacità di generalizzazione e predittive del modello sviluppato.

**Input:**
- `data/processed/` — immagini reali
- `data/synthetic/fine_tuned/` — immagini sintetiche

**Stessa architettura di** `05`, diverso insieme di training.

**Output:**
- `experiments/exp20260617_real_synth_resnet50_fine_tuned_batch_size_16/`

---

### `06b_Classificatore_RealSynthetic_ResNet-50_FineTuned_Full.ipynb`

**Scopo:** Variante di `06` con fine-tuning completo di tutti gli strati del backbone ResNet-50. La realizzazione di questa variante nasce dalla volontà di comprendere se le prestazioni non eccellenti ottenute nel notebook precedente fossero dovute ad uno scongelamento errato dei livelli della backbone. 
Nel notebook sono presenti celle relative alla valutazione in Validation Set del classificatore, per consentire una comprensione iniziale delle capacità di generalizzazione e predittive del modello sviluppato.

**Input:** stesso di `06`

**Output:**
- `experiments/exp20260618_real_synth_resnet50_fine_tuned_all_layers/`

---

### `07_Val_Classificatori_RS_AllVSPart.ipynb`

**Scopo:** Confronto delle due configurazioni Real+Synth sul validation set — fine-tuning parziale (`06`) vs fine-tuning completo (`06b`) — per scegliere la configurazione finale da portare al test set. Vengono confrontate le prestazioni in Validation Set, ai fini della Model Selection. Questo attraverso grafici e tabelle. L'obiettivo era distinguere tra due scenari principali: modello sistematicamente incerto, con probabilità vicine alla soglia decisionale; modello apparentemente sicuro, ma spesso orientato verso la classe sbagliata;

**Input:**
- Modelli da `exp20260617_real_synth_...` e `exp20260618_real_synth_...`
- `data/processed/metadata/val.csv`

**Output:**
- `results/07_val_classificatori_allVSpart/` — metriche di confronto, predizioni sul val set

---

### `08_Classificatore_Synthetic_ResNet-50_FineTuned.ipynb`

**Scopo:** Classificatore addestrato esclusivamente su immagini sintetiche generate da SD2.1 (nessun dato reale nel training set), per valutare se il modello diffusivo produce immagini sufficientemente informative. In questo caso il dataset risulta bilanciato. Applicando la medesima metodologia, nel notebook sono presenti celle relative alla valutazione in Validation Set del classificatore, per consentire una comprensione iniziale delle capacità di generalizzazione e predittive del modello sviluppato.

**Input:** `data/synthetic/fine_tuned/` (train sintetico) + `data/processed/` (val e test reali)

**Output:**
- `experiments/exp20260619_full_synth_resnet50_fine_tuned_batch_size_16/`

---

### `09_Test_Classificatori.ipynb`

**Scopo:** Valutazione finale delle tre configurazioni principali di classificatore sul test set reale (438 immagini: 365 sani, 73 malati). Notebook di riferimento per la risposta alla Seconda Domanda di Ricerca.

**Input:**
- Modelli `.keras` dai tre esperimenti: `baseline`, `real_synth`, `full_synth`
- `data/processed/metadata/test.csv`

**Metriche calcolate:** AUC, Accuracy, Precision, Recall, F1-score, soglia ottimale (Youden), distribuzione delle probabilità previste, analisi dei casi difficili (errati da tutti i modelli), ROC-Curve, PrecisionRecall-Curve

**Output:**
- `results/09_test_classificatori/tables/...`
- `results/09_test_classificatori/predictions/...`
- `results/09_test_classificatori/figures/...`

---

### `10_Classificatore_Real_Augmented_ResNet-50_FineTuned.ipynb`

**Scopo:** Classificatore addestrato con dati reali + augmentation tradizionale (dati provenienti dal notebook `02`). Utilizzato come termine di paragone rispetto al classificatore *Real + Synthetic* per la Domanda di Ricerca D3 (sostenibilità).

**Input:**
- `data/processed/` — immagini reali
- `data/real_augmented/` — campioni positivi augmentati

**Stessa architettura ResNet-50 in due fasi degli altri classificatori.**

**Output:**
- `experiments/exp_trad_aug_resnet50/` — modello, log training
- Tracciamento CO₂ tramite `eco_tracker.py`

---
### `11_Classificatore_RealSyntheticPositive_ResNet-50_FineTuned.ipynb`

**Scopo:** Classificatore addestrato con dati reali + sintetici positivi. Utilizzato come termine di paragone rispetto al classificatore del notebook precedente per la Domanda di Ricerca D3 (sostenibilità).

**Input:**
- `data/processed/` — immagini reali
- `data/synthetic/positive` — campioni positivi sintetici

**Stessa architettura ResNet-50 in due fasi degli altri classificatori.**

**Output:**
- `experiments/exp_synth_pos_resnet50/` — modello, log training
- Tracciamento CO₂ tramite `eco_tracker.py`

---

### `12_Valutazione_Sostenibilità.ipynb`

**Scopo:** Confronto prestazionale e ambientale tra le strategie Real+Augmented (augmentation tradizionale) e Real+Synth (augmentation diffusiva). Notebook di riferimento per la risposta alla D3.

**Input:**
- Modelli da `exp_trad_aug_resnet50` e `exp20260617_real_synth_...`
- `data/processed/metadata/test.csv`
- File `ecotracker/*.json` con dati di consumo energetico e CO₂ raccolti durante training e generazione

**Output:**
- `results/test_trad_aug_vs_real_synth/` — metriche comparative, predizioni
- Grafici: confronto AUC/F1, CO₂ per fase, trade-off prestazioni/costo ambientale

---

## Team

| Nome | GitHub |
|---|---|
| Enzo Fumagalli | [@EnzoFumagalli](https://github.com/EnzoFumagalli) |
| Federico Marras | [@MarrasFederico](https://github.com/MarrasFederico) |
| Alexandro Sanna | [@AlexandroSanna](https://github.com/AlexandroSanna) |
| Samuele Nonnis | [@SamueleNonnis](https://github.com/SamueleNonnis) |

---

## Acknowledgements 
Questo progetto è stato realizzato dal gruppo composto da Enzo Fumagalli, Federico Marras, Alexandro Sanna e Samuele Nonnis, nell’ambito dell'insegnamento Deep Learning, annualità 2026, erogato dal Corso di Laurea in Informatica Applicata e Data Analytics (IADA) dell'Università degli Studi di Cagliari.
