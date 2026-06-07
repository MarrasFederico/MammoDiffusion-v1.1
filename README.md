<div align="center">

<!-- Sostituisci il link qui sotto con il tuo logo -->
<img src="https://drive.google.com/file/d/1iOR3yjEsapSYa4BQd2JCcLxW-3z2QP_o/view?usp=drive_link" alt="MammoDiffusion Logo" width="180"/>

# 🩻 MammoDiffusion

**Generazione condizionata di immagini mammografiche sintetiche tramite modelli diffusivi**  
per il miglioramento della classificazione del cancro al seno.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RSNA%20Breast%20Cancer-pink?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

</div>

---

## 📋 Indice

- [Descrizione del Progetto](#-descrizione-del-progetto)
- [Domande di Ricerca](#-domande-di-ricerca)
- [Struttura della Repository](#-struttura-della-repository)
- [Dataset](#-dataset)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Risultati](#-risultati)
- [Team](#-team)

---

## 📌 Descrizione del Progetto

**MammoDiffusion** è un progetto di Deep Learning focalizzato sulla costruzione e l'addestramento di un **modello diffusivo** per la generazione condizionata di immagini mammografiche sintetiche.

L'obiettivo principale è valutare se le immagini sintetiche generate da modelli diffusivi possano migliorare l'identificazione di casi positivi al cancro al seno, affrontando il problema dello sbilanciamento delle classi e le limitazioni legate alla disponibilità di dati clinici reali.

Il progetto utilizza il dataset **RSNA Breast Cancer Detection**, filtrato sulla sola vista **MLO (Mediolateral Oblique)**.

---

## 🔬 Domande di Ricerca

### D1 — Qualità della Generazione
> *Un'architettura diffusiva costruita from scratch e una fine-tuned sono in grado di generare campioni realistici e sufficientemente vari?*

- **Obiettivo:** Valutare qualità e varietà dei campioni generati
- **Metodi:** Addestramento di un modello diffusivo from scratch + fine-tuning di un modello pre-addestrato
- **Valutazione:** Metriche **FID** (Fréchet Inception Distance) e **Inception Score**

---

### D2 — Impatto sulla Classificazione
> *L'aggiunta di campioni sintetici migliora le prestazioni dei classificatori rispetto all'utilizzo dei soli dati reali?*

- **Obiettivo:** Confrontare le capacità di classificatori addestrati in diverse configurazioni
- **Metodi:** Fine-tuning di un modello pre-addestrato su tre configurazioni:
  - Solo dati reali
  - Dati reali + augmentation tramite diffusione
  - Solo campioni sintetici
- **Valutazione:** Metriche di classificazione (AUC, F1, Accuracy, Recall)

---

### D3 — Sostenibilità e Responsible AI ♻️
> *La data augmentation tramite generazione è realmente conveniente rispetto a quella tradizionale, sia in termini di prestazioni che di sostenibilità?*

- **Obiettivo:** Determinare se i guadagni prestazionali giustifichino il costo computazionale della generazione diffusiva
- **Metodi:** Confronto tra classificatori addestrati con dati sintetici da diffusione vs. tecniche di augmentation tradizionali
- **Valutazione:** Trade-off prestazioni / costo computazionale / impatto ambientale (CO₂, tempo di training)

---

## 📁 Struttura della Repository

```
MammoDiffusion/
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb       # Analisi esplorativa del dataset
│   ├── 02_preprocessing.ipynb          # Preprocessing delle immagini
│   ├── 03_diffusion_scratch.ipynb      # Modello diffusivo from scratch
│   ├── 04_diffusion_finetuning.ipynb   # Fine-tuning modello pre-addestrato
│   └── 05_classification.ipynb         # Training e valutazione classificatori
│
├── 🐍 src/
│   ├── models/                         # Architetture dei modelli
│   ├── data/                           # Dataloader e preprocessing
│   ├── training/                       # Loop di training
│   └── evaluation/                     # Metriche e valutazione
│
├── 📊 results/                         # Grafici, metriche, output (su Drive)
│
├── 🗂️ data/                            # Dataset (NON incluso nella repo → su Drive)
│   ├── raw/                            # Immagini originali RSNA
│   └── processed/                      # Immagini preprocessate (MLO)
│
├── 📄 README.md                        # Questo file
├── 📦 requirements.txt                 # Dipendenze Python
└── 🚫 .gitignore                       # File esclusi dalla repo
```

> ⚠️ **Nota:** La cartella `data/` **non è inclusa** nella repository per via delle dimensioni. Il dataset è disponibile su Google Drive condiviso del team.

---

## 🗄️ Dataset

| Proprietà | Dettaglio |
|---|---|
| **Nome** | RSNA Breast Cancer Detection |
| **Fonte** | [Kaggle — RSNA Breast Cancer 512 PNGs](https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs) |
| **Vista** | MLO (Mediolateral Oblique) |
| **Formato** | PNG, 512×512 px |
| **Accesso** | Google Drive del team |

---

## ⚙️ Installazione

### 1. Clona la repository

```bash
git clone https://github.com/team/MammoDiffusion.git
cd MammoDiffusion
```

### 2. Crea un ambiente virtuale (consigliato)

```bash
python -m venv venv

# Attivazione su Windows
venv\Scripts\activate

# Attivazione su Mac/Linux
source venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Scarica il dataset

Scarica la cartella `data/` da **Google Drive** (link condiviso dal team) e posizionala nella root del progetto.

---

## 🚀 Utilizzo

Apri ed esegui i notebook nella cartella `notebooks/` nell'ordine numerico indicato:

```
01 → 02 → 03 → 04 → 05
```

Ogni notebook è indipendente e include commenti esplicativi sui passaggi principali.

---

## 📈 Risultati

> *Sezione in aggiornamento durante lo sviluppo del progetto.*

I risultati (grafici, metriche, modelli) vengono salvati su **Google Drive** nella cartella condivisa del team.

---

## 👥 Team

| Nome | GitHub |
|---|---|
| Enzo Fumagalli | [@EnzoFumagalli](https://github.com/EnzoFumagalli) |
| Federico Marras | [@FedericoMarras](https://github.com/FedericoMarras) |
| Alexandro Sanna | [@username](https://github.com/AlexandroSanna) |
| Samuele Nonni | [@SamueleNonnis](https://github.com/SamueleNonnis) |

---

<div align="center">
  <sub>Progetto realizzato per il corso di Deep Learning — A.A. 2024/2025</sub>
</div>