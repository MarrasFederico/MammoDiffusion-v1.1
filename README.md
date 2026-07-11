<div align="center">

<img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>

# MammoDiffusion v2

**Generazione condizionata di mammografie sintetiche e valutazione della loro utilita per la classificazione del tumore al seno.**

[English version](README_EN.md)

</div>

## Obiettivo

MammoDiffusion confronta diffusori fine-tuned e modelli latent diffusion addestrati from scratch sul dataset RSNA Breast Cancer Detection. Le immagini sintetiche non vengono giudicate soltanto con FID, Inception Score e PRDC: il progetto misura anche il loro effetto su classificatori valutati esclusivamente su validation e test reali.

ResNet-50 e stato usato nella consegna originale come baseline sostenibile e riproducibile. La versione v2 estende il confronto a MaxViT-Tiny-512, MammoFM e RAD-DINO, introduce VAE adattati, LoRA e una U-Net from scratch piu evoluta, e cerca il miglior sistema assoluto anche quando richiede una combinazione costosa di dati reali, aumentati e sintetici.

## Domande di ricerca

1. I diffusori from scratch e Stable Diffusion 2.1 fine-tuned producono mammografie realistiche e sufficientemente varie?
2. I dati sintetici migliorano AUC, F1 e recall della classe positiva rispetto ai soli dati reali?
3. Un classificatore nativo a 512x512 chiarisce meglio l'utilita dei sintetici rispetto alla baseline ResNet-50 a 224x224?
4. Il fine-tuning del VAE di Stable Diffusion cambia FID, IS, PRDC e prestazioni downstream?
5. Inserire il VAE di Stable Diffusion nel diffusore from scratch migliora il VAE addestrato da zero?
6. La U-Net v3 con v-prediction e Min-SNR migliora le precedenti versioni from scratch?
7. LoRA raggiunge una qualita comparabile al fine-tuning completo con costo inferiore?
8. Real+Augmented+Synthetic supera le singole strategie di augmentation?
9. Quale sorgente sintetica e piu utile per ciascun classificatore, mantenendo split, seed, soglie e budget confrontabili?

## Struttura

```text
MammoDiffusion/
|-- assets/
|-- data/                              # locale, esclusa da Git
|-- experiments/
|   |-- diffusers/                     # cache e checkpoint esclusi da Git
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
|   |-- pretrained_model/              # singola copia locale di SD 2.1
|   `-- utility/
|-- results/
|   |-- preprocessing/
|   |-- diffusers/
|   |-- classifiers/
|   |-- comparisons/
`-- old/                               # struttura piatta locale, esclusa da Git
```

La lettera finale identifica una configurazione dello stesso classificatore. Il suffisso `z` e riservato al confronto conclusivo della famiglia. Notebook, esperimenti e risultati condividono lo stesso prefisso, per esempio `02a` per MaxViT RealOnly e `02z` per il confronto MaxViT.

Tutti i notebook attivi contengono un bootstrap che individua automaticamente la root del progetto ed espone `notebooks/utility/`; possono quindi essere avviati dalla root o da qualunque sottocartella di `notebooks/`.

## Notebook

### Preprocessing

| Notebook | Funzione |
|---|---|
| `1_preprocessing/01_Preprocessing_RSNA_512_gray_MLO.ipynb` | Preprocessing RSNA, selezione MLO, split e immagini grayscale 512x512 |
| `1_preprocessing/02_Data_Augmentation_Trad.ipynb` | Augmentation tradizionale e relativo metadata |

### Diffusori

| ID | Notebook | Esperimento principale |
|---|---|---|
| 01 | `2_diffusers/01_SD21_Baseline_50steps.ipynb` | SD2.1 baseline |
| 02 | `2_diffusers/02_SD21_Filtered_100steps.ipynb` | Fine-tuning SD2.1, sampling e filtro |
| 03 | `2_diffusers/03_SD21_VAE_FineTuned.ipynb` | Fine-tuning del VAE SD2.1 |
| 04 | `2_diffusers/04_SD21_LoRA.ipynb` | Fine-tuning LoRA della U-Net |
| 05 | `2_diffusers/05_LDM_Basic_FromScratch.ipynb` | LDM base from scratch |
| 06 | `2_diffusers/06_LDM_Extra1361_FromScratch.ipynb` | LDM con VAE custom e dataset bilanciato |
| 07 | `2_diffusers/07_LDM_SDVAE_Extra1361.ipynb` | U-Net riaddestrata su latenti del VAE SD |
| 08 | `2_diffusers/08_LDM_v3_SDVAE_FromScratch.ipynb` | U-Net v3, v-prediction, Min-SNR e SD-VAE |

I notebook 07 e 08 applicano lo stesso flusso a entrambe le classi:

```text
generate -> filter -> validate (validation reale) -> test (test reale)
```

Le metriche finali sono separate in `metrics/positive/` e `metrics/negative/`. Per compatibilita, gli output positivi principali vengono copiati anche nei vecchi path piatti.

### Classificatori

| Famiglia | Notebook disponibili |
|---|---|
| ResNet-50 (`01`) | `01a` RealOnly, `01b` RealSynth partial, `01c` RealSynth full, `01d` SyntheticOnly, `01e` RealAugmented, `01f` RealSynthPositive |
| MaxViT-512 (`02`) | `02a`-`02f` equivalenti ResNet, `02i` RealAugSynth FromScratch, `02j` RealAugSynth FineTuned |
| MammoFM (`03`) | `03a` RealOnly, `03b` RealSynth FineTuned, `03c` RealSynth FromScratch, `03d` RealAugmented |
| RAD-DINO (`04`) | `04a` RealOnly, `04b` RealSynth |

I notebook di validation, test e confronto sono in `4_comparisons_and_test/`. `00z` confronta i diffusori; `01z` e `02z` chiudono rispettivamente le famiglie ResNet e MaxViT; `03z` confronta ResNet-50 e MaxViT-512.

## Dati sintetici

| Cartella | Sorgente |
|---|---|
| `data/synthetic/fine_tuned/` | SD2.1 fine-tuned |
| `data/synthetic/fine_tuned_vaeft/` | SD2.1 con VAE adattato |
| `data/synthetic/fine_tuned_lora/` | SD2.1 LoRA |
| `data/synthetic/fromscratch/` | LDM con VAE custom |
| `data/synthetic/fromscratch_new/` | LDM con SD-VAE |
| `data/synthetic/fromscratch_v3/` | LDM v3 con SD-VAE |

Ogni cartella finale contiene `positive/` e `negative/`. `data/` e esclusa da Git e deve essere preparata o ripristinata localmente.

## Strategia dei confronti

Per evitare di scegliere la sorgente sintetica sul test set, il progetto seguira due fasi:

1. valutazione di tutte le sorgenti sintetiche candidate con ricette e budget fissi, usando soltanto il validation set reale per ranking e soglie;
2. uso della migliore variante from scratch e della migliore variante fine-tuned nelle configurazioni piu costose `Real+Synth` e `Real+Augmented+Synthetic`, con un'unica valutazione finale sul test reale.

La matrice verra applicata a ResNet-50, MaxViT-512, MammoFM e RAD-DINO. Dove un backbone rende il confronto completo proibitivo, la selezione della sorgente restera comune e documentata, invece di cambiare dataset a posteriori per favorire un modello.

## Installazione

```bash
git clone https://github.com/MarrasFederico/MammoDiffusion-v2.git
cd MammoDiffusion-v2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch/TensorFlow e CUDA devono essere compatibili con la GPU utilizzata. Il notebook 08 configura esplicitamente RTX 5060 Ti/Blackwell e il percorso `libdevice`; il notebook 04 imposta la GPU prima di importare PyTorch. Dopo un cambio di GPU e necessario riavviare il kernel.

Il modello base condiviso e risolto da `notebooks/pretrained_model/stable-diffusion-2-1-base` oppure dalla variabile `MAMMODIFFUSION_SD21_BASE`. I VAE e gli adapter modificati restano invece nelle rispettive cartelle di esperimento.

## Demo Gradio

La demo locale e in `assets/mammodiffusion_gradio/`. Usa i path correnti degli esperimenti 02 e 06 e salva gli output temporanei fuori da Git.

## Riproducibilita

- split reali invariati tra gli esperimenti;
- soglie decisionali calcolate sul validation set con criterio di Youden;
- test set reale mantenuto separato dalla selezione di checkpoint, filtri e sorgenti sintetiche;
- seed e budget annotati nei notebook e nei manifest;
- FID, IS e PRDC calcolati per classe;
- checkpoint, dataset e cache pesanti esclusi da Git, metriche e codice versionati.

## Evoluzione

La priorita immediata e completare 04 LoRA e rieseguire 08 nella struttura definitiva. I risultati determineranno quali dataset sintetici usare nei confronti downstream completi. La direzione del progetto non e scegliere la tecnica meno costosa per principio: in mammografia un costo maggiore e accettabile quando produce un miglioramento robusto e riproducibile, soprattutto sul recall della classe positiva.

## Team

| Nome | GitHub |
|---|---|
| Enzo Fumagalli | [@EnzoFumagalli](https://github.com/EnzoFumagalli) |
| Federico Marras | [@MarrasFederico](https://github.com/MarrasFederico) |
| Alexandro Sanna | [@AlexandroSanna](https://github.com/AlexandroSanna) |
| Samuele Nonnis | [@SamueleNonnis](https://github.com/SamueleNonnis) |

Progetto sviluppato per l'insegnamento di Deep Learning, annualita 2026, Corso di Laurea in Informatica Applicata e Data Analytics dell'Universita degli Studi di Cagliari.
