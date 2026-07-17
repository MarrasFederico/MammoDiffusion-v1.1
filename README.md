<p align="center">
  <img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>
</p>

# MammoDiffusion v2

MammoDiffusion è un progetto notebook-first per studiare immagini mammografiche sintetiche e la loro utilità downstream. Il flusso publication v2 è modulare, leggibile e riproducibile: i notebook mostrano configurazione, dati, audit, chiamate alle utility, metriche, grafici e artefatti. Non esiste una pipeline automatica obbligatoria.

## Domande scientifiche

- **RQ1:** quale generatore fine-tuned e quale from-scratch bilanciano meglio fedeltà, diversità, coverage, efficienza e assenza di memorizzazione del train?
- **RQ2:** aggiungere mammografie positive sintetiche migliora la classificazione rispetto a dati reali soli e augmentation tradizionale?
- **RQ3:** l’effetto è coerente tra MaxViT-512 e Mammo-FM?

## Workflow notebook-first

1. Eseguire i notebook dei generatori se mancano gli output.
2. Eseguire `notebooks/3_generator_benchmark/01_Unified_Generator_Benchmark.ipynb`.
3. Ispezionare metriche RAW/FILTERED e pannelli diagnostici.
4. In `notebooks/3_generator_benchmark/02_Generator_Selection.ipynb`, scegliere manualmente un generatore fine-tuned e uno from-scratch.
5. Eseguire una volta `notebooks/04_classifiers/01_MaxViT512.ipynb`; cicla automaticamente 4 condizioni × 3 seed.
6. Eseguire una volta `notebooks/04_classifiers/02_MammoFM.ipynb`; cicla automaticamente 4 condizioni × 3 seed.
7. Eseguire `notebooks/04_classifiers/03_Validation_Comparison.ipynb` e congelare decisioni e soglie sulla validation.
8. Eseguire opzionalmente `notebooks/04_classifiers/04_Final_Evaluation_and_Report.ipynb` solo dopo aver identificato onestamente il dataset finale.

Il protocollo conserva esattamente **2 architetture × 4 condizioni × 3 seed = 24 esperimenti**. Le architetture sono MaxViT-512 e Mammo-FM; RAD-DINO è soltanto un feature extractor medico nel benchmark generativo. ResNet-50 è una baseline storica V1.

## Rigore scientifico

Il benchmark richiede un pool sintetico di 1.361 immagini, ma usa tutte le positive reali disponibili nella validation e valuta subset bilanciati di dimensione `min(real_reference_count, synthetic_pool_count)`. KID e PRDC usano repeated subsampling senza reinserimento; FID è secondaria e usa una sola ripetizione di default. Train memorization, validation similarity e duplicazione sintetica sono risultati distinti.

Checkpoint, early stopping e scheduler downstream monitorano validation PR-AUC. Il budget massimo di optimizer update è fisso entro ciascuna architettura. Validation e bootstrap sono patient-level; gli otto confronti principali usano correzione di Holm.

## Risultati e checkpoint

Il codice attivo scrive o consuma quattro radici canoniche:

- `results/preprocessing/`: riepiloghi di preprocessing e augmentation;
- `results/diffusers/`: metriche, grafici, sostenibilità e output degli sweep generativi;
- `results/publication_v2/`: benchmark/provenienza dei generatori, training dei classificatori e output di pubblicazione;
- `results/sustainability/`: analisi trasversali dei consumi.

Gli stage specifici dello sweep dell'esperimento 08 sono mantenuti perché prodotti dal phase planner e necessari alla provenienza delle singole run. Il default delle utility Keras-v2 è `results/diffusers/04_ldm_keras_v2`, mai la radice di `results/`.

I checkpoint dei classificatori in `results/publication_v2/classifiers/` sono stato di resume e valutazione: `checkpoint_latest`, `checkpoint_previous` e tutte le rappresentazioni del best checkpoint non devono essere potati. Il precedente test interno risulta storicamente riutilizzato e non costituisce una conferma indipendente incontaminata; vedere [stato del dataset finale](docs/PROTOCOL.md#7-historical-internal-test--status-and-limitation).

La pipeline scripted precedente è archiviata nel tag `publication-pipeline-scripted-v1`; la matrice da 300 job nel tag `classifier-matrix-v2-full`.

## Demo Gradio

`assets/mammodiffusion_gradio/app.py` legge la selezione corrente da `configs/selected_generators.json` e propone i due vincitori di famiglia con i rispettivi best checkpoint:

- G02, Stable Diffusion 2.1 fine-tuned, `checkpoint-3000`, sampling canonico a 100 step;
- G07, LDM from-scratch con SD-VAE, `ldm_unet_best_eval.keras` selezionato allo step 130000, sampling a 100 step.

La demo ha un README dedicato perché costituisce un'applicazione avviabile separatamente: [istruzioni Gradio](assets/mammodiffusion_gradio/README.md).

## Consegna portabile e Google Drive

Per un hand-off completo caricare `notebooks/`, `configs/`, `experiments/`, `results/` e il materiale `data/` consentito dal progetto. In particolare vanno inclusi:

- `notebooks/utility/diffusers_repo`, compresa la sua `.git` per verificare il commit fissato;
- `notebooks/pretrained_model/stable-diffusion-2-1-base`;
- i dataset sintetici filtrati usati dal benchmark e dai classificatori.

Conservare checkpoint, latenti, cache di checkpoint-validation, output di evaluation ed embedding cache. Escludere soltanto cache Hugging Face/composizioni rigenerabili, `__pycache__`, file `*.pyc`, smoke test e code di lavoro vuote. `experiments/diffusers/` può essere caricato come primo blocco, ma da solo non basta a riprendere l'esecuzione su un'altra macchina.

## Documentazione

- [Protocollo consolidato](docs/PROTOCOL.md) — disegno sperimentale, benchmark generativo, amendment Option B, selezione G02/G07, protocollo downstream 2 × 4 × 3, stato del test storico ed esecuzione manuale.
- [Stato dei generatori](docs/GENERATOR_STATUS.md)
- [Asset SD2.1/Diffusers condivisi](docs/SHARED_ASSETS.md)
- [Analisi di sostenibilità](docs/SUSTAINABILITY_ANALYSIS.md)
- [Licenza accademica Mammo-FM](docs/mammo_fm_license_note.md)

Dataset, immagini sintetiche, embedding e pesi restano locali e non devono essere committati. I pesi Mammo-FM sono soggetti alla relativa licenza accademica e non sono redistribuiti.

<details>
<summary>English overview</summary>

MammoDiffusion v2 is a notebook-first study of synthetic mammography and downstream utility. The workflow compares eligible fine-tuned and from-scratch generators, evaluates real-only, traditional-augmentation and synthetic-augmentation conditions, and measures consistency across MaxViT-512 and Mammo-FM. The first two classifier notebooks each run four conditions over seeds 17, 42 and 73, for 24 classifier experiments in total. Generator selection is validation-only and currently retains G02 and G07. Dataset files, synthetic images, embeddings and model weights remain local; Mammo-FM weights retain their academic-license restrictions.

</details>
