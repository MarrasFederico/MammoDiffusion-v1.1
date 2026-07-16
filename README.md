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
2. Eseguire `05_Unified_Generator_Benchmark.ipynb`.
3. Ispezionare metriche RAW/FILTERED e pannelli diagnostici.
4. In `06_Generator_Selection.ipynb`, scegliere manualmente un generatore fine-tuned e uno from-scratch.
5. Eseguire `07_MaxViT512_Downstream.ipynb` per 4 condizioni × 3 seed.
6. Eseguire `08_MammoFM_Downstream.ipynb` per 4 condizioni × 3 seed.
7. Eseguire `09_Downstream_Validation_Comparison.ipynb` e congelare decisioni e soglie sulla validation.
8. Eseguire opzionalmente `10_Final_Evaluation_and_Report.ipynb` solo dopo aver identificato onestamente il dataset finale.

Il protocollo conserva esattamente **2 architetture × 4 condizioni × 3 seed = 24 esperimenti**. Le architetture sono MaxViT-512 e Mammo-FM; RAD-DINO è soltanto un feature extractor medico nel benchmark generativo. ResNet-50 è una baseline storica V1.

## Rigore scientifico

Il benchmark richiede un pool sintetico di 1.361 immagini, ma usa tutte le positive reali disponibili nella validation e valuta subset bilanciati di dimensione `min(real_reference_count, synthetic_pool_count)`. KID e PRDC usano repeated subsampling senza reinserimento; FID è secondaria e usa una sola ripetizione di default. Train memorization, validation similarity e duplicazione sintetica sono risultati distinti.

Checkpoint, early stopping e scheduler downstream monitorano validation PR-AUC. Il budget massimo di optimizer update è fisso entro ciascuna architettura. Validation e bootstrap sono patient-level; gli otto confronti principali usano correzione di Holm.

## Risultati e dati legacy

Il nuovo codice scopre soltanto `results/publication_v2/`. I risultati V1 e della matrice ritirata restano nelle posizioni storiche e non contaminano il protocollo corrente. Il precedente test interno risulta storicamente riutilizzato: non è una conferma indipendente incontaminata. Vedere [protocollo consolidato — stato del dataset finale](docs/PROTOCOL.md#7-historical-internal-test--status-and-limitation).

La pipeline scripted precedente è archiviata nel tag `publication-pipeline-scripted-v1`; la matrice da 300 job nel tag `classifier-matrix-v2-full`.

## Documentazione

- [Protocollo consolidato](docs/PROTOCOL.md) — disegno sperimentale, benchmark generativo, amendment Option B, selezione G02/G07, protocollo downstream 2 × 4 × 3, stato del test storico ed esecuzione manuale.
- [Stato dei generatori](docs/GENERATOR_STATUS.md)
- [Asset SD2.1/Diffusers condivisi](docs/SHARED_ASSETS.md)
- [Analisi di sostenibilità](docs/SUSTAINABILITY_ANALYSIS.md)
- [Licenza accademica Mammo-FM](docs/mammo_fm_license_note.md)

Dataset, immagini sintetiche, embedding e pesi restano locali e non devono essere committati. I pesi Mammo-FM sono soggetti alla relativa licenza accademica e non sono redistribuiti.
