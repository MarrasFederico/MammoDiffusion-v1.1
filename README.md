<div align="center">

<img src="assets/logo_MammoDiffusion.png" alt="MammoDiffusion Logo" width="180"/>

# MammoDiffusion v2

**Benchmark publication-oriented di generatori di mammografie sintetiche e verifica downstream compatta.**

[English version](README_EN.md)

</div>

## Obiettivo scientifico

MammoDiffusion v2 confronta generatori fine-tuned e from scratch sul dataset RSNA Breast Cancer Detection. Il progetto concentra la profondità scientifica sulla qualità delle immagini generate e usa pochi classificatori downstream per verificare l’utilità dei sintetici.

- **RQ1 — Qualità dei generatori:** quale generatore fine-tuned e quale from scratch offrono il miglior equilibrio tra fedeltà, diversità, coverage e assenza di memorizzazione?
- **RQ2 — Utilità downstream:** aggiungere mammografie sintetiche positive migliora la classificazione rispetto ai soli reali e alla traditional augmentation?
- **RQ3 — Robustezza rispetto al classificatore:** l’effetto è coerente tra un modello general-purpose moderno e un foundation model mammography-specific?

Il test non partecipa alla selezione di generatori, checkpoint o soglie.

## Disegno compatto

Il benchmark unificato valuta tutti i candidati validi del registry con 1.361 positivi per generatore, output RAW e FILTERED separati, bootstrap deterministico e due spazi indipendenti:

- InceptionV3 per FID/KID standard e continuità con la letteratura;
- RAD-DINO congelato come encoder medico indipendente, con la limitazione che non è mammography-specific.

KID RAD-DINO è il criterio primario. FID, precision, recall, density, coverage, LPIPS, MS-SSIM, duplicati, validità tecnica e nearest neighbour train/validation completano la valutazione. Viene proposto un vincitore `finetuned` e uno `from_scratch`; l’approvazione manuale firmata è obbligatoria prima dei classificatori.

La verifica downstream contiene soltanto:

```text
2 architetture: MaxViT-512, Mammo-FM
4 condizioni: real_only, real_augmented,
              real_plus_best_finetuned_positive,
              real_plus_best_fromscratch_positive
3 seed: 17, 42, 73
= 24 job primari e 8 ensemble
```

MaxViT-512 è il backbone general-purpose convolution/transformer. Mammo-FM è il foundation model con pretraining di dominio. ResNet-50 rimane una baseline storica V1; RAD-DINO non è un classificatore downstream.

## Flusso canonico

```text
preprocessing
→ traditional augmentation
→ sviluppo/generazione
→ generator_benchmark
→ generator_selection
→ approvazione esplicita
→ downstream_validation
→ ensemble dei tre seed
→ freeze del protocollo
→ locked_test one-shot
→ statistiche patient-level
→ final_report
```

I job si eseguono manualmente uno alla volta. Non esiste uno scheduler da cluster e non sono richiesti certificati GPU o canary firmati. Il runner mantiene dataset validation, lock per singolo experiment ID, checkpoint atomici, resume, validation inference, metriche e stato terminale.

## Struttura attiva

```text
configs/
  generator_benchmark_protocol.json
  generator_registry.json
  downstream_classifier_protocol.json
  downstream_classifier_jobs.json
  optional_downstream_ablations.json
notebooks/
  1_preprocessing/
  2_diffusers/
  3_generator_benchmark/
  4_downstream_classifiers/
  utility/
scripts/
  run_generator_benchmark.py
  approve_generator_selection.py
  run_downstream_classifier.py
  build_downstream_ensembles.py
  lock_downstream_test.py
docs/
  publication_experimental_design.md
  execution_guide.md
```

`configs/approved_generators.json` viene creato soltanto dopo un benchmark reale e un’approvazione esplicita; non contiene vincitori hardcoded.

## Esecuzione

Consultare [`docs/execution_guide.md`](docs/execution_guide.md). I punti di ingresso principali sono:

```bash
python scripts/run_generator_benchmark.py --dry-run
python scripts/list_downstream_jobs.py
python scripts/status_downstream_classifiers.py --json
python scripts/run_downstream_classifier.py \
  --architecture maxvit512 \
  --condition real_only \
  --seed 17 \
  --gpu 0
```

La metrica downstream primaria è PR-AUC patient-level. L’ensemble medio dei seed è il risultato principale; le soglie vengono selezionate su validation e congelate prima del test. Otto confronti preregistrati usano bootstrap patient-level e correzione Holm.

## Licenze e dati pesanti

Dataset, immagini sintetiche, pesi e checkpoint non sono versionati. Mammo-FM è utilizzabile solo secondo la licenza accademica applicabile: vedere [`docs/mammo_fm_license_note.md`](docs/mammo_fm_license_note.md). Non committare pesi originali o derivati Mammo-FM.

I risultati ResNet-50 V1 restano storici e separati dalle statistiche V2: [`docs/legacy_v1_classifier_results.md`](docs/legacy_v1_classifier_results.md). La pipeline completa precedente alla semplificazione è recuperabile dal tag Git `classifier-matrix-v2-full`.

## Documentazione

- [`docs/publication_experimental_design.md`](docs/publication_experimental_design.md): protocollo completo e limitazioni;
- [`docs/generator_benchmark_protocol.md`](docs/generator_benchmark_protocol.md): inclusione, metriche, bootstrap e ranking;
- [`docs/downstream_classifier_protocol.md`](docs/downstream_classifier_protocol.md): fairness, ensemble e statistiche;
- [`docs/execution_guide.md`](docs/execution_guide.md): sequenza operativa minima.

## Team

| Nome | GitHub |
|---|---|
| Enzo Fumagalli | [@EnzoFumagalli](https://github.com/EnzoFumagalli) |
| Federico Marras | [@MarrasFederico](https://github.com/MarrasFederico) |
| Alexandro Sanna | [@AlexandroSanna](https://github.com/AlexandroSanna) |
| Samuele Nonnis | [@SamueleNonnis](https://github.com/SamueleNonnis) |

Progetto sviluppato per l’insegnamento di Deep Learning, annualità 2026, Corso di Laurea in Informatica Applicata e Data Analytics dell’Università degli Studi di Cagliari.
