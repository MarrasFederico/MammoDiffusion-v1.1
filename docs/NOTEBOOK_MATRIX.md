# Notebook matrix v2

La matrice notebook-first vive in `notebooks/3_classifiers_matrix/` e non modifica i notebook
storici in `notebooks/3_classifiers/`.

- Stage 1: 112 notebook dedicati (`4 architetture × 28 varianti`), ciascuno gestisce i seed
  `17, 42, 73` e chiama `classifier_experiment_runner.execute_configuration`.
- Notebook generali: `00_Matrice_Esperimenti.ipynb` e `00b_Stato_Esecuzioni.ipynb`.
- Stage 2: generazione differita con `scripts/create_classifier_stage2_notebooks.py`; il comando
  rifiuta union assenti, firme non valide e union vuote. Per ogni generatore selezionato produce
  20 notebook (`5 regimi × 4 architetture`).
- Confronti: `04a_v2`–`04h_v2` in `notebooks/4_comparisons_and_test/`, tutti validation-only.

L’inventario macchina è in `results/notebook_inventory/notebook_inventory.{json,csv}`. Ogni
notebook Stage 1 ha un job logico e uno stato dataset; non esistono notebook orfani.

## Blocker dataset

- `RSB_FULL_04_sd21_lora`: richiede 2722 immagini per classe, ma la sorgente canonica dichiarata
  ne risolve 1361 e manca un manifest full deterministico.
- `RSP_CONTROLLED_05_ldm_basic_fromscratch` e `RSP_FULL_05_ldm_basic_fromscratch`: i file G05 non
  sono risolvibili dalla posizione corrente, dal lineage di migrazione o da un manifest canonico.

Questi tre dataset restano visibili in 12 notebook `BLOCKED`, ma non entrano nei 300 job
eseguibili Stage 1.

## Semantica

`MODE = "auto"` esegue checkpoint reuse/train → validation → metriche → ensemble quando i tre
seed sono completi. `locked-test` non è una modalità ammessa dai notebook di configurazione.
