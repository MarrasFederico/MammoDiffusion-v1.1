# Struttura dei notebook

La migrazione da `notebooks2/` e' stata completata. La cartella `notebooks/` e' ora
la sorgente ufficiale; la precedente struttura piatta e' conservata localmente in
`old/` ed e' esclusa da Git.

## Convenzioni

- `1_preprocessing/`: preparazione e augmentation dei dati.
- `2_diffusers/`: esperimenti generativi numerati in ordine evolutivo.
- `3_classifiers/`: famiglie di classificatori; la lettera distingue la ricetta dati.
- `4_comparisons_and_test/`: validazione, test e confronti; il suffisso `z` indica il
  confronto conclusivo disponibile per una famiglia.
- `utility/`: helper condivisi importati da tutti i notebook tramite bootstrap.

I path di esperimenti e risultati seguono la stessa numerazione, per esempio
`notebooks/3_classifiers/02a_*`, `experiments/classifiers/maxvit512/02a_*` e
`results/classifiers/maxvit512/02x_*`.
