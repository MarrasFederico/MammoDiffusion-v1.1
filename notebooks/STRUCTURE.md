# Struttura dei notebook

La migrazione da `notebooks2/` e' stata completata. La cartella `notebooks/` e' ora
la sorgente ufficiale; la precedente struttura piatta e' conservata localmente in
`old/` ed e' esclusa da Git.

## Convenzioni

- `1_preprocessing/`: preparazione e augmentation dei dati.
- `2_diffusers/`: esperimenti generativi numerati in ordine evolutivo.
- `3_generator_benchmark/`: benchmark unificato e selezione per famiglia.
- `4_downstream_classifiers/`: MaxViT-512, Mammo-FM, confronto validation e test locked.
- `utility/`: helper condivisi importati da tutti i notebook tramite bootstrap.

I notebook downstream sono parametrizzati con `condition` e `seed`; non esiste un notebook per
ognuno dei 24 job. Il runner canonico è `scripts/run_downstream_classifier.py`.
