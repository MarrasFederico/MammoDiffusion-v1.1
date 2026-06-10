# MammoDiffusion Studio

Interfaccia Gradio locale, ispirata al layout essenziale di Fooocus, per
generare mammografie MLO con il modello Stable Diffusion 2.1 fine-tuned.

L'app usa esattamente:

- il checkpoint scelto sulla validation: `checkpoint-3000`;
- i prompt positivo e negativo definiti nel notebook 3;
- `50` inference step e guidance scale `7.5` come valori predefiniti.

Le immagini vengono generate sequenzialmente per contenere l'uso della VRAM e
salvate in `assets/mammodiffusion_gradio/outputs/`. Questa cartella è separata
da `data/synthetic/fine_tuned`, quindi usare la demo non modifica il dataset
ufficiale né i risultati finali.

## Prerequisiti

I pesi non vengono pubblicati su GitHub. Prima dell'avvio devono essere
presenti localmente:

- `experiments/20260607_sd21_rsna_mlo_512/pretrained_model/stable-diffusion-2-1-base`;
- `experiments/20260607_sd21_rsna_mlo_512/model/checkpoint-3000/unet`.

## Avvio

Dalla root del progetto, usando lo stesso ambiente del notebook 3:

```bash
python -m pip install -r assets/mammodiffusion_gradio/requirements.txt
python assets/mammodiffusion_gradio/app.py --open-browser
```

L'interfaccia sarà disponibile su <http://127.0.0.1:7860>.

Per renderla raggiungibile nella rete locale:

```bash
python assets/mammodiffusion_gradio/app.py --host 0.0.0.0
```

L'opzione `--share` crea invece un link Gradio pubblico temporaneo. Non usarla
con dati sensibili.

## Controlli

- **Etichetta** seleziona il prompt standard positivo o negativo.
- **Numero di immagini** genera da 1 a 12 immagini.
- **Seed iniziale** vale `-1` per un seed casuale; impostando un intero, la
  generazione è riproducibile.
- Le impostazioni avanzate consentono di modificare step e guidance scale
  senza cambiare i valori predefiniti usati nell'esperimento.

La demo è destinata esclusivamente a ricerca e presentazione, non all'uso
clinico.

## Esempi

Alcuni output riproducibili generati dalla demo con etichetta positiva,
checkpoint `checkpoint-3000` e seed consecutivi a partire da `42` sono
disponibili nella cartella [`examples/`](examples/).

| Seed 42 | Seed 43 |
|---|---|
| ![Output positivo seed 42](examples/positive_seed_42.png) | ![Output positivo seed 43](examples/positive_seed_43.png) |

| Seed 44 | Seed 45 |
|---|---|
| ![Output positivo seed 44](examples/positive_seed_44.png) | ![Output positivo seed 45](examples/positive_seed_45.png) |
