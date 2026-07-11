# MammoDiffusion Studio

Interfaccia Gradio locale, ispirata al layout essenziale di Fooocus, per
generare mammografie MLO con due generatori MammoDiffusion selezionabili dalla
GUI.

L'app usa esattamente:

- il modello Stable Diffusion 2.1 fine-tuned del notebook 02:
  `checkpoint-3000`;
- il diffusore from scratch con VAE custom del notebook 06:
  `ldm_step070000.keras`;
- i prompt positivo e negativo definiti per il fine-tuning SD2.1;
- per il modello 02, `50` inference step e guidance scale `7.5` come valori
  predefiniti;
- per il modello 06, `100` sample step e guidance scale `1.5` come valori
  predefiniti.

Le immagini vengono generate sequenzialmente per contenere l'uso della VRAM e
salvate in `assets/mammodiffusion_gradio/outputs/`. Questa cartella è separata
da `data/synthetic/fine_tuned`, quindi usare la demo non modifica il dataset
ufficiale né i risultati finali.

Ogni richiesta di generazione viene eseguita in un subprocess dedicato. Quando
il subprocess termina, la VRAM usata dal modello selezionato viene rilasciata
dal sistema, permettendo di passare dal modello 02 al 06, e viceversa, senza
riavviare Gradio. I log tecnici del worker vengono salvati in `worker.log`
dentro la cartella output della singola generazione.

## Prerequisiti

I pesi non vengono pubblicati su GitHub. Prima dell'avvio devono essere
presenti localmente:

- `notebooks/pretrained_model/stable-diffusion-2-1-base`;
- `experiments/diffusers/02_sd21_filtered_100steps/model/checkpoint-3000/unet`;
- `experiments/diffusers/06_ldm_extra1361_fromscratch/checkpoints_ldm/ldm_step070000.keras`;
- `experiments/diffusers/06_ldm_extra1361_fromscratch/models/vae_decoder_best.keras`;
- `experiments/diffusers/06_ldm_extra1361_fromscratch/latents/latent_stats.npz`.

## Avvio

Dalla root del progetto, usando un ambiente con PyTorch/Diffusers e
TensorFlow/Keras:

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

- **Modello** seleziona `checkpoint-3000` del fine-tuning 02 oppure
  `ldm_step070000.keras` del diffusore from scratch 06.
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
modello 02 `checkpoint-3000` e seed consecutivi a partire da `42` sono
disponibili nella cartella [`examples/`](examples/).

| Seed 42 | Seed 43 |
|---|---|
| ![Output positivo seed 42](examples/positive_seed_42.png) | ![Output positivo seed 43](examples/positive_seed_43.png) |

| Seed 44 | Seed 45 |
|---|---|
| ![Output positivo seed 44](examples/positive_seed_44.png) | ![Output positivo seed 45](examples/positive_seed_45.png) |
