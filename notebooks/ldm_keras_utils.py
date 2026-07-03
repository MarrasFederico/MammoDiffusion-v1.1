from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ldm_project_paths import ExperimentPaths, find_project_root, get_experiment_paths

IMG_SIZE = 512
CHANNELS = 1
LATENT_SIZE = 64
LATENT_CHANNELS = 4
NUM_CLASSES = 2
EMBED_DIM = 128
MODEL_CHANNELS = 64
NUM_DIFF_STEPS = 1000
CFG_SCALE = 3.0
SAMPLE_STEPS = 100
CLASS_NAMES = {0: "Negativo (sano)", 1: "Positivo (cancro)"}


@dataclass(frozen=True)
class DiffusionSchedule:
    """Contenitore immutabile con tutti i coefficienti del noise schedule, precalcolati una volta sola e riusati a ogni step di training/sampling."""

    betas: tf.Tensor
    alphas: tf.Tensor
    alpha_bars: tf.Tensor
    sqrt_alpha_bars: tf.Tensor
    sqrt_one_minus_alpha_bars: tf.Tensor
    alpha_bars_prev: tf.Tensor
    posterior_variance: tf.Tensor
    posterior_log_variance: tf.Tensor


def configure_tensorflow(seed: int = 42) -> None:
    """Fissa i seed di TF/numpy per la riproducibilità degli esperimenti e stampa la GPU rilevata."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    gpus = tf.config.list_physical_devices("GPU")

    # Non usare memory_growth:
    # TensorFlow prealloca quasi tutta la memoria GPU disponibile.
    print("TF version:", tf.__version__)
    print("GPU disponibili:", gpus)


_VRAM_LAST_CURRENT: dict[str, int] = {}


def vram_gb(label: str = "VRAM", device: str = "GPU:0") -> None:
    """Stampa memoria GPU corrente/di picco e la variazione rispetto all'ultima chiamata, utile per individuare leak o picchi durante training/sampling."""
    try:
        info = tf.config.experimental.get_memory_info(device)
        current = int(info["current"])
        peak = int(info["peak"])
        previous = _VRAM_LAST_CURRENT.get(device)
        delta = 0 if previous is None else current - previous
        _VRAM_LAST_CURRENT[device] = current
        print(
            f"[{label}] current={current / 1e9:.2f} GB | "
            f"peak={peak / 1e9:.2f} GB | "
            f"delta={delta / 1e9:+.2f} GB"
        )
    except Exception as exc:
        print(f"[{label}] VRAM non disponibile su {device}: {exc}")


@tf.keras.utils.register_keras_serializable()
class SinusoidalTimeEmbedding(layers.Layer):
    """Trasforma il timestep di diffusione t in un vettore continuo (embedding sinusoidale + MLP) da iniettare nei ResBlock della U-Net."""

    def __init__(self, embed_dim, **kwargs):
        """Salva la dimensione dell'embedding e crea l'MLP a due livelli che proietta le frequenze sinusoidali."""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense1 = layers.Dense(embed_dim * 4, activation="relu")
        self.dense2 = layers.Dense(embed_dim * 4)

    def call(self, t):
        """Codifica ogni timestep con coppie seno/coseno a frequenze decrescenti (stile Transformer) e le passa attraverso l'MLP."""
        t = tf.cast(t, tf.float32)
        half = self.embed_dim // 2
        freqs = tf.exp(
            -math.log(10000.0) * tf.range(half, dtype=tf.float32) / float(half)
        )
        args = t[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return self.dense2(self.dense1(emb))

    def get_config(self):
        """Aggiunge embed_dim alla config standard del layer, necessario per ricostruire il layer al caricamento del modello salvato."""
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class LabelEmbedding(layers.Layer):
    """Mappa l'etichetta di classe (incluso lo slot "unconditional" per il classifier-free guidance) in un vettore della stessa dimensione del time embedding."""

    def __init__(self, num_classes, embed_dim, **kwargs):
        """Crea la tabella di embedding con una riga extra rispetto a num_classes, riservata alla classe fittizia usata per il training/sampling unconditional."""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = layers.Embedding(num_classes + 1, embed_dim * 4)

    def call(self, y):
        """Restituisce l'embedding associato alle etichette y."""
        return self.embedding(y)

    def get_config(self):
        """Aggiunge num_classes ed embed_dim alla config standard, necessari per ricostruire il layer al caricamento del modello salvato."""
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class ResBlock(layers.Layer):
    """Blocco residuale della U-Net che combina le feature spaziali con il time/label embedding e proietta sul numero di canali target."""

    def __init__(self, channels, embed_dim, **kwargs):
        """Crea le due convoluzioni con GroupNorm del ramo principale, la proiezione dell'embedding e la skip connection 1x1 per adattare i canali in ingresso."""
        super().__init__(**kwargs)
        self.channels = channels
        self.embed_dim = embed_dim
        self.norm1 = layers.GroupNormalization(groups=min(32, channels))
        self.conv1 = layers.Conv2D(channels, 3, padding="same")
        self.norm2 = layers.GroupNormalization(groups=min(32, channels))
        self.conv2 = layers.Conv2D(channels, 3, padding="same")
        self.emb_proj = layers.Dense(channels)
        self.skip = layers.Conv2D(channels, 1)
        self.act = layers.LeakyReLU(alpha=0.2)

    def call(self, x, emb):
        """Applica norm+conv alle feature, inietta l'embedding (broadcast spaziale) tra le due convoluzioni e somma la skip connection."""
        h = self.conv1(self.act(self.norm1(x)))
        emb_out = tf.reshape(self.emb_proj(self.act(emb)), [-1, 1, 1, self.channels])
        h = self.conv2(self.act(self.norm2(h + emb_out)))
        return h + self.skip(x)

    def get_config(self):
        """Aggiunge channels ed embed_dim alla config standard, necessari per ricostruire il layer al caricamento del modello salvato."""
        config = super().get_config()
        config.update({"channels": self.channels, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    """Blocco di self-attention spaziale usato nei livelli più profondi della U-Net per modellare dipendenze a lungo raggio tra regioni dell'immagine."""

    def __init__(self, channels, num_heads=4, **kwargs):
        """Crea la GroupNorm, il layer di multi-head attention (con key/value dim derivati dai canali) e la proiezione finale."""
        super().__init__(**kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.norm = layers.GroupNormalization(groups=min(32, channels))
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels // num_heads,
            value_dim=channels // num_heads,
        )
        self.proj = layers.Dense(channels)

    def call(self, x):
        """Appiattisce la mappa spaziale (H*W) in una sequenza di token, applica self-attention e ripristina la forma originale con una skip connection."""
        batch = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = self.channels
        h = self.norm(x)
        h = tf.reshape(h, [batch, height * width, channels])
        h = self.attn(h, h)
        h = self.proj(h)
        h = tf.reshape(h, [batch, height, width, channels])
        return x + h

    def get_config(self):
        """Aggiunge channels e num_heads alla config standard, necessari per ricostruire il layer al caricamento del modello salvato."""
        config = super().get_config()
        config.update({"channels": self.channels, "num_heads": self.num_heads})
        return config


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Calcola la KL divergence in forma chiusa tra due gaussiane diagonali, usata nel termine di loss variazionale della diffusione (VLB)."""
    return 0.5 * (
        logvar2
        - logvar1
        + tf.exp(logvar1 - logvar2)
        + tf.square(mean1 - mean2) * tf.exp(-logvar2)
        - 1.0
    )


def build_schedule(num_steps: int = NUM_DIFF_STEPS, s: float = 0.008) -> DiffusionSchedule:
    """Costruisce il cosine beta schedule (Nichol & Dhariwal) e precalcola tutti i coefficienti derivati (alpha_bar, varianze posteriori) in un'unica DiffusionSchedule."""
    t = np.linspace(0.0, float(num_steps), num_steps + 1, dtype=np.float64)
    f = np.cos((t / float(num_steps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    alpha_bars_np = f / f[0]
    betas_np = 1.0 - alpha_bars_np[1:] / alpha_bars_np[:-1]
    betas_np = np.clip(betas_np, 1e-4, 0.9999).astype(np.float32)

    betas = tf.constant(betas_np, dtype=tf.float32)
    alphas = 1.0 - betas
    alpha_bars = tf.math.cumprod(alphas)
    alpha_bars_prev = tf.concat([[1.0], alpha_bars[:-1]], axis=0)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=tf.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=tf.sqrt(1.0 - alpha_bars),
        alpha_bars_prev=alpha_bars_prev,
        posterior_variance=posterior_variance,
        posterior_log_variance=tf.math.log(tf.maximum(posterior_variance, 1e-20)),
    )


def extract(values, t, x_shape):
    """Estrae per ogni elemento del batch il coefficiente dello schedule corrispondente al proprio timestep t e lo reshapa per il broadcasting sull'immagine/latente."""
    batch_size = tf.shape(t)[0]
    out = tf.gather(values, t)
    return tf.reshape(out, [batch_size, 1, 1, 1])


def get_learned_log_variance(v, t, schedule: DiffusionSchedule):
    """Interpola in log-spazio tra il limite inferiore (beta_tilde) e superiore (beta) della varianza posteriore, pesando con il valore v predetto dalla rete (parametrizzazione learned-variance di IDDPM)."""
    shape = tf.shape(v)
    log_beta_t = tf.math.log(extract(schedule.betas, t, shape))
    log_beta_tilde_t = tf.math.log(
        extract(schedule.posterior_variance + 1e-8, t, shape)
    )
    v_sigmoid = tf.sigmoid(tf.clip_by_value(v, -8.0, 8.0))
    log_var = v_sigmoid * log_beta_t + (1.0 - v_sigmoid) * log_beta_tilde_t
    return tf.clip_by_value(log_var, -20.0, 2.0)


def get_learned_log_variance_eff(v, beta_eff, posterior_variance_eff):
    """Variante di get_learned_log_variance che usa beta e varianza posteriore "effettivi" già calcolati per il salto multi-step del sampling DDIM-like, evitando di rifare l'extract sullo schedule completo."""
    log_beta_eff = tf.math.log(beta_eff)
    log_beta_tilde_eff = tf.math.log(posterior_variance_eff + 1e-8)
    v_sigmoid = tf.sigmoid(tf.clip_by_value(v, -8.0, 8.0))
    log_var = v_sigmoid * log_beta_eff + (1.0 - v_sigmoid) * log_beta_tilde_eff
    return tf.clip_by_value(log_var, -20.0, 2.0)


def p_sample_ldm(
    ldm_model,
    schedule: DiffusionSchedule,
    z_t,
    t_int: int,
    t_prev_int: int,
    y_batch,
    guidance_scale: float = CFG_SCALE,
):
    """Esegue un singolo step di reverse diffusion da t a t_prev con classifier-free guidance: predice eps condizionato/non condizionato in un'unica chiamata batchata, ricava la stima di z0 e ne deriva media e varianza del passo precedente (nessun rumore aggiunto se t_prev è 0)."""
    batch_size = tf.shape(z_t)[0]
    t_batch = tf.fill([batch_size], t_int)
    t_prev = tf.fill([batch_size], t_prev_int)
    y_uncond = tf.fill([batch_size], NUM_CLASSES)

    z_in = tf.concat([z_t, z_t], axis=0)
    t_in = tf.concat([t_batch, t_batch], axis=0)
    y_in = tf.concat([y_batch, y_uncond], axis=0)

    out_combined = ldm_model([z_in, t_in, y_in], training=False)
    eps_combined, v_combined = tf.split(out_combined, 2, axis=-1)
    eps_cond, eps_uncond = tf.split(eps_combined, 2, axis=0)
    v_cond, _ = tf.split(v_combined, 2, axis=0)

    eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    v_pred = v_cond

    shape = tf.shape(z_t)
    ab_t = extract(schedule.alpha_bars, t_batch, shape)
    ab_prev = extract(schedule.alpha_bars, t_prev, shape)
    sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t_batch, shape)

    z0_pred = (z_t - sqrt_omab * eps_pred) / tf.sqrt(ab_t)
    b_eff = 1.0 - ab_t / ab_prev
    mean = (
        tf.sqrt(ab_prev) * b_eff / (1.0 - ab_t) * z0_pred
        + tf.sqrt(ab_t / ab_prev) * (1.0 - ab_prev) / (1.0 - ab_t) * z_t
    )

    if t_prev_int == 0:
        return mean

    post_var_eff = b_eff * (1.0 - ab_prev) / (1.0 - ab_t)
    log_var = get_learned_log_variance_eff(v_pred, b_eff, post_var_eff)
    std = tf.exp(0.5 * log_var)
    return mean + std * tf.random.normal(tf.shape(z_t))


def sampling_timesteps(num_steps: int) -> tuple[int, ...]:
    """Sottocampiona in modo uniforme i NUM_DIFF_STEPS timestep di training per ottenere una sequenza decrescente di num_steps timestep da usare nel sampling accelerato."""
    stride = max(1, NUM_DIFF_STEPS // int(num_steps))
    return tuple(range(0, NUM_DIFF_STEPS, stride))[::-1]


def make_compiled_sampler(
    ldm_model,
    vae_decoder,
    schedule: DiffusionSchedule,
    latent_mean,
    latent_std,
    num_steps: int = SAMPLE_STEPS,
    guidance_scale: float = CFG_SCALE,
    decode_on_cpu: bool = False,
):
    """Costruisce e compila (tf.function) un sampler end-to-end (loop di denoising + decodifica VAE) per un singolo seed/label, così da generare immagini a velocità da grafo invece che in eager mode."""
    timesteps = tf.constant(sampling_timesteps(num_steps), dtype=tf.int32)
    n_timesteps = tf.shape(timesteps)[0]
    guidance_scale_tensor = tf.constant(float(guidance_scale), dtype=tf.float32)
    latent_mean_tensor = tf.convert_to_tensor(latent_mean, dtype=tf.float32)
    latent_std_tensor = tf.convert_to_tensor(latent_std, dtype=tf.float32)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32, name="label"),
            tf.TensorSpec(shape=(2,), dtype=tf.int32, name="seed"),
        ],
        reduce_retracing=True,
    )
    def compiled_sampler(label, seed):
        """Genera un'immagine campionando rumore latente a partire dal seed dato, eseguendo il while_loop di reverse diffusion e infine decodificando con il VAE."""
        print(
            "[make_compiled_sampler] tracing sampling graph "
            f"num_steps={num_steps} guidance_scale={guidance_scale:g} "
            f"decode_on_cpu={decode_on_cpu}"
        )
        seed = tf.ensure_shape(tf.cast(seed, tf.int32), [2])
        z0 = tf.random.stateless_normal(
            (1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS),
            seed=seed,
        )

        def denoise_step(index, z_t):
            """Corpo del while_loop: esegue un singolo step di reverse diffusion con CFG (stessa logica di p_sample_ldm ma in forma graph-friendly con indici tensoriali) e avanza l'indice del timestep."""
            t_int = tf.gather(timesteps, index)
            t_prev_int = tf.cond(
                index + 1 < n_timesteps,
                lambda: tf.gather(timesteps, index + 1),
                lambda: tf.constant(0, dtype=tf.int32),
            )
            t_batch = tf.fill([1], t_int)
            t_prev = tf.fill([1], t_prev_int)
            y_batch = tf.reshape(label, [1])
            y_uncond = tf.fill([1], NUM_CLASSES)

            z_in = tf.concat([z_t, z_t], axis=0)
            t_in = tf.concat([t_batch, t_batch], axis=0)
            y_in = tf.concat([y_batch, y_uncond], axis=0)

            out_combined = ldm_model([z_in, t_in, y_in], training=False)
            eps_combined, v_combined = tf.split(out_combined, 2, axis=-1)
            eps_cond, eps_uncond = tf.split(eps_combined, 2, axis=0)
            v_cond, _ = tf.split(v_combined, 2, axis=0)

            eps_pred = eps_uncond + guidance_scale_tensor * (eps_cond - eps_uncond)
            v_pred = v_cond

            shape = tf.shape(z_t)
            ab_t = extract(schedule.alpha_bars, t_batch, shape)
            ab_prev = extract(schedule.alpha_bars, t_prev, shape)
            sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t_batch, shape)

            z0_pred = (z_t - sqrt_omab * eps_pred) / tf.sqrt(ab_t)
            b_eff = 1.0 - ab_t / ab_prev
            post_var_eff = b_eff * (1.0 - ab_prev) / (1.0 - ab_t)
            mean = (
                tf.sqrt(ab_prev) * b_eff / (1.0 - ab_t) * z0_pred
                + tf.sqrt(ab_t / ab_prev) * (1.0 - ab_prev) / (1.0 - ab_t) * z_t
            )

            def without_noise():
                """Ramo usato all'ultimo step (t_prev=0): restituisce la sola media, senza rumore stocastico."""
                return mean

            def with_noise():
                """Ramo usato per gli step intermedi: aggiunge rumore gaussiano riproducibile (seed derivato da stateless_fold_in) scalato per la deviazione standard appresa."""
                log_var = get_learned_log_variance_eff(v_pred, b_eff, post_var_eff)
                std = tf.exp(0.5 * log_var)
                noise_seed = tf.random.experimental.stateless_fold_in(seed, index + 1)
                return mean + std * tf.random.stateless_normal(
                    tf.shape(z_t),
                    seed=noise_seed,
                )

            z_next = tf.cond(tf.equal(t_prev_int, 0), without_noise, with_noise)
            z_next = tf.ensure_shape(z_next, [1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS])
            return index + 1, z_next

        _, z_final = tf.while_loop(
            lambda index, _z: index < n_timesteps,
            denoise_step,
            loop_vars=[
                tf.constant(0, dtype=tf.int32),
                tf.ensure_shape(z0, [1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS]),
            ],
            parallel_iterations=1,
        )

        z_denorm = z_final * latent_std_tensor + latent_mean_tensor
        if decode_on_cpu:
            with tf.device("/CPU:0"):
                images = vae_decoder(z_denorm, training=False)
        else:
            images = vae_decoder(z_denorm, training=False)
        return tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)

    return compiled_sampler


def load_latent_stats(latent_stats_path: Path):
    """Carica media e deviazione standard dei latenti VAE salvate durante il training, necessarie per normalizzare/denormalizzare lo spazio latente in cui opera la diffusione."""
    stats = np.load(str(latent_stats_path))
    latent_mean = tf.constant(stats["latent_mean"], dtype=tf.float32)
    latent_std = tf.constant(stats["latent_std"], dtype=tf.float32)
    return latent_mean, latent_std


def load_vae_decoder(model_path: Path):
    """Carica il decoder del VAE già addestrato e lo congela, dato che in fase di generazione serve solo per l'inferenza."""
    decoder = tf.keras.models.load_model(str(model_path), compile=False)
    decoder.trainable = False
    return decoder


def load_ldm_model(model_path: Path):
    """Carica la U-Net della diffusione da un checkpoint salvato e la congela, dato che in fase di generazione serve solo per l'inferenza."""
    model = tf.keras.models.load_model(str(model_path), compile=False)
    model.trainable = False
    return model


def sample_ldm(
    ldm_model,
    vae_decoder,
    schedule: DiffusionSchedule,
    latent_mean,
    latent_std,
    num_images: int = 1,
    label: int = 1,
    num_steps: int = SAMPLE_STEPS,
    guidance_scale: float = CFG_SCALE,
    verbose: bool = False,
):
    """Versione eager (non compilata in grafo) del sampling: genera num_images immagini per la label data eseguendo in sequenza tutti gli step di reverse diffusion con p_sample_ldm e poi decodificando col VAE. Utile per generazioni una tantum o debug, dove make_compiled_sampler non è necessario."""
    z = tf.random.normal((num_images, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS))
    y = tf.fill([num_images], int(label))
    timesteps = sampling_timesteps(num_steps)

    for idx, t in enumerate(timesteps):
        t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
        z = p_sample_ldm(
            ldm_model,
            schedule,
            z,
            int(t),
            int(t_prev),
            y,
            guidance_scale=guidance_scale,
        )
        if verbose and ((idx + 1) % 20 == 0 or idx == 0):
            print(f"  Step {idx + 1}/{len(timesteps)} (t={t}->{t_prev})")
        if (idx + 1) % 25 == 0:
            try:
                tf.test.experimental.sync_devices()
            except Exception:
                pass

    z_denorm = z * latent_std + latent_mean
    images = vae_decoder(z_denorm, training=False)
    return tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)
