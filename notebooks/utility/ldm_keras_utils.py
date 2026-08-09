from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Imported for its side effect: register v3 U-Net custom layers
# (SinusoidalTimeEmbeddingV3, ResBlockV3, etc.) with Keras so load_model works
# on v3 checkpoints without requiring each caller to import them explicitly.
# Callers of load_ldm_model must know which U-Net version they are loading.
import ldm_v3_unet_keras  # noqa: F401

LATENT_SIZE = 64
LATENT_CHANNELS = 4
NUM_CLASSES = 2
NUM_DIFF_STEPS = 1000
CFG_SCALE = 3.0
SAMPLE_STEPS = 100
CLASS_NAMES = {0: "Negative (non-cancer label)", 1: "Positive (cancer label)"}


@dataclass(frozen=True)
class DiffusionSchedule:
    """Immutable diffusion-schedule coefficients, precomputed once and reused at every step."""

    betas: tf.Tensor
    alphas: tf.Tensor
    alpha_bars: tf.Tensor
    sqrt_alpha_bars: tf.Tensor
    sqrt_one_minus_alpha_bars: tf.Tensor
    alpha_bars_prev: tf.Tensor
    posterior_variance: tf.Tensor
    posterior_log_variance: tf.Tensor


def configure_tensorflow(seed: int = 42, allow_gpu_memory_growth: bool = False) -> None:
    """Seed TensorFlow and NumPy for reproducibility and report the detected GPU."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    gpus = tf.config.list_physical_devices("GPU")

    if allow_gpu_memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Cannot enable TensorFlow GPU memory growth because the GPU is already initialized."
                ) from exc

    print("TF version:", tf.__version__)
    print("Available GPUs:", gpus)


_VRAM_LAST_CURRENT: dict[str, int] = {}


def vram_gb(label: str = "VRAM", device: str = "GPU:0") -> None:
    """Print current/peak GPU memory and change since the previous call."""
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
        print(f"[{label}] VRAM unavailable on {device}: {exc}")


@tf.keras.utils.register_keras_serializable()
class SinusoidalTimeEmbedding(layers.Layer):
    """Map diffusion timestep t to a sinusoidal-plus-MLP embedding for U-Net ResBlocks."""

    def __init__(self, embed_dim, **kwargs):
        """Store the embedding size and create the two-layer sinusoidal projection MLP."""
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        half = embed_dim // 2
        frequencies = np.exp(
            -math.log(10000.0) * np.arange(half, dtype=np.float32) / float(half)
        )
        self._frequencies = tf.constant(frequencies, dtype=tf.float32)
        self.dense1 = layers.Dense(embed_dim * 4, activation="relu")
        self.dense2 = layers.Dense(embed_dim * 4)

    def call(self, t):
        """Encode each timestep with Transformer-style sine/cosine pairs and apply the MLP."""
        t = tf.cast(t, tf.float32)
        freqs = tf.cast(self._frequencies, tf.float32)
        args = t[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return self.dense2(self.dense1(emb))

    def get_config(self):
        """Add embed_dim to the layer config so saved models can reconstruct it."""
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class LabelEmbedding(layers.Layer):
    """Map class labels, including CFG's unconditional slot, to time-embedding-sized vectors."""

    def __init__(self, num_classes, embed_dim, **kwargs):
        """Create an embedding table with one extra row for unconditional training and sampling."""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = layers.Embedding(num_classes + 1, embed_dim * 4)

    def call(self, y):
        """Return embeddings for labels y."""
        return self.embedding(y)

    def get_config(self):
        """Add num_classes and embed_dim to the config for saved-model reconstruction."""
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class ResBlock(layers.Layer):
    """U-Net residual block that combines spatial features with time/label embeddings."""

    def __init__(self, channels, embed_dim, **kwargs):
        """Create GroupNorm convolutions, the embedding projection, and a 1x1 channel skip."""
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
        """Apply norm/convolution, inject the broadcast embedding, and add the skip connection."""
        h = self.conv1(self.act(self.norm1(x)))
        emb_out = tf.reshape(self.emb_proj(self.act(emb)), [-1, 1, 1, self.channels])
        h = self.conv2(self.act(self.norm2(h + emb_out)))
        return h + self.skip(x)

    def get_config(self):
        """Add channels and embed_dim to the config for saved-model reconstruction."""
        config = super().get_config()
        config.update({"channels": self.channels, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    """Spatial self-attention block for long-range dependencies at deep U-Net levels."""

    def __init__(self, channels, num_heads=4, **kwargs):
        """Create GroupNorm, channel-derived multi-head attention, and the final projection."""
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
        """Flatten HxW to tokens, apply self-attention, restore the shape, and add the skip."""
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
        """Add channels and num_heads to the config for saved-model reconstruction."""
        config = super().get_config()
        config.update({"channels": self.channels, "num_heads": self.num_heads})
        return config


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Compute closed-form KL divergence between diagonal Gaussians for the VLB term."""
    return 0.5 * (
        logvar2
        - logvar1
        + tf.exp(logvar1 - logvar2)
        + tf.square(mean1 - mean2) * tf.exp(-logvar2)
        - 1.0
    )


def build_schedule(num_steps: int = NUM_DIFF_STEPS, s: float = 0.008) -> DiffusionSchedule:
    """Build the Nichol-Dhariwal cosine schedule and all derived coefficients."""
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
    """Gather each batch element's timestep coefficient and reshape it for broadcasting."""
    batch_size = tf.shape(t)[0]
    out = tf.gather(values, t)
    return tf.reshape(out, [batch_size, 1, 1, 1])


def predict_epsilon_from_model_output(model_output, sample, alpha_bar_t, parameterization: str = "eps"):
    """Convert raw U-Net output to epsilon independently of training parameterization.

    This keeps the epsilon-based DDPM/DDIM sampler unchanged.

    * ``parameterization="eps"``: the network already predicts epsilon.
    * ``parameterization="v"`` (Salimans & Ho, 2022): the network predicts
      ``v = sqrt(ab_t)*eps - sqrt(1-ab_t)*x0``, giving
      ``eps = sqrt(ab_t)*v + sqrt(1-ab_t)*sample``.

    ``alpha_bar_t`` must be broadcastable to ``sample`` (for example shape
    ``(B, 1, 1, 1)``; see ``extract``). ``sample`` is the current noisy latent z_t.
    """
    if parameterization == "eps":
        return model_output
    if parameterization == "v":
        sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
        return sqrt_alpha_bar_t * model_output + sqrt_one_minus_alpha_bar_t * sample
    raise ValueError(f"Unknown parameterization: {parameterization}")


def get_learned_log_variance(v, t, schedule: DiffusionSchedule):
    """Interpolate posterior log variance between beta_tilde and beta using learned v."""
    shape = tf.shape(v)
    log_beta_t = tf.math.log(extract(schedule.betas, t, shape))
    log_beta_tilde_t = tf.math.log(
        extract(schedule.posterior_variance + 1e-8, t, shape)
    )
    v_sigmoid = tf.sigmoid(tf.clip_by_value(v, -8.0, 8.0))
    log_var = v_sigmoid * log_beta_t + (1.0 - v_sigmoid) * log_beta_tilde_t
    return tf.clip_by_value(log_var, -20.0, 2.0)


def get_learned_log_variance_eff(v, beta_eff, posterior_variance_eff):
    """Use precomputed effective beta/posterior variance for a DDIM-like multi-step jump."""
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
    parameterization: str = "eps",
):
    """Run one CFG reverse-diffusion step from t to t_prev in a single batched call."""
    batch_size = tf.shape(z_t)[0]
    t_batch = tf.fill([batch_size], t_int)
    t_prev = tf.fill([batch_size], t_prev_int)
    y_uncond = tf.fill([batch_size], NUM_CLASSES)

    z_in = tf.concat([z_t, z_t], axis=0)
    t_in = tf.concat([t_batch, t_batch], axis=0)
    y_in = tf.concat([y_batch, y_uncond], axis=0)

    out_combined = ldm_model([z_in, t_in, y_in], training=False)
    raw_combined, v_combined = tf.split(out_combined, 2, axis=-1)
    raw_cond, raw_uncond = tf.split(raw_combined, 2, axis=0)
    v_cond, _ = tf.split(v_combined, 2, axis=0)

    raw_pred = raw_uncond + guidance_scale * (raw_cond - raw_uncond)
    v_pred = v_cond

    shape = tf.shape(z_t)
    ab_t = extract(schedule.alpha_bars, t_batch, shape)
    ab_prev = extract(schedule.alpha_bars, t_prev, shape)
    sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t_batch, shape)

    eps_pred = predict_epsilon_from_model_output(raw_pred, z_t, ab_t, parameterization)
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
    """Uniformly subsample training timesteps into a descending accelerated-sampling sequence."""
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
    parameterization: str = "eps",
):
    """Compile an end-to-end denoising and VAE-decoding sampler for one seed and label."""
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
        """Generate one image from seeded latent noise, reverse diffusion, and VAE decoding."""
        print(
            "[make_compiled_sampler] tracing sampling graph "
            f"num_steps={num_steps} guidance_scale={guidance_scale:g} "
            f"decode_on_cpu={decode_on_cpu} parameterization={parameterization}"
        )
        seed = tf.ensure_shape(tf.cast(seed, tf.int32), [2])
        z0 = tf.random.stateless_normal(
            (1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS),
            seed=seed,
        )

        def denoise_step(index, z_t):
            """Run one graph-friendly CFG reverse step and advance the timestep index."""
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
            raw_combined, v_combined = tf.split(out_combined, 2, axis=-1)
            raw_cond, raw_uncond = tf.split(raw_combined, 2, axis=0)
            v_cond, _ = tf.split(v_combined, 2, axis=0)

            raw_pred = raw_uncond + guidance_scale_tensor * (raw_cond - raw_uncond)
            v_pred = v_cond

            shape = tf.shape(z_t)
            ab_t = extract(schedule.alpha_bars, t_batch, shape)
            ab_prev = extract(schedule.alpha_bars, t_prev, shape)
            sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t_batch, shape)

            eps_pred = predict_epsilon_from_model_output(raw_pred, z_t, ab_t, parameterization)
            z0_pred = (z_t - sqrt_omab * eps_pred) / tf.sqrt(ab_t)
            b_eff = 1.0 - ab_t / ab_prev
            post_var_eff = b_eff * (1.0 - ab_prev) / (1.0 - ab_t)
            mean = (
                tf.sqrt(ab_prev) * b_eff / (1.0 - ab_t) * z0_pred
                + tf.sqrt(ab_t / ab_prev) * (1.0 - ab_prev) / (1.0 - ab_t) * z_t
            )

            def without_noise():
                """Return only the mean at the final step, without stochastic noise."""
                return mean

            def with_noise():
                """Add reproducible Gaussian noise scaled by the learned standard deviation."""
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


def make_compiled_latent_sampler(
    ldm_model,
    schedule: DiffusionSchedule,
    latent_mean,
    latent_std,
    num_steps: int = SAMPLE_STEPS,
    guidance_scale: float = CFG_SCALE,
    parameterization: str = "eps",
):
    """Build a graph-friendly sampler that returns the final denormalized latent."""
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
    def compiled_latent_sampler(label, seed):
        print(
            "[make_compiled_latent_sampler] tracing sampling graph "
            f"num_steps={num_steps} guidance_scale={guidance_scale:g} "
            f"parameterization={parameterization}"
        )
        seed = tf.ensure_shape(tf.cast(seed, tf.int32), [2])
        z0 = tf.random.stateless_normal(
            (1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS),
            seed=seed,
        )

        def denoise_step(index, z_t):
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
            raw_combined, v_combined = tf.split(out_combined, 2, axis=-1)
            raw_cond, raw_uncond = tf.split(raw_combined, 2, axis=0)
            v_cond, _ = tf.split(v_combined, 2, axis=0)

            raw_pred = raw_uncond + guidance_scale_tensor * (raw_cond - raw_uncond)
            shape = tf.shape(z_t)
            ab_t = extract(schedule.alpha_bars, t_batch, shape)
            ab_prev = extract(schedule.alpha_bars, t_prev, shape)
            sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t_batch, shape)

            eps_pred = predict_epsilon_from_model_output(raw_pred, z_t, ab_t, parameterization)
            z0_pred = (z_t - sqrt_omab * eps_pred) / tf.sqrt(ab_t)
            b_eff = 1.0 - ab_t / ab_prev
            post_var_eff = b_eff * (1.0 - ab_prev) / (1.0 - ab_t)
            mean = (
                tf.sqrt(ab_prev) * b_eff / (1.0 - ab_t) * z0_pred
                + tf.sqrt(ab_t / ab_prev) * (1.0 - ab_prev) / (1.0 - ab_t) * z_t
            )

            def without_noise():
                return mean

            def with_noise():
                log_var = get_learned_log_variance_eff(v_cond, b_eff, post_var_eff)
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
        return z_final * latent_std_tensor + latent_mean_tensor

    return compiled_latent_sampler


def load_latent_stats(latent_stats_path: Path):
    """Load VAE latent mean and standard deviation for diffusion-space normalization."""
    stats = np.load(str(latent_stats_path))
    latent_mean = tf.constant(stats["latent_mean"], dtype=tf.float32)
    latent_std = tf.constant(stats["latent_std"], dtype=tf.float32)
    return latent_mean, latent_std


def load_vae_decoder(model_path: Path):
    """Load and freeze the trained VAE decoder for generation-only inference."""
    decoder = tf.keras.models.load_model(str(model_path), compile=False)
    decoder.trainable = False
    return decoder


def load_ldm_model(model_path: Path):
    """Load and freeze the diffusion U-Net checkpoint for generation-only inference."""
    model = tf.keras.models.load_model(str(model_path), compile=False)
    model.trainable = False
    return model
