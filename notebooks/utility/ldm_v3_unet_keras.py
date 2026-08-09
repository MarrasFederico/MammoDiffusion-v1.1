"""Build the Stable-Diffusion-inspired LDM v3 U-Net in Keras/TensorFlow.

Compared with v2 (``train_ldm.build_ldm_unet``), v3 introduces:

* **Upsample(nearest 2x) + Conv2D 3x3** instead of ``Conv2DTranspose`` to avoid
  the checkerboard artifacts associated with transposed convolutions (Odena et
  al., 2016). Stable Diffusion 1.x/2.x uses the same pattern.
* **Stable-Diffusion-style ResBlocks**: two ``GroupNorm -> SiLU -> Conv3x3``
  sequences with a linearly projected, broadcast time-and-label embedding
  (FiLM-like), instead of adding the embedding only halfway through the block.
* **Downsample = Conv2D stride=2**, unchanged from v2 and repeated here for clarity.
* **Shape compatibility**: inputs and outputs match v2 ``build_ldm_unet``, so
  the training and inference loops require no other changes.

``train_ldm.py`` imports this module for ``--unet-version v3``; a notebook can
also import it directly for a focused trial.
"""
from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

__all__ = [
    "SinusoidalTimeEmbeddingV3",
    "LabelEmbeddingV3",
    "ResBlockV3",
    "SelfAttentionBlockV3",
    "UpsampleConvBlock",
    "DownsampleConv",
    "build_ldm_unet_v3",
]


# ---------------------------------------------------------------------------
# Time and label embeddings (same signature as v2, with SiLU activation)
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class SinusoidalTimeEmbeddingV3(layers.Layer):
    """DDPM-style time embedding: sinusoidal encoding followed by a two-layer SiLU MLP."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        half = self.embed_dim // 2
        freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / float(half))
        self._freqs = tf.constant(freqs.astype("float32"), dtype=tf.float32)
        self.dense1 = layers.Dense(embed_dim * 4)
        self.dense2 = layers.Dense(embed_dim * 4)

    def call(self, t):
        t = tf.cast(t, tf.float32)
        freqs = tf.cast(self._freqs, tf.float32)
        args = t[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        emb = tf.nn.silu(self.dense1(emb))
        emb = self.dense2(emb)
        return emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class LabelEmbeddingV3(layers.Layer):
    """Embed ``num_classes + 1`` labels; the last entry is CFG's null class."""

    def __init__(self, num_classes, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.embedding = layers.Embedding(num_classes + 1, embed_dim * 4)

    def call(self, y):
        return self.embedding(y)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "embed_dim": self.embed_dim})
        return cfg


# ---------------------------------------------------------------------------
# ResBlock stile SD: [GN -> SiLU -> Conv3x3] x 2 + embed FiLM + skip 1x1
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class ResBlockV3(layers.Layer):
    """SD-style ResBlock: two GroupNorm-SiLU-Conv3x3 stages with FiLM-projected embedding."""

    def __init__(self, channels, embed_dim, groups=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.groups = int(groups) if groups is not None else min(32, self.channels)
        self.norm1 = layers.GroupNormalization(groups=self.groups)
        self.conv1 = layers.Conv2D(self.channels, 3, padding="same")
        self.norm2 = layers.GroupNormalization(groups=self.groups)
        self.conv2 = layers.Conv2D(self.channels, 3, padding="same")
        # FiLM-like projection of the embedding into a channel-wise bias for the second convolution.
        self.emb_proj = layers.Dense(self.channels)
        self.skip = layers.Conv2D(self.channels, 1)

    def call(self, x, emb):
        # Main branch
        h = self.norm1(x)
        h = tf.nn.silu(h)
        h = self.conv1(h)
        # Inject the embedding as a bias broadcast over H and W.
        emb_bias = self.emb_proj(tf.nn.silu(emb))
        h = h + tf.reshape(emb_bias, [-1, 1, 1, self.channels])
        h = self.norm2(h)
        h = tf.nn.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "embed_dim": self.embed_dim, "groups": self.groups})
        return cfg


@tf.keras.utils.register_keras_serializable()
class SelfAttentionBlockV3(layers.Layer):
    """Self-attention SD-style con GroupNorm pre-attention e residuo."""

    def __init__(self, channels, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.norm = layers.GroupNormalization(groups=min(32, self.channels))
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels // num_heads,
            value_dim=channels // num_heads,
        )
        self.proj = layers.Dense(self.channels)

    def call(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self.channels
        h = self.norm(x)
        h = tf.reshape(h, [B, H * W, C])
        h = self.attn(h, h)
        h = self.proj(h)
        h = tf.reshape(h, [B, H, W, C])
        return x + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "num_heads": self.num_heads})
        return cfg


# ---------------------------------------------------------------------------
# Up/Down sampling con schema Stable Diffusion
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class UpsampleConvBlock(layers.Layer):
    """Upsample 2x with nearest-neighbor interpolation, then apply a 3x3 Conv2D."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.up = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv = layers.Conv2D(self.channels, 3, padding="same")

    def call(self, x):
        return self.conv(self.up(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels})
        return cfg


@tf.keras.utils.register_keras_serializable()
class DownsampleConv(layers.Layer):
    """SD-style downsampling: stride-2 Conv2D with same padding, explicitly matching v2."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.conv = layers.Conv2D(self.channels, 3, strides=2, padding="same")

    def call(self, x):
        return self.conv(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels})
        return cfg


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_ldm_unet_v3(
    *,
    latent_size: int,
    latent_channels: int,
    model_channels: int,
    embed_dim: int,
    num_classes: int,
    num_attention_heads: int = 4,
):
    """Build the LDM v3 U-Net with the same input/output signature as v2.

    Args:
        latent_size: Latent spatial side, e.g. 64 for a 512x512 input and 8x VAE.
        latent_channels: Number of latent channels, e.g. 4 for the SD VAE.
        model_channels: ``MODEL_CHANNELS`` from train_ldm; doubled to match v2.
        embed_dim: Time/label embedding size, expanded 4x by the embedding layers.
        num_classes: Conditioning classes, excluding CFG's null class.
        num_attention_heads: Attention heads at low-resolution feature maps.

    Returns:
        A ``tf.keras.Model`` with inputs ``lat_input``, ``t_input``, and
        ``y_input`` and one v2-compatible ``latent_channels * 2`` output for
        epsilon plus learned variance.
    """
    C = int(model_channels) * 2  # Match v2: C = MODEL_CHANNELS * 2.

    lat_input = layers.Input(shape=(latent_size, latent_size, latent_channels), name="lat_input")
    t_input = layers.Input(shape=(), dtype=tf.int32, name="t_input")
    y_input = layers.Input(shape=(), dtype=tf.int32, name="y_input")

    time_emb = SinusoidalTimeEmbeddingV3(embed_dim)(t_input)
    label_emb = LabelEmbeddingV3(num_classes, embed_dim)(y_input)
    emb = time_emb + label_emb

    # Stem
    x = layers.Conv2D(C, 3, padding="same")(lat_input)

    # Encoder stage 1
    x1 = ResBlockV3(C, embed_dim)(x, emb)
    x1 = ResBlockV3(C, embed_dim)(x1, emb)
    p1 = DownsampleConv(C)(x1)

    # Encoder stage 2
    x2 = ResBlockV3(C * 2, embed_dim)(p1, emb)
    x2 = ResBlockV3(C * 2, embed_dim)(x2, emb)
    p2 = DownsampleConv(C * 2)(x2)

    # Encoder stage 3 (con attention)
    x3 = ResBlockV3(C * 4, embed_dim)(p2, emb)
    x3 = SelfAttentionBlockV3(C * 4, num_heads=num_attention_heads)(x3)
    x3 = ResBlockV3(C * 4, embed_dim)(x3, emb)
    p3 = DownsampleConv(C * 4)(x3)

    # Bottleneck (2x ResBlock + 2x SelfAttn)
    b = ResBlockV3(C * 4, embed_dim)(p3, emb)
    b = SelfAttentionBlockV3(C * 4, num_heads=num_attention_heads)(b)
    b = ResBlockV3(C * 4, embed_dim)(b, emb)
    b = SelfAttentionBlockV3(C * 4, num_heads=num_attention_heads)(b)

    # Decoder stage 3 (Upsample+Conv instead of Conv2DTranspose).
    u3 = UpsampleConvBlock(C * 4)(b)
    u3 = layers.Concatenate()([u3, x3])
    u3 = ResBlockV3(C * 4, embed_dim)(u3, emb)
    u3 = SelfAttentionBlockV3(C * 4, num_heads=num_attention_heads)(u3)
    u3 = ResBlockV3(C * 4, embed_dim)(u3, emb)

    # Decoder stage 2
    u2 = UpsampleConvBlock(C * 2)(u3)
    u2 = layers.Concatenate()([u2, x2])
    u2 = ResBlockV3(C * 2, embed_dim)(u2, emb)
    u2 = ResBlockV3(C * 2, embed_dim)(u2, emb)

    # Decoder stage 1
    u1 = UpsampleConvBlock(C)(u2)
    u1 = layers.Concatenate()([u1, x1])
    u1 = ResBlockV3(C, embed_dim)(u1, emb)
    u1 = ResBlockV3(C, embed_dim)(u1, emb)

    # Head: GroupNorm + SiLU + Conv1x1 projected to ``latent_channels * 2``.
    # Activation("swish") is required instead of Lambda(tf.nn.silu, ...): this
    # Keras version cannot serialize a Lambda around a raw TensorFlow function.
    # Swish and SiLU are the same function: x * sigmoid(x).
    out = layers.GroupNormalization(groups=min(32, C))(u1)
    out = layers.Activation("swish", name="head_silu")(out)
    out = layers.Conv2D(latent_channels * 2, 3, padding="same")(out)

    return tf.keras.Model([lat_input, t_input, y_input], out, name=f"ldm_unet_v3_{latent_size}px")


# ---------------------------------------------------------------------------
# Diffusion training helpers (v-prediction target and Min-SNR weighting)
# ---------------------------------------------------------------------------

def make_v_target(x0, noise, t, sqrt_alpha_bars_arr, sqrt_one_minus_alpha_bars_arr):
    """Target di v-prediction (Salimans & Ho 2022): v = sqrt(ab)*eps - sqrt(1-ab)*x0."""
    def _extract(values, t):
        vals = tf.gather(values, t)
        return tf.reshape(vals, [-1, 1, 1, 1])
    sqrt_ab = _extract(sqrt_alpha_bars_arr, t)
    sqrt_omab = _extract(sqrt_one_minus_alpha_bars_arr, t)
    return sqrt_ab * noise - sqrt_omab * x0


def min_snr_weight(t, alpha_bars_arr, gamma: float = 5.0, parameterization: str = "eps"):
    """Return per-sample Min-SNR-gamma weights (Hang et al., 2023).

    For SNR(t) = ab_t / (1 - ab_t):

    * ``parameterization="eps"`` uses
      w(t) = min(SNR(t), gamma) / SNR(t), the paper's original formula;
    * ``parameterization="v"`` uses
      w(t) = min(SNR(t), gamma) / (SNR(t) + 1), because v-prediction loss is
      already implicitly scaled by SNR+1 relative to epsilon loss. This matches
      Hugging Face Diffusers' ``snr_gamma`` convention.

    The weighting de-emphasizes easy, high-SNR timesteps relative to a uniform
    loss. The returned tensor has shape ``(batch, 1, 1, 1)`` for direct
    multiplication with the element-wise loss.
    """
    ab = tf.gather(alpha_bars_arr, t)
    ab = tf.reshape(ab, [-1, 1, 1, 1])
    snr = ab / tf.maximum(1.0 - ab, 1e-8)
    min_snr = tf.minimum(snr, gamma)
    if parameterization == "v":
        w = min_snr / tf.maximum(snr + 1.0, 1e-8)
    elif parameterization == "eps":
        w = min_snr / tf.maximum(snr, 1e-8)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
    return w
