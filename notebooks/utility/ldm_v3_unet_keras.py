"""LDM v3 U-Net builder (Keras/TF), design ispirato a Stable Diffusion.

Rispetto alla versione v2 (`train_ldm_v2.build_ldm_unet`) introduce:

* **Upsample(nearest 2x) + Conv2D 3x3** al posto di `Conv2DTranspose`, per eliminare
  i noti *checkerboard artifacts* di ConvTranspose (Odena et al. 2016). E' lo stesso
  schema usato dalla U-Net di Stable Diffusion 1.x/2.x.
* **ResBlock in stile SD**: `GroupNorm -> SiLU -> Conv3x3` due volte, con iniezione
  dell'embedding tempo+label via proiezione lineare seguita da broadcast (FiLM-like),
  invece della somma solo a meta' blocco. Attivazione SiLU al posto di LeakyReLU.
* **Downsample = Conv2D stride=2** (invariato rispetto a v2, ma mantenuto qui per completezza).
* Compatibilita' shape: la firma degli input e degli output resta identica a `build_ldm_unet`
  di v2, cosi' l'intera pipeline di training/inference puo' riutilizzare la stessa
  loop senza altre modifiche.

Il modulo puo' essere importato da `train_ldm_v2.py` quando l'utente passa
`--unet-version v3`, oppure caricato da un notebook per uno smoke test dedicato.
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
# Embedding tempo + label (identici come firma a v2, solo attivazione SiLU)
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class SinusoidalTimeEmbeddingV3(layers.Layer):
    """Time embedding stile DDPM: encoding sinusoidale seguito da MLP (2 dense + SiLU)."""

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
    """Label embedding con `num_classes+1` voci (l'ultima e' la classe nulla per CFG)."""

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
    """ResBlock SD-style: 2x (GroupNorm + SiLU + Conv3x3) con embedding proiettato via FiLM."""

    def __init__(self, channels, embed_dim, groups=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.groups = int(groups) if groups is not None else min(32, self.channels)
        self.norm1 = layers.GroupNormalization(groups=self.groups)
        self.conv1 = layers.Conv2D(self.channels, 3, padding="same")
        self.norm2 = layers.GroupNormalization(groups=self.groups)
        self.conv2 = layers.Conv2D(self.channels, 3, padding="same")
        # FiLM-like: proiezione dell'embedding in shift (bias) per i canali della seconda conv.
        self.emb_proj = layers.Dense(self.channels)
        self.skip = layers.Conv2D(self.channels, 1)

    def call(self, x, emb):
        # Ramo principale
        h = self.norm1(x)
        h = tf.nn.silu(h)
        h = self.conv1(h)
        # Iniezione embedding (bias broadcast su H, W)
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
    """Upsample nearest x2 seguito da Conv2D 3x3. Sostituisce Conv2DTranspose."""

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
    """Downsample stile SD: Conv2D stride=2 con padding=same. Identico a v2 (esplicitato)."""

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
# Builder principale
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
    """Costruisce la U-Net LDM v3 con la stessa firma input/output di v2.

    Argomenti
    ---------
    latent_size: int
        Lato spaziale dei latenti (es. 64 per input 512x512 con VAE 8x).
    latent_channels: int
        Numero di canali dei latenti (es. 4 per il VAE di SD).
    model_channels: int
        `MODEL_CHANNELS` come definito in train_ldm_v2 (viene raddoppiato per matchare v2).
    embed_dim: int
        Dimensione embedding tempo/label (moltiplicata x4 dagli embedding layer, come v2).
    num_classes: int
        Numero di classi condizionanti (senza contare la classe nulla per CFG).
    num_attention_heads: int
        Numero di teste dell'attention nelle feature map basse.

    Output
    ------
    tf.keras.Model con 3 input (`lat_input`, `t_input`, `y_input`) e 1 output della
    stessa forma di quello di v2 (`latent_channels * 2` per supportare eps+var, come v2).
    """
    C = int(model_channels) * 2  # coerente con v2: C = MODEL_CHANNELS * 2

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

    # Decoder stage 3 (Upsample+Conv al posto di Conv2DTranspose)
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

    # Head: GroupNorm + SiLU + Conv1x1 (proiezione ai `latent_channels * 2`)
    # Nota: usare Activation("swish") invece di Lambda(tf.nn.silu, ...) e' necessario
    # perche' Lambda con una funzione TF raw non e' serializzabile da model.save()
    # su questa versione di Keras (fallisce con "Cannot serialize object ... Signature").
    # swish e silu sono la stessa funzione (x * sigmoid(x)).
    out = layers.GroupNormalization(groups=min(32, C))(u1)
    out = layers.Activation("swish", name="head_silu")(out)
    out = layers.Conv2D(latent_channels * 2, 3, padding="same")(out)

    return tf.keras.Model([lat_input, t_input, y_input], out, name=f"ldm_unet_v3_{latent_size}px")


# ---------------------------------------------------------------------------
# Diffusion training helpers (v-prediction target e min-SNR weighting)
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
    """Peso per campione secondo Min-SNR-γ (Hang et al. 2023).

    Con SNR(t) = ab_t / (1 - ab_t):
    * `parameterization="eps"`: w(t) = min(SNR(t), gamma) / SNR(t) (formula originale del paper);
    * `parameterization="v"`: w(t) = min(SNR(t), gamma) / (SNR(t) + 1), correzione necessaria
      perche' la loss v-prediction e' gia' implicitamente scalata da (SNR+1) rispetto alla
      loss epsilon (stessa convenzione usata da HuggingFace diffusers per `snr_gamma`).

    Riduce l'enfasi sui timestep facili (SNR alto) rispetto alla loss uniforme.
    Restituisce un tensore di shape (batch, 1, 1, 1) da moltiplicare direttamente
    alla loss per-elemento.
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
