"""Utility condivise per i classificatori Mammo-FM (EfficientNet-B5, PyTorch) fine-tuned.

Sostituisce `mammodino_utils.py` (ora archiviato in `_deprecated_mammodino/`): i checkpoint
MammoDINO (GE HealthCare, arXiv:2510.11883) non risultano pubblicamente disponibili, mentre
**Mammo-FM** (batmanlab, arXiv:2512.00198) e' un foundation model mammografico con checkpoint
pubblici su Hugging Face (`batmanLab/Mammo-FM`). Segue lo stesso stile dei moduli helper del
progetto (`maxvit_utils.py`, `medfoundation_utils.py`): le callback in stile Keras
(EarlyStopping, ModelCheckpoint, CSVLogger) sono riusate da `maxvit_utils`.

Architettura verificata direttamente sul checkpoint reale (`Mammo-FM_BatmanlabTrained_CLIP.tar`,
scaricato e ispezionato durante lo sviluppo di questo helper):

- Il checkpoint e' un dict torch (`torch.load`) con chiavi `config` e `model` (stesso formato
  di `batmanlab/Mammo-CLIP`, di cui Mammo-FM e' l'evoluzione "piu' grande e forte"); il campo
  `config["model"]["image_encoder"]` dichiara `{"source": "cnn", "name":
  "tf_efficientnet_b5_ns-detect", "model_type": "cnn"}`.
- L'encoder immagine e' un EfficientNet-B5 **custom** (stessa implementazione, quasi verbatim,
  del pacchetto pip `efficientnet_pytorch` di lukemelas, con attributi `_conv_stem`, `_blocks`,
  `_conv_head`, ecc.), con **input a 3 canali** (`_conv_stem.weight` ha shape (48, 3, 3, 3)):
  le mammografie grayscale vanno quindi replicate su RGB, NON usate a 1 canale.
- I pesi dell'encoder immagine sono salvati con prefisso `"image_encoder."` nel CLIP state dict
  completo (che include anche `text_encoder.*`, `image_projection.*`, `text_projection.*`,
  `logit_scale`, irrilevanti per la classificazione); il layer `_fc` finale non e' incluso nel
  checkpoint (Mammo-CLIP/Mammo-FM lo lascia non addestrato: si estraggono solo le feature
  pooled via global average pooling, `out_dim=2048` per B5).
- Deserializzare il checkpoint richiede il pacchetto `omegaconf` (il campo `config` e' salvato
  con un oggetto Hydra/OmegaConf); costruire l'encoder richiede il pacchetto `efficientnet_pytorch`.
  Entrambi vengono importati in modo robusto con un messaggio chiaro se mancanti.
- Normalizzazione ufficiale (da `configs/pre_train_b5_clip.yaml` del repo Mammo-CLIP):
  media=0.3089279, std=0.25053555408335154 (scalari, non statistiche ImageNet), applicate dopo
  una normalizzazione min-max per immagine in [0,1] (`img -= img.min(); img /= img.max()`,
  come in `ImageClassificationDataset` del repo ufficiale).
- Risoluzione nativa di pre-training: 1520x912 (non quadrata). Questo progetto usa 512x512
  quadrati per coerenza con tutti gli altri classificatori (MaxViT-512, RAD-DINO, ecc.):
  EfficientNet e' interamente convoluzionale (global average pooling finale), quindi funziona
  correttamente anche a 512x512, sebbene le prestazioni assolute possano differire da quelle
  riportate nel paper alla risoluzione nativa. Scelta documentata, non un bug silenzioso.
- Il preprocessing ufficiale di pre-training include anche CLAHE (equalizzazione adattiva del
  contrasto, vedi `configs/transform/clahe.yaml`): disattivato di default in questo progetto
  (`USE_CLAHE_PREPROCESSING=False`) per coerenza con gli altri classificatori del progetto, che
  non la usano. Puo' essere abilitato per un esperimento dedicato futuro.

**Licenza dei pesi Mammo-FM**: rilasciati con una "Custom Academic License for Model Weights"
(non commerciale, solo ricerca accademica; **nessun uso clinico/diagnostico**; nessuna
ridistribuzione dei pesi). Questo helper scarica il checkpoint nella cache locale di
Hugging Face (`~/.cache/huggingface/hub`, fuori dal repository) e non lo copia mai altrove:
non salvare ne' ridistribuire i pesi nel repository del progetto.

Nessun fallback silenzioso: se il checkpoint Mammo-FM non e' configurato, non e' raggiungibile,
o non e' compatibile con l'architettura attesa, questo modulo solleva `MammoFMConfigError` e
NON sostituisce mai Mammo-FM con un backbone generico (DINOv2/RAD-DINO/ImageNet).
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from maxvit_utils import (  # noqa: F401  (re-esportati per comodita' dei notebook)
    BinaryFocalLoss,
    CSVLogger,
    EarlyStopping,
    History,
    ModelCheckpoint,
    bootstrap_balanced,
    compute_pos_weight,
    count_trainable_params,
    optimal_threshold_youden,
    refreeze_batchnorm,
)

# ---------------------------------------------------------------------------
# Configurazione di default
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO = "batmanLab/Mammo-FM"
DEFAULT_CHECKPOINT_NAME = "Mammo-FM_BatmanlabTrained_CLIP.tar"
DEFAULT_IMG_SIZE = 512  # progetto: 512x512 (coerente con MaxViT-512/RAD-DINO). Nativo Mammo-FM: 1520x912.
DEFAULT_MAMMOFM_MEAN = 0.3089279  # statistiche ufficiali di pre-training Mammo-CLIP/Mammo-FM (scalare, non ImageNet)
DEFAULT_MAMMOFM_STD = 0.25053555408335154

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_LABEL_RE = re.compile(r"_label([01])")

# Encoder immagine CNN supportati dai checkpoint Mammo-FM/Mammo-CLIP ufficiali: mappa il nome
# dichiarato in config["model"]["image_encoder"]["name"] sull'architettura equivalente del
# pacchetto `efficientnet_pytorch` e sulla dimensione delle feature pooled (out_dim), verificate
# scaricando e ispezionando il checkpoint reale `Mammo-FM_BatmanlabTrained_CLIP.tar`.
_SUPPORTED_CNN_ENCODERS = {
    "tf_efficientnet_b5_ns-detect": ("efficientnet-b5", 2048),
    "tf_efficientnetv2-detect": ("efficientnet-b2", 1408),
}

# Prefissi comuni con cui i CLIP checkpoint (Mammo-FM/Mammo-CLIP e varianti) annidano i tensori
# dell'image encoder nello state dict completo (che include anche text encoder e projection
# head). Rimossi iterativamente (non in un unico passaggio) perche' possono comparire annidati
# in qualsiasi ordine/combinazione, es. "module.image_encoder._conv_stem..." -> "_conv_stem...".
_CHECKPOINT_PREFIX_CANDIDATES = (
    "module.", "model.", "backbone.", "encoder.", "visual.",
    "image_encoder.", "clip.", "student.", "teacher.", "net.",
)


def _strip_known_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _CHECKPOINT_PREFIX_CANDIDATES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


class MammoFMConfigError(RuntimeError):
    """Sollevato quando Mammo-FM non e' configurato o i pesi non sono disponibili/compatibili.

    Deliberatamente NON viene mai gestito con un fallback silenzioso su un backbone generico
    (DINOv2/RAD-DINO/ImageNet): il notebook deve fermarsi e l'utente deve correggere la
    configurazione (MAMMOFM_HF_REPO/MAMMOFM_CHECKPOINT_NAME o MAMMOFM_LOCAL_CHECKPOINT_PATH).
    """


# ---------------------------------------------------------------------------
# Modello: encoder immagine EfficientNet-B5 (Mammo-FM/Mammo-CLIP) + testa lineare
# ---------------------------------------------------------------------------

class MammoFMImageEncoder(nn.Module):
    """Wrapper minimale attorno a `efficientnet_pytorch.EfficientNet` per estrarne le feature
    pooled (global average pooling), coerente con come il checkpoint CLIP ufficiale Mammo-FM
    usa l'encoder immagine (vedi `breastclip/model/modules/efficientnet_custom.py` del repo
    batmanlab/Mammo-CLIP: stessa architettura, stessi nomi di parametro `_conv_stem`, `_blocks`,
    `_conv_head`, ecc.). Il layer `_fc` finale non viene mai usato: il checkpoint ufficiale non
    lo include (si allena solo per la fase di proiezione CLIP), quindi si estraggono solo le
    feature pooled prima del classificatore.
    """

    def __init__(self, arch_name: str, out_dim: int):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as exc:
            raise MammoFMConfigError(
                "Il pacchetto 'efficientnet_pytorch' non e' installato: e' necessario per "
                "costruire l'encoder immagine EfficientNet usato dai checkpoint Mammo-FM "
                "ufficiali (stessa implementazione custom di batmanlab/Mammo-CLIP). "
                "Installa con: pip install efficientnet_pytorch"
            ) from exc
        self.model = EfficientNet.from_name(arch_name, num_classes=1)  # num_classes ignorato: _fc non e' usato
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_features = self.model.extract_features(x)
        return F.adaptive_avg_pool2d(raw_features, 1).flatten(1)


class MammoFMClassifier(nn.Module):
    """Encoder immagine Mammo-FM (EfficientNet-B5) + testa lineare per classificazione binaria."""

    def __init__(self, image_encoder: MammoFMImageEncoder, hidden_size: int,
                 num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.image_encoder = image_encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(x)
        return self.classifier(self.dropout(features))


def _resolve_checkpoint_path(hf_repo: Optional[str], checkpoint_name: Optional[str],
                              use_local_checkpoint: bool, local_checkpoint_path: Optional[str]) -> str:
    if use_local_checkpoint:
        if not local_checkpoint_path or not Path(local_checkpoint_path).is_file():
            raise MammoFMConfigError(
                f"USE_LOCAL_CHECKPOINT=True ma il checkpoint '{local_checkpoint_path}' non "
                "esiste. Imposta MAMMOFM_LOCAL_CHECKPOINT_PATH su un file .tar valido con i "
                "pesi Mammo-FM (es. scaricato manualmente da "
                "https://huggingface.co/batmanLab/Mammo-FM)."
            )
        return str(local_checkpoint_path)

    if not hf_repo or not checkpoint_name:
        raise MammoFMConfigError(
            "Mammo-FM non configurato: imposta MAMMOFM_HF_REPO e MAMMOFM_CHECKPOINT_NAME "
            "(repository/checkpoint Hugging Face ufficiali batmanLab/Mammo-FM), oppure "
            "USE_LOCAL_CHECKPOINT=True con MAMMOFM_LOCAL_CHECKPOINT_PATH valorizzato. Questo "
            "notebook non usera' mai un backbone generico come sostituto silenzioso."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MammoFMConfigError(
            "Il pacchetto 'huggingface_hub' non e' installato: e' necessario per scaricare il "
            "checkpoint Mammo-FM. Installa con: pip install huggingface_hub"
        ) from exc
    try:
        return hf_hub_download(repo_id=hf_repo, filename=checkpoint_name)
    except Exception as exc:
        raise MammoFMConfigError(
            f"Impossibile scaricare il checkpoint Mammo-FM '{checkpoint_name}' dal repository "
            f"Hugging Face '{hf_repo}': {exc}\n"
            "Verifica la connessione di rete e il nome del repository/file, oppure imposta "
            "USE_LOCAL_CHECKPOINT=True con un checkpoint gia' scaricato manualmente. Nessun "
            "fallback automatico su un modello generico."
        ) from exc


def _load_raw_checkpoint(ckpt_path: str) -> dict:
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        raise MammoFMConfigError(
            f"Impossibile deserializzare il checkpoint Mammo-FM '{ckpt_path}': modulo mancante "
            f"({exc}). La configurazione del checkpoint e' salvata con omegaconf/Hydra: "
            "installa con 'pip install omegaconf' e riprova."
        ) from exc
    if not isinstance(raw, dict) or "model" not in raw or "config" not in raw:
        raise MammoFMConfigError(
            f"Formato checkpoint non riconosciuto in '{ckpt_path}': attese le chiavi 'model' e "
            "'config' (formato ufficiale Mammo-CLIP/Mammo-FM: torch.save di un dict con "
            "{'model': ..., 'config': ..., ...}). Il notebook non tenta un caricamento "
            "alternativo/generico."
        )
    return raw


def _resolve_encoder_arch(enc_config: dict) -> tuple:
    name = str(enc_config.get("name", "")).strip()
    source = str(enc_config.get("source", "")).strip().lower()
    if source != "cnn" or name not in _SUPPORTED_CNN_ENCODERS:
        raise MammoFMConfigError(
            f"Encoder immagine del checkpoint Mammo-FM non supportato da questo helper "
            f"(source={enc_config.get('source')!r}, name={name!r}). Questa implementazione "
            "supporta solo gli encoder CNN EfficientNet usati dai checkpoint Mammo-FM "
            f"ufficiali: {sorted(_SUPPORTED_CNN_ENCODERS)}. Nessun fallback automatico su "
            "un'altra architettura (es. DINOv2/RAD-DINO/ViT generico)."
        )
    return _SUPPORTED_CNN_ENCODERS[name]


def build_mammofm_model(
    hf_repo: Optional[str] = DEFAULT_HF_REPO,
    checkpoint_name: Optional[str] = DEFAULT_CHECKPOINT_NAME,
    use_local_checkpoint: bool = False,
    local_checkpoint_path: Optional[str] = None,
    num_classes: int = 1,
    dropout: float = 0.1,
):
    """Costruisce il classificatore Mammo-FM fine-tunabile (EfficientNet-B5 + testa lineare).

    Ritorna ``(model, mean, std, img_size, hidden_size, backend, source_desc)``.
    Solleva ``MammoFMConfigError`` se il checkpoint non e' configurato, non e' raggiungibile,
    o non e' compatibile con l'architettura attesa: non sostituisce mai silenziosamente
    Mammo-FM con un backbone generico.
    """
    ckpt_path = _resolve_checkpoint_path(hf_repo, checkpoint_name, use_local_checkpoint, local_checkpoint_path)
    raw = _load_raw_checkpoint(ckpt_path)

    try:
        enc_config = raw["config"]["model"]["image_encoder"]
    except (KeyError, TypeError) as exc:
        raise MammoFMConfigError(
            f"Checkpoint '{ckpt_path}' privo di config['model']['image_encoder']: impossibile "
            "determinare l'architettura dell'encoder immagine Mammo-FM."
        ) from exc

    arch_name, out_dim = _resolve_encoder_arch(enc_config)
    image_encoder = MammoFMImageEncoder(arch_name=arch_name, out_dim=out_dim)

    full_state_dict = raw["model"]
    cleaned = {_strip_known_prefixes(k): v for k, v in full_state_dict.items()}
    backbone_keys = set(image_encoder.model.state_dict().keys())
    matched_keys = backbone_keys & set(cleaned.keys())
    match_ratio = len(matched_keys) / max(len(backbone_keys), 1)
    if match_ratio < 0.5:
        raise MammoFMConfigError(
            f"Il checkpoint '{ckpt_path}' non sembra compatibile con l'architettura "
            f"'{arch_name}' (solo {len(matched_keys)}/{len(backbone_keys)} tensori "
            "corrispondenti per nome). Verifica MAMMOFM_CHECKPOINT_NAME/il file di checkpoint: "
            "il notebook non procede con un caricamento parziale/errato spacciandolo per "
            "Mammo-FM."
        )
    missing, unexpected = image_encoder.model.load_state_dict(cleaned, strict=False)
    if missing:
        warnings.warn(
            f"Checkpoint Mammo-FM: {len(missing)} tensori del backbone non trovati nel "
            f"checkpoint (rimangono inizializzati casualmente), es. {list(missing)[:3]} "
            "(atteso: il layer '_fc' finale, non incluso nel checkpoint ufficiale e non usato "
            "da questo helper, che estrae solo le feature pooled)."
        )

    model = MammoFMClassifier(image_encoder, hidden_size=out_dim, num_classes=num_classes, dropout=dropout)
    source_desc = (
        f"local_checkpoint:{ckpt_path}" if use_local_checkpoint
        else f"huggingface:{hf_repo}/{checkpoint_name}"
    )
    source_desc += f" (encoder={arch_name}, config_name={enc_config.get('name')!r})"
    return model, DEFAULT_MAMMOFM_MEAN, DEFAULT_MAMMOFM_STD, DEFAULT_IMG_SIZE, out_dim, "efficientnet_pytorch", source_desc


# ---------------------------------------------------------------------------
# Freeze / unfreeze (fine-tuning reale, nessun adapter/LoRA)
# ---------------------------------------------------------------------------

def freeze_backbone_all(model: MammoFMClassifier) -> None:
    for p in model.image_encoder.parameters():
        p.requires_grad_(False)


def unfreeze_head(model: MammoFMClassifier) -> None:
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    for p in model.dropout.parameters():
        p.requires_grad_(True)


def unfreeze_last_n_blocks(model: MammoFMClassifier, n: int = 2) -> None:
    """Scongela le ultime n stage MBConv di EfficientNet-B5 + la testa conv finale
    (`_conv_head`/`_bn1`), analogo allo sblocco degli ultimi block Transformer per i backbone
    ViT (fine-tuning parziale reale, non adapter)."""
    backbone = model.image_encoder.model  # efficientnet_pytorch.EfficientNet
    blocks = list(backbone._blocks)
    if not blocks:
        raise MammoFMConfigError("Impossibile individuare i blocchi MBConv del backbone Mammo-FM.")
    for block in blocks[-int(n):]:
        for p in block.parameters():
            p.requires_grad_(True)
    for p in backbone._conv_head.parameters():
        p.requires_grad_(True)
    for p in backbone._bn1.parameters():
        p.requires_grad_(True)


def unfreeze_all(model: MammoFMClassifier) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Preprocessing / Dataset (grayscale->RGB, normalizzazione ufficiale Mammo-FM)
# ---------------------------------------------------------------------------

def _apply_clahe(arr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE come nel preprocessing ufficiale di pre-training Mammo-CLIP/Mammo-FM
    (`configs/transform/clahe.yaml`). Usata solo se `USE_CLAHE_PREPROCESSING=True` nel notebook
    (disattivata di default per coerenza con gli altri classificatori del progetto)."""
    try:
        import cv2
    except ImportError as exc:
        raise MammoFMConfigError(
            "USE_CLAHE_PREPROCESSING=True richiede il pacchetto opencv (cv2), non installato. "
            "Installa con: pip install opencv-python-headless"
        ) from exc
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(np.clip(arr, 0, 255).astype(np.uint8)).astype(np.float32)


class MammoFMDataset(Dataset):
    """Carica mammografie grayscale, le replica su 3 canali (richiesto dall'encoder EfficientNet-B5
    di Mammo-FM: `_conv_stem` ha 3 canali in ingresso nel checkpoint ufficiale) e applica la
    normalizzazione ufficiale Mammo-CLIP/Mammo-FM: min-max per immagine in [0,1], poi
    standardizzazione con `mean`/`std` (scalari, non statistiche ImageNet)."""

    def __init__(self, paths, labels, mean: float, std: float, img_size: int,
                 augment: bool = False, use_clahe: bool = False):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = float(mean)
        self.std = float(std)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.use_clahe = bool(use_clahe)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        with Image.open(path) as image:
            gray = image.convert("L").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)

        if self.use_clahe:
            arr = _apply_clahe(arr)

        if self.augment:
            # Il preprocessing orienta gia' il tessuto: evitare flip mantiene la stessa
            # convenzione anatomica degli altri classificatori del progetto.
            arr = np.clip(arr + np.random.uniform(-0.05 * 255, 0.05 * 255), 0.0, 255.0)

        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        arr = (arr - self.mean) / self.std

        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).float()  # 1 canale -> 3 canali (RGB)
        return tensor, torch.tensor(label, dtype=torch.float32)


def make_mammofm_dataloader(df: pd.DataFrame, path_col: str, label_col: str,
                             mean: float, std: float, img_size: int,
                             batch_size: int = 8, shuffle: bool = False, augment: bool = False,
                             use_clahe: bool = False, seed: int = 42, num_workers: int = 2,
                             drop_last: Optional[bool] = None) -> DataLoader:
    """Costruisce un DataLoader coerente con lo stile di `maxvit_utils.make_dataloader`.

    `drop_last` di default resta `None` (equivalente a `drop_last=shuffle`); i notebook
    Mammo-FM lo impostano esplicitamente a `False` sul train loader per non perdere immagini
    nell'ultimo batch parziale (il dataset reale e' piccolo e sbilanciato).
    """
    dataset = MammoFMDataset(
        paths=df[path_col].values, labels=df[label_col].values,
        mean=mean, std=std, img_size=img_size, augment=augment, use_clahe=use_clahe,
    )
    generator = torch.Generator().manual_seed(seed)
    if drop_last is None:
        drop_last = shuffle
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last,
        generator=generator if shuffle else None,
    )


# ---------------------------------------------------------------------------
# Dataset loader con tracciamento della sorgente (real / synthetic / augmented)
# ---------------------------------------------------------------------------

def image_paths(directory) -> list:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def load_real_split(base_path, split: str) -> pd.DataFrame:
    """Carica le immagini reali gia' preprocessate di uno split (train/val/test)."""
    rows = []
    split_dir = Path(base_path) / "data" / "processed" / split
    for label in (0, 1):
        for path in image_paths(split_dir / str(label)):
            rows.append({
                "processed_path": str(path), "cancer": label,
                "source": "real", "source_detail": f"real_{split}", "split": split,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"Nessuna immagine reale trovata in {split_dir}")
    return df


def load_synthetic_both(root, source_name: str, split_label: str = "train") -> pd.DataFrame:
    """Carica le sintetiche filtrate (positive/negative) da una cartella `root`."""
    root = Path(root)
    rows = []
    for folder, label in (("negative", 0), ("positive", 1)):
        folder_path = root / folder
        paths = image_paths(folder_path)
        if not paths:
            raise FileNotFoundError(
                f"Nessuna immagine sintetica '{folder}' in {folder_path}. "
                "Esegui prima il notebook generativo/di filtraggio corrispondente, "
                "oppure verifica il path configurato."
            )
        for path in paths:
            rows.append({
                "processed_path": str(path), "cancer": label,
                "source": "synthetic", "source_detail": source_name, "split": split_label,
            })
    return pd.DataFrame(rows)


def load_augmented_positive(base_path, split_label: str = "train") -> pd.DataFrame:
    """Carica il dataset di augmentation tradizionale (`data/real_augmented`)."""
    aug_dir = Path(base_path) / "data" / "real_augmented"
    rows = []
    if not aug_dir.is_dir():
        raise FileNotFoundError(
            f"Cartella {aug_dir} mancante. Esegui prima 02_Data_Augmentation_Trad.ipynb."
        )
    for path in image_paths(aug_dir):
        match = _LABEL_RE.search(path.name)
        if match is None:
            continue
        rows.append({
            "processed_path": str(path), "cancer": int(match.group(1)),
            "source": "augmented", "source_detail": "traditional_positive_augmentation", "split": split_label,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"Nessuna immagine augmentata riconosciuta in {aug_dir}")
    return df


def print_counts(name: str, df: pd.DataFrame) -> None:
    labels = df["cancer"].astype(int)
    print(f"{name:24s} tot={len(df):5d} | sano(0)={(labels == 0).sum():5d} | malato(1)={(labels == 1).sum():5d}")


def source_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["source_detail"], df["cancer"])
    table = table.rename(columns={0: "sano_0", 1: "malato_1"})
    table["totale"] = table.sum(axis=1)
    return table


def check_duplicate_paths(df: pd.DataFrame, path_col: str = "processed_path") -> int:
    """Segnala (warning) eventuali path duplicati nel dataframe combinato di training."""
    dup_mask = df.duplicated(subset=[path_col], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        examples = df.loc[dup_mask, path_col].unique()[:5].tolist()
        warnings.warn(f"Trovati {n_dup} path duplicati nel dataset combinato: {examples} ...")
    return n_dup


def check_no_split_overlap(splits: dict) -> None:
    """Verifica che nessun path reale compaia in piu' di uno split (anti data-leakage).

    ``splits`` e' un dict {nome_split: dataframe} con colonna ``processed_path``.
    """
    seen = {}
    for name, df in splits.items():
        for p in df["processed_path"]:
            if p in seen and seen[p] != name:
                raise AssertionError(
                    f"Data leakage rilevato: '{p}' presente sia in '{seen[p]}' che in '{name}'."
                )
            seen[p] = name


# ---------------------------------------------------------------------------
# Training con mixed precision, gradient clipping, gradient accumulation
# ---------------------------------------------------------------------------

def train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler=None,
                         grad_clip_norm: Optional[float] = None, accumulation_steps: int = 1,
                         start_batch: int = 0, global_step: int = 0, max_optimizer_updates: int | None = None,
                         on_optimizer_step=None) -> dict:
    from sklearn.metrics import roc_auc_score

    model.train()
    refreeze_batchnorm(model)  # le BatchNorm2d di EfficientNet congelate restano in eval()
    total_loss, n_seen = 0.0, 0
    y_true, y_prob = [], []
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(loader):
        if step < start_batch:
            continue
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=(scaler is not None and device.type == "cuda")):
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_batch = (step + 1) == len(loader)
        if (step + 1) % accumulation_steps == 0 or is_last_batch:
            if grad_clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if on_optimizer_step is not None:
                on_optimizer_step(global_step, step)
            if max_optimizer_updates is not None and global_step >= max_optimizer_updates:
                break

        total_loss += loss.item() * accumulation_steps * imgs.size(0)
        n_seen += imgs.size(0)
        y_true.extend(labels.detach().cpu().numpy())
        y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy())

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / max(n_seen, 1), "auc": auc, "global_step": global_step, "last_batch": step}


@torch.no_grad()
def evaluate_amp(model, loader, criterion, device) -> dict:
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss, n_seen = 0.0, 0
    y_true, y_prob = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        n_seen += imgs.size(0)
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(torch.sigmoid(logits).cpu().numpy())

    y_true_arr, y_prob_arr = np.array(y_true), np.array(y_prob)
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / max(n_seen, 1), "auc": auc, "y_true": y_true_arr, "y_prob": y_prob_arr}


def fit_mammofm(model, train_loader, val_loader, optimizer, criterion, epochs: int, device,
                early_stopping=None, checkpoint=None, csv_logger=None, lr_scheduler=None,
                use_amp: bool = True, grad_clip_norm: Optional[float] = 1.0,
                accumulation_steps: int = 1, start_epoch: int = 1, start_batch: int = 0,
                global_step: int = 0, max_optimizer_updates: int | None = None,
                scaler=None, on_optimizer_step=None, on_epoch_end=None) -> History:
    """Training loop stile Keras (fit) con AMP + gradient clipping + gradient accumulation.

    Non duplica `maxvit_utils.fit`: qui serve gestire esplicitamente le BatchNorm2d
    dell'encoder EfficientNet (assenti nei backbone ViT usati da MammoDINO/RAD-DINO) oltre
    ad AMP/accumulo del gradiente.
    """
    history = History()
    scaler = scaler or (torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch_amp(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, grad_clip_norm=grad_clip_norm, accumulation_steps=accumulation_steps,
            start_batch=start_batch if epoch == start_epoch else 0, global_step=global_step,
            max_optimizer_updates=max_optimizer_updates, on_optimizer_step=on_optimizer_step,
        )
        global_step = train_metrics.pop("global_step")
        val_metrics = evaluate_amp(model, val_loader, criterion, device)
        history.append(train_metrics, val_metrics)

        print(f"Epoch {epoch}/{epochs} - loss: {train_metrics['loss']:.4f} - auc: {train_metrics['auc']:.4f} "
              f"- val_loss: {val_metrics['loss']:.4f} - val_auc: {val_metrics['auc']:.4f}")

        if csv_logger is not None:
            csv_logger.log({
                "epoch": epoch, "loss": train_metrics["loss"], "auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"], "val_auc": val_metrics["auc"],
            })
        if checkpoint is not None:
            checkpoint.step(val_metrics["auc"], model)
        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["auc"])
        if early_stopping is not None:
            early_stopping.step(val_metrics["auc"], model)
            if early_stopping.stop:
                print(f"Early stopping all'epoca {epoch} (best val_auc={early_stopping.best:.4f})")
                break
        if on_epoch_end is not None:
            on_epoch_end(epoch, global_step, scaler, history)
        if max_optimizer_updates is not None and global_step >= max_optimizer_updates:
            break

    if early_stopping is not None:
        early_stopping.restore(model)
    history.global_step = global_step
    history.scaler = scaler
    return history


def predict_with_probs(model, df: pd.DataFrame, path_col: str, label_col: str,
                        mean: float, std: float, img_size: int, batch_size: int, device) -> tuple:
    """Predice su un dataframe con shuffle=False: l'ordine di y_true/y_prob combacia con df."""
    loader = make_mammofm_dataloader(
        df, path_col, label_col, mean, std, img_size, batch_size=batch_size, shuffle=False,
    )
    model.eval()
    y_prob = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(-1)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy())
    y_prob = np.array(y_prob)
    y_true = df[label_col].values.astype(int)
    if len(y_prob) != len(df):
        raise RuntimeError("Mismatch tra numero di predizioni e righe del dataframe (shuffle inatteso?).")
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Metriche estese
# ---------------------------------------------------------------------------

def compute_full_metrics(y_true, y_prob, threshold: float, split: str,
                          experiment_name: str, config_name: str, extra: Optional[dict] = None):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    metrics = {
        "experiment_name": experiment_name,
        "config": config_name,
        "split": split,
        "n_samples": int(len(y_true)),
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc_average_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Sano", "Malato"], zero_division=0
        ),
    }
    if extra:
        metrics.update(extra)
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history: History, title_suffix: str, save_path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set(title=f"Loss {title_suffix}", xlabel="Epoca", ylabel="Loss")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(history.history["auc"], label="Train AUC", linewidth=2)
    ax2.plot(history.history["val_auc"], label="Val AUC", linewidth=2)
    ax2.set(title=f"AUC {title_suffix}", xlabel="Epoca", ylabel="AUC")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_pr_confusion(y_true, y_prob, y_pred, title: str, save_path) -> None:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Sano", "Malato"]); axes[0].set_yticklabels(["Sano", "Malato"])
    axes[0].set_xlabel("Predetto"); axes[0].set_ylabel("Reale"); axes[0].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         fontsize=13, fontweight="bold", color=color)

    axes[1].plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC curve")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(rec, prec, lw=2, label=f"AP={ap:.4f}")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title("Precision-Recall curve")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_curve(y_true, y_prob, title: str, save_path, n_bins: int = 10) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "o-", label=f"Brier={brier:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibrazione ideale")
    ax.set_xlabel("Probabilita' predetta media")
    ax.set_ylabel("Frazione di positivi osservata")
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Salvataggio risultati
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_training_history_csv(history: History, path) -> None:
    path = Path(path)
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_training_history_combined_csv(history_phase1: History, history_phase2: History, path) -> None:
    """Salva un unico CSV con lo storico di entrambe le fasi, con colonna `phase`.

    Non sostituisce i CSV separati per fase (`training_history_fase1.csv` /
    `training_history_fase2.csv`), che restano comunque salvati a parte.
    """
    df1 = pd.DataFrame(history_phase1.history)
    df1.insert(0, "epoch", np.arange(1, len(df1) + 1))
    df1.insert(1, "phase", "head_training")

    df2 = pd.DataFrame(history_phase2.history)
    df2.insert(0, "epoch", np.arange(1, len(df2) + 1))
    df2.insert(1, "phase", "fine_tuning")

    combined = pd.concat([df1, df2], ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def save_test_predictions(df_test: pd.DataFrame, y_true, y_prob, threshold: float,
                           path_col: str, source_col: str, save_path, split_name: str = "test") -> pd.DataFrame:
    out = pd.DataFrame({
        "path": df_test[path_col].values,
        "label": np.asarray(y_true).astype(int),
        "probability": np.asarray(y_prob).astype(float),
        "prediction": (np.asarray(y_prob) >= threshold).astype(int),
        "split": split_name,
        "source": df_test[source_col].values if source_col in df_test.columns else "real",
    })
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    return out


def auto_interpret(test_metrics: dict, val_metrics: dict, config_name: str) -> str:
    """Genera un breve testo di interpretazione automatica dei risultati del test set."""
    auc = test_metrics["roc_auc"]
    if auc >= 0.9:
        qualita = "eccellente"
    elif auc >= 0.8:
        qualita = "buona"
    elif auc >= 0.7:
        qualita = "discreta"
    elif auc >= 0.6:
        qualita = "debole"
    else:
        qualita = "scarsa (vicina al caso)"

    delta_auc = test_metrics["roc_auc"] - val_metrics["roc_auc"]
    overfit_note = ""
    if abs(delta_auc) > 0.05:
        overfit_note = (
            f"\n- **Attenzione**: scarto ROC-AUC val->test di {delta_auc:+.4f}, "
            "possibile overfitting sul validation set o shift di distribuzione tra i due split."
        )

    lines = [
        f"### Interpretazione automatica — {config_name}",
        "",
        f"- ROC-AUC sul test set: **{test_metrics['roc_auc']:.4f}** -> capacita' discriminativa {qualita}.",
        f"- PR-AUC (average precision): **{test_metrics['pr_auc_average_precision']:.4f}**.",
        f"- Balanced Accuracy: **{test_metrics['balanced_accuracy']:.4f}**, F1: **{test_metrics['f1']:.4f}**.",
        f"- Sensitivity/Recall: **{test_metrics['recall_sensitivity']:.4f}**, "
        f"Specificity: **{test_metrics['specificity']:.4f}**.",
        f"- Brier score: **{test_metrics['brier_score']:.4f}** (piu' basso = probabilita' meglio calibrate)."
        + overfit_note,
    ]
    return "\n".join(lines)


__all__ = [
    "BinaryFocalLoss", "CSVLogger", "EarlyStopping", "History", "ModelCheckpoint",
    "bootstrap_balanced", "compute_pos_weight", "count_trainable_params", "optimal_threshold_youden",
    "refreeze_batchnorm",
    "MammoFMConfigError", "MammoFMImageEncoder", "MammoFMClassifier",
    "build_mammofm_model",
    "freeze_backbone_all", "unfreeze_head", "unfreeze_last_n_blocks", "unfreeze_all",
    "MammoFMDataset", "make_mammofm_dataloader",
    "image_paths", "load_real_split", "load_synthetic_both", "load_augmented_positive",
    "print_counts", "source_table", "check_duplicate_paths", "check_no_split_overlap",
    "train_one_epoch_amp", "evaluate_amp", "fit_mammofm", "predict_with_probs",
    "compute_full_metrics",
    "plot_training_history", "plot_roc_pr_confusion", "plot_calibration_curve",
    "seed_everything", "save_json", "save_training_history_csv", "save_training_history_combined_csv",
    "save_test_predictions",
    "auto_interpret",
    "DEFAULT_HF_REPO", "DEFAULT_CHECKPOINT_NAME", "DEFAULT_IMG_SIZE",
    "DEFAULT_MAMMOFM_MEAN", "DEFAULT_MAMMOFM_STD",
]
