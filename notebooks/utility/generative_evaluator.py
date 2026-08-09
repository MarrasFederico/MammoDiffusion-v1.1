"""Compute FID and Inception Score for grayscale mammography images."""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms


# The torchmetrics buffer warning is harmless for the roughly 3,000 images used here.
warnings.filterwarnings(
    "ignore",
    message="Metric `InceptionScore` will save all extracted features in buffer",
    category=UserWarning,
    module="torchmetrics",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GenerativeEvaluator")


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
)


class GrayscaleImageDataset(Dataset):
    """Load grayscale images as three-channel uint8 tensors for InceptionV3.

    The output has shape ``(3, H, W)`` and values in ``[0, 255]``, as required
    by ``FrechetInceptionDistance`` and ``InceptionScore``.

    Args:
        root: Directory containing the images.
        image_size: Spatial output size; InceptionV3 normally uses 299.
    """

    def __init__(self, root: str | Path, image_size: int = 299) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"image directory not found: {self.root}")

        from parallel_generation_utils import metric_image_paths
        self.paths = metric_image_paths(self.root, SUPPORTED_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(
                f"image directory is empty: {self.root}. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )
        logger.info("found %d images in %s", len(self.paths), self.root)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),  # (1, H, W), float32 in [0, 1].
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L")
        tensor = self.transform(img)
        tensor = tensor.repeat(3, 1, 1)
        tensor = (tensor * 255).to(torch.uint8)
        return tensor

class GenerativeEvaluator:
    """Compute FID and Inception Score between real and generated image sets.

    Args:
        real_dir: Directory containing real images.
        generated_dir: Directory containing generated images.
        batch_size: DataLoader batch size.
        image_size: InceptionV3 input resolution.
        num_workers: Number of DataLoader worker processes.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for automatic selection.
        feature_dim: InceptionV3 feature dimension for FID (64/192/768/2048).
    """

    def __init__(self, real_dir: str | Path, generated_dir: str | Path, batch_size: int = 32, image_size: int = 299,
                num_workers: int = 0, device: str | None = None, feature_dim: int = 2048, is_splits: int = 10) -> None:

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.feature_dim = feature_dim
        self.is_splits = is_splits
        self.real_features_: np.ndarray | None = None
        self.generated_features_: np.ndarray | None = None

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Using device: %s", self.device)

        self._real_loader = self.build_loader(real_dir)
        self._gen_loader = self.build_loader(generated_dir)
        n_gen = len(self._gen_loader.dataset)
        is_buffer_mb = (n_gen * 1000 * 4) / 1024**2
        logger.info("estimated IS logit buffer: %d images x 1000 logits x float32 ~= %.1f MB", n_gen, is_buffer_mb)
        if is_buffer_mb > 500:
            logger.warning("IS buffer exceeds 500 MB")

        self._fid = FrechetInceptionDistance(
            feature=self.feature_dim,
            normalize=False,  # Input is uint8.
        ).to(self.device)

        # The InceptionScore buffer does not fill up at this dataset size.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Metric `InceptionScore` will save all extracted features in buffer",
                category=UserWarning,
            )
            self._inception_score = InceptionScore(
                feature="logits_unbiased",
                splits=self.is_splits,
                normalize=False,
            ).to(self.device)

    def build_loader(self, directory: str | Path) -> DataLoader:
        """Create the DataLoader for one image directory."""
        dataset = GrayscaleImageDataset(directory, image_size=self.image_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def update_fid(self, loader: DataLoader, real: bool) -> np.ndarray:
        """Extract Inception features once and update the FID accumulators."""
        tag = "real" if real else "generated"
        logger.info("extracting Inception features for %s images", tag)
        extracted_features = []
        # Mirror FrechetInceptionDistance.update and retain the same features for PRDC.
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                if self._fid.normalize and not self._fid.used_custom_model:
                    batch = (batch * 255).byte()

                features = self._fid.inception(batch)
                self._fid.orig_dtype = features.dtype
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                extracted_features.append(features.detach().cpu().float())

                features_double = features.double()
                if real:
                    self._fid.real_features_sum += features_double.sum(dim=0)
                    self._fid.real_features_cov_sum += features_double.t().mm(features_double)
                    self._fid.real_features_num_samples += batch.shape[0]
                else:
                    self._fid.fake_features_sum += features_double.sum(dim=0)
                    self._fid.fake_features_cov_sum += features_double.t().mm(features_double)
                    self._fid.fake_features_num_samples += batch.shape[0]

        logger.info("-> %s", tag)
        return torch.cat(extracted_features, dim=0).numpy()

    def update_is(self, loader: DataLoader) -> None:
        """Accumulate generated-image logits for Inception Score."""
        logger.info("extracting Inception logits for IS")
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            self._inception_score.update(batch)
        logger.info("IS updated")

    def compute_with_features(self) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Compute FID/IS and return the Inception features used by FID."""
        self._fid.reset()
        self._inception_score.reset()

        # FID requires features from both distributions.
        self.real_features_ = self.update_fid(self._real_loader, real=True)
        self.generated_features_ = self.update_fid(self._gen_loader, real=False)
        # IS is computed only on generated images.
        self.update_is(self._gen_loader)

        logger.info("computing metrics")
        fid_value: float = self._fid.compute().item()
        is_mean, is_std = self._inception_score.compute()

        results = {
            "FID":     round(fid_value, 4),
            "IS_mean": round(is_mean.item(), 4),
            "IS_std":  round(is_std.item(), 4),
        }
        logger.info("results: %s", results)
        return results, self.real_features_, self.generated_features_


    def compute(self) -> dict[str, float]:
        """Compute FID and Inception Score while preserving the historical API."""
        results, _, _ = self.compute_with_features()
        return results
