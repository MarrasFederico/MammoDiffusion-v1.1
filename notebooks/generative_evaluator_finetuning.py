#classi e procedure per calcolare fid e inception score per immagini mammografiche in scala di grigi
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


#warning relativo al buffer di torchmetrics viene soppresso poichè si lavora con ~3k immagini
warnings.filterwarnings(
    "ignore",
    message="Metric `InceptionScore` will save all extracted features in buffer",
    category=UserWarning,
    module="torchmetrics",
)


#configurazione del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GenerativeEvaluator")


#estensioni per il dataset
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
)


"""
-carica immagini da directory, le converte in scala di grigi e le replica su 3 canali così che InceptionV3 possa elaborarle 
-restituisce tipo di dato uint8 con shape (3, H, W) e valori in [0, 255] per compatibilità
 con FrechetInceptionDistance e InceptionScore 
----------
root: str / Path - directory che contiene le immagini 
image_size: int - valore di default che ridimensiona le dimensioni spaziali (per inception 299)
"""
class GrayscaleImageDataset(Dataset):

    #indicizza i file immagine in "root" e inizializza pipeline
    def __init__(self, root: str | Path, image_size: int = 299) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"cartella delle immagini non trovata: {self.root}")

        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"la cartella è vuota -> {self.root}. "
                f"estensioni supportate: {SUPPORTED_EXTENSIONS}"
            )
        logger.info("trovate %d immagini dentro %s", len(self.paths), self.root)

        #pipeline di trasformazione da greyscale e resize 
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), #forza 1 canale
            transforms.Resize((image_size, image_size), #porta a image_size x image_size
                               interpolation=transforms.InterpolationMode.BILINEAR, #interpolazione
                               antialias=True), #riduce l'aliasing durante ridimensionamento
            transforms.ToTensor(),           #(1, H, W) con tipo float32 [0, 1]
        ])


    #rende il numero di immagini presenti nella cartella
    def __len__(self) -> int:
        return len(self.paths)


    #carica l'immagine idx in greyscale, la copia su 3 canali e la restituisce in uint8 (3,H,W)
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L")   #grayscale
        tensor = self.transform(img)                     #(1, H, W) float32
        tensor = tensor.repeat(3, 1, 1)                  #(3, H, W) float32
        tensor = (tensor * 255).to(torch.uint8)          #(3, H, W) uint8
        return tensor



"""
-calcola FID e Inception Score tra un set di immagini reali e generate
----------
-real_dir : str / Path - directory contenente le immagini reali
-generated_dir : str / Path - directory contenente le immagini generate
-batch_size : int - numero di immagini per il batch DataLoader
-image_size : int - risoluzione da passare ad InceptionV3 (default=299)
-num_workers : int - processi di DataLoader (default=0) 
-device : str | None - dispositivo es:"cuda", "cpu", o None per auto-detection
-feature_dim : int - dimensionalita' delle feature su InceptionV3 per FID (64/192/768/2048).
"""
class GenerativeEvaluator:

    #costruisce DataLoader per immagini reali/generate e inizializza metriche FID e IS
    def __init__(self, real_dir: str | Path, generated_dir: str | Path, batch_size: int = 32, image_size: int = 299,
                num_workers: int = 0, device: str | None = None, feature_dim: int = 2048) -> None:
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.feature_dim = feature_dim
        self.real_features_: np.ndarray | None = None
        self.generated_features_: np.ndarray | None = None

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Using device: %s", self.device)

        self._real_loader = self.build_loader(real_dir) #DataLoader immagini reali
        self._gen_loader = self.build_loader(generated_dir) #DataLoader immagini generate
        n_gen = len(self._gen_loader.dataset) #numero di immagini generate
        is_buffer_mb = (n_gen * 1000 * 4) / 1024**2 #stima RAM del buffer logit di IS
        logger.info("stima buffer logit IS: %d immagini x 1000 logits x float32 ~= %.1f MB", n_gen, is_buffer_mb)
        if is_buffer_mb > 500:
            logger.warning("buffer IS eccede i 500 MB")

        #metrica FID (aspetta uint8)
        self._fid = FrechetInceptionDistance(
            feature=self.feature_dim,
            normalize=False,          #si aspetta uint8
        ).to(self.device)

        #il warning viene soppresso perche lavorando con circa 3k immagini 
        #il buffer di InceptionScore non si satura
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Metric `InceptionScore` will save all extracted features in buffer",
                category=UserWarning,
            )
            self._inception_score = InceptionScore(
                feature="logits_unbiased",
                splits=10,
                normalize=False,
            ).to(self.device)

    
    #crea il DataLoader per le immagini di una cartella
    def build_loader(self, directory: str | Path) -> DataLoader:
        dataset = GrayscaleImageDataset(directory, image_size=self.image_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size, #immagini per batch
            shuffle=False, #ordine fisso (non serve per FID/IS)
            num_workers=self.num_workers, #processi di caricamento
            pin_memory=(self.device.type == "cuda"), #pin memory solo su GPU
            drop_last=False, #tiene anche l'ultimo batch incompleto
        )


    #estrae una sola volta le feature Inception e aggiorna gli stessi accumulatori usati dal FID
    def update_fid(self, loader: DataLoader, real: bool) -> np.ndarray:
        tag = "real" if real else "generated"
        logger.info("estraendo inception features per immagini %s", tag)
        extracted_features = []
        #replica FrechetInceptionDistance.update, conservando le stesse feature per PRDC
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


    #accumula logit inception delle immagini generate nella Inception Score
    def update_is(self, loader: DataLoader) -> None:
        logger.info("estraendo inception logits per IS")
        #un batch alla volta
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)#sposta il batch sul device
            self._inception_score.update(batch)#accumula i logit
        logger.info("IS aggiornato") #fine aggiornamento IS


    """
    procedura per avviare una valutazione
    -------
    dizionario con chiavi:
    -"FID" - Frechet Inception Distance
    -"IS_mean" - mean Inception Score
    -"IS_std" - std Inception Score
    """
    def compute_with_features(self) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Calcola FID/IS e restituisce le stesse feature Inception usate dal FID."""
        self._fid.reset()
        self._inception_score.reset()

        #FID richiede feature da entrambe le distribuzioni
        self.real_features_ = self.update_fid(self._real_loader, real=True)
        self.generated_features_ = self.update_fid(self._gen_loader, real=False)
        #IS calcolata solo sulle immagini generate
        self.update_is(self._gen_loader)

        logger.info("calcolo metriche")
        fid_value: float = self._fid.compute().item()
        is_mean, is_std = self._inception_score.compute()

        results = {
            "FID":     round(fid_value, 4),
            "IS_mean": round(is_mean.item(), 4),
            "IS_std":  round(is_std.item(), 4),
        }
        logger.info("risultati: %s", results)
        return results, self.real_features_, self.generated_features_


    def compute(self) -> dict[str, float]:
        """Calcola FID e Inception Score mantenendo l'API storica."""
        results, _, _ = self.compute_with_features()
        return results