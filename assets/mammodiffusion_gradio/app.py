"""Local Gradio demo for the MammoDiffusion generators."""

from __future__ import annotations

import argparse
import gc
import json
import os
import secrets
import subprocess
import sys
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
UTILITY_DIR = NOTEBOOKS_DIR / "utility"

SD_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "diffusers" / "02_sd21_filtered_100steps"
SD_BASE_MODEL_DIR = NOTEBOOKS_DIR / "pretrained_model" / "stable-diffusion-2-1-base"
SD_CHECKPOINT_DIR = SD_EXPERIMENT_DIR / "model" / "checkpoint-3000"

LDM_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "diffusers" / "06_ldm_extra1361_fromscratch"
LDM_MODEL_PATH = LDM_EXPERIMENT_DIR / "checkpoints_ldm" / "ldm_step070000.keras"
LDM_VAE_DECODER_PATH = LDM_EXPERIMENT_DIR / "models" / "vae_decoder_best.keras"
LDM_LATENT_STATS_PATH = LDM_EXPERIMENT_DIR / "latents" / "latent_stats.npz"
LDM_DEFAULT_CUDA_ROOT = Path("/home/fede/miniforge3/envs/tf-gpu")

OUTPUT_DIR = APP_DIR / "outputs"

SD_DEFAULT_STEPS = 50
SD_DEFAULT_GUIDANCE = 7.5
LDM_DEFAULT_STEPS = 100
LDM_DEFAULT_GUIDANCE = 1.5
RESOLUTION = 512
MAX_IMAGES = 12

SD_MODEL_CHOICE = "Fine-tuned SD 2.1 (notebook 02 - checkpoint-3000)"
LDM_MODEL_CHOICE = "LDM from scratch custom-VAE (notebook 06 - step 70000)"
DEFAULT_MODEL_CHOICE = SD_MODEL_CHOICE

PROMPTS = {
    "Positiva": (
        "grayscale MLO mammogram, breast cancer positive, "
        "malignant finding, suspicious lesion, medical imaging"
    ),
    "Negativa": (
        "grayscale MLO mammogram, breast cancer negative, "
        "no malignant finding, normal screening mammogram, medical imaging"
    ),
}

LABEL_KEYS = {
    "Positiva": "positive",
    "Negativa": "negative",
}

LDM_CLASS_IDS = {
    "Negativa": 0,
    "Positiva": 1,
}

LDM_CLASS_PREVIEWS = {
    "Positiva": "LDM custom-VAE del notebook 06, condizionato sulla classe positiva.",
    "Negativa": "LDM custom-VAE del notebook 06, condizionato sulla classe negativa.",
}

MODEL_DEFAULTS = {
    SD_MODEL_CHOICE: {
        "slug": "sd21_02_checkpoint-3000",
        "status_name": "Stable Diffusion 2.1 fine-tuned (notebook 02)",
        "checkpoint": "checkpoint-3000",
        "default_steps": SD_DEFAULT_STEPS,
        "default_guidance": SD_DEFAULT_GUIDANCE,
    },
    LDM_MODEL_CHOICE: {
        "slug": "ldm_06_step070000",
        "status_name": "LDM from scratch custom-VAE (notebook 06)",
        "checkpoint": "ldm_step070000.keras",
        "default_steps": LDM_DEFAULT_STEPS,
        "default_guidance": LDM_DEFAULT_GUIDANCE,
    },
}


@dataclass
class LdmRuntime:
    tf: Any
    np: Any
    make_compiled_sampler: Any
    ldm_model: Any
    vae_decoder: Any
    schedule: Any
    latent_mean: Any
    latent_std: Any
    sampler: Any | None = None
    sampler_steps: int | None = None
    sampler_guidance: float | None = None


SD_PIPELINE: StableDiffusionPipeline | None = None
LDM_RUNTIME: LdmRuntime | None = None
MODEL_LOCK = threading.Lock()


CSS = """
:root {
    --md-bg: #080c14;
    --md-panel: #101725;
    --md-panel-soft: #151e2e;
    --md-border: #29354a;
    --md-text-soft: #a7b2c5;
    --md-accent: #ff7a18;
    --md-accent-hover: #ff913f;
}

body, .gradio-container {
    background: var(--md-bg) !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
}

#md-header {
    padding: 18px 4px 8px 4px;
}

#md-header h1 {
    margin: 0;
    color: #f7f9fc;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

#md-header p {
    margin: 4px 0 0 0;
    color: var(--md-text-soft);
}

#main-gallery {
    min-height: 570px;
    border: 1px solid var(--md-border) !important;
    border-radius: 10px !important;
    background: #0b101a !important;
    overflow: hidden;
}

#main-gallery .grid-wrap {
    background: #0b101a !important;
}

#control-panel, #options-panel {
    padding: 10px !important;
    border: 1px solid var(--md-border);
    border-radius: 10px;
    background: var(--md-panel);
}

#prompt-preview textarea {
    min-height: 84px !important;
    color: #e8edf6 !important;
    background: var(--md-panel-soft) !important;
}

#generate-button {
    min-height: 84px !important;
    border: 1px solid #ff9e55 !important;
    color: white !important;
    background: var(--md-accent) !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
}

#generate-button:hover {
    background: var(--md-accent-hover) !important;
}

#status-box {
    color: var(--md-text-soft);
}

.md-warning {
    color: #8490a5;
    font-size: 0.84rem;
}
"""

THEME = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)


def is_valid_keras_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def validate_sd_model_paths() -> None:
    required = [
        SD_BASE_MODEL_DIR / "model_index.json",
        SD_CHECKPOINT_DIR / "unet" / "config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Modello fine-tuned del notebook 02 non trovato. File richiesti mancanti:\n"
            + details
        )


def validate_ldm_model_paths() -> None:
    required = [
        LDM_MODEL_PATH,
        LDM_VAE_DECODER_PATH,
        LDM_LATENT_STATS_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    invalid_keras = [
        path
        for path in [LDM_MODEL_PATH, LDM_VAE_DECODER_PATH]
        if path.exists() and not is_valid_keras_file(path)
    ]
    if missing or invalid_keras:
        lines = []
        if missing:
            lines.append("File mancanti:")
            lines.extend(f"- {path}" for path in missing)
        if invalid_keras:
            lines.append("File Keras non validi:")
            lines.extend(f"- {path}" for path in invalid_keras)
        raise FileNotFoundError(
            "Modello LDM del notebook 06 step 70000 non trovato o non valido.\n" + "\n".join(lines)
        )


def configure_ldm_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if "XLA_FLAGS" in os.environ:
        return

    env_cuda_root = os.environ.get("MAMMODIFFUSION_CUDA_ROOT")
    cuda_roots = []
    if env_cuda_root:
        cuda_roots.append(Path(env_cuda_root).expanduser())
    cuda_roots.extend(
        [
            Path(sys.executable).resolve().parents[1],
            LDM_DEFAULT_CUDA_ROOT,
            Path("/usr/local/cuda"),
        ]
    )

    for cuda_root in cuda_roots:
        libdevice = cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.exists():
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_root}"
            return


def release_sd_pipeline() -> None:
    global SD_PIPELINE

    if SD_PIPELINE is None:
        return

    pipeline = SD_PIPELINE
    SD_PIPELINE = None
    del pipeline
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def release_ldm_runtime() -> None:
    global LDM_RUNTIME

    if LDM_RUNTIME is None:
        return

    runtime = LDM_RUNTIME
    LDM_RUNTIME = None
    tf = runtime.tf
    del runtime
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


def release_loaded_models() -> None:
    with MODEL_LOCK:
        release_sd_pipeline()
        release_ldm_runtime()
    gc.collect()


def get_sd_pipeline() -> StableDiffusionPipeline:
    """Load the selected validation checkpoint once, on first SD generation."""
    global SD_PIPELINE

    if str(UTILITY_DIR) not in sys.path:
        sys.path.insert(0, str(UTILITY_DIR))
    from sd_vae_utils import ensure_diffusers_available

    ensure_diffusers_available(PROJECT_ROOT)

    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    if SD_PIPELINE is not None:
        return SD_PIPELINE

    with MODEL_LOCK:
        if SD_PIPELINE is not None:
            return SD_PIPELINE

        release_ldm_runtime()
        validate_sd_model_paths()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        unet = UNet2DConditionModel.from_pretrained(
            str(SD_CHECKPOINT_DIR / "unet"),
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            str(SD_BASE_MODEL_DIR),
            unet=unet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        pipeline = pipeline.to(device)
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
        pipeline.set_progress_bar_config(disable=True)

        SD_PIPELINE = pipeline
        return SD_PIPELINE


def import_ldm_utils(seed: int):
    configure_ldm_environment()

    if str(UTILITY_DIR) not in sys.path:
        sys.path.insert(0, str(UTILITY_DIR))

    import numpy as np
    import tensorflow as tf
    from ldm_keras_utils import (
        build_schedule,
        configure_tensorflow,
        load_latent_stats,
        load_ldm_model,
        load_vae_decoder,
        make_compiled_sampler,
    )

    configure_tensorflow(seed=seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return (
        np,
        tf,
        build_schedule,
        load_latent_stats,
        load_ldm_model,
        load_vae_decoder,
        make_compiled_sampler,
    )


def get_ldm_runtime(sample_steps: int, guidance_scale: float, seed: int) -> LdmRuntime:
    """Load the notebook 06 LDM and rebuild the sampler only when settings change."""
    global LDM_RUNTIME

    with MODEL_LOCK:
        release_sd_pipeline()
        validate_ldm_model_paths()

        if LDM_RUNTIME is None:
            (
                np,
                tf,
                build_schedule,
                load_latent_stats,
                load_ldm_model,
                load_vae_decoder,
                make_compiled_sampler,
            ) = import_ldm_utils(seed=seed)

            LDM_RUNTIME = LdmRuntime(
                tf=tf,
                np=np,
                make_compiled_sampler=make_compiled_sampler,
                ldm_model=load_ldm_model(LDM_MODEL_PATH),
                vae_decoder=load_vae_decoder(LDM_VAE_DECODER_PATH),
                schedule=build_schedule(),
                latent_mean=None,
                latent_std=None,
            )
            LDM_RUNTIME.latent_mean, LDM_RUNTIME.latent_std = load_latent_stats(
                LDM_LATENT_STATS_PATH
            )

        if (
            LDM_RUNTIME.sampler is None
            or LDM_RUNTIME.sampler_steps != int(sample_steps)
            or LDM_RUNTIME.sampler_guidance != float(guidance_scale)
        ):
            LDM_RUNTIME.sampler = LDM_RUNTIME.make_compiled_sampler(
                ldm_model=LDM_RUNTIME.ldm_model,
                vae_decoder=LDM_RUNTIME.vae_decoder,
                schedule=LDM_RUNTIME.schedule,
                latent_mean=LDM_RUNTIME.latent_mean,
                latent_std=LDM_RUNTIME.latent_std,
                num_steps=int(sample_steps),
                guidance_scale=float(guidance_scale),
                decode_on_cpu=False,
            )
            LDM_RUNTIME.sampler_steps = int(sample_steps)
            LDM_RUNTIME.sampler_guidance = float(guidance_scale)

        return LDM_RUNTIME


def prompt_for_selection(model_choice: str, label: str) -> str:
    if model_choice == LDM_MODEL_CHOICE:
        return LDM_CLASS_PREVIEWS[label]
    return PROMPTS[label]


def defaults_for_model(model_choice: str, label: str):
    release_loaded_models()
    defaults = MODEL_DEFAULTS[model_choice]
    return (
        prompt_for_selection(model_choice, label),
        gr.update(value=defaults["default_steps"]),
        gr.update(value=defaults["default_guidance"]),
        f"Pronto. Selezionato: {defaults['status_name']}.",
    )


def make_output_session(model_choice: str, label: str, base_seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = MODEL_DEFAULTS[model_choice]["slug"]
    session_dir = (
        OUTPUT_DIR / f"{timestamp}_{model_slug}_{LABEL_KEYS[label]}_seed-{base_seed}"
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def save_image(
    image,
    destination: Path,
    model_choice: str,
    label: str,
    conditioning: str,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> None:
    model_defaults = MODEL_DEFAULTS[model_choice]
    metadata = PngInfo()
    metadata.add_text("model", model_defaults["status_name"])
    metadata.add_text("checkpoint", model_defaults["checkpoint"])
    metadata.add_text("class", LABEL_KEYS[label])
    metadata.add_text("conditioning", conditioning)
    metadata.add_text("seed", str(seed))
    metadata.add_text("inference_steps", str(steps))
    metadata.add_text("guidance_scale", str(guidance_scale))
    image.save(destination, pnginfo=metadata)


def ldm_image_to_pil(runtime: LdmRuntime, image_np) -> Image.Image:
    arr = runtime.np.asarray(image_np)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = runtime.np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    arr = runtime.np.clip(arr * 255.0, 0, 255).astype(runtime.np.uint8)
    return Image.fromarray(arr, mode="L")


def generate_sd_image(
    prompt: str,
    image_seed: int,
    inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    import torch

    pipeline = get_sd_pipeline()
    device = pipeline.device.type
    generator = torch.Generator(device=device).manual_seed(image_seed)

    with MODEL_LOCK, torch.inference_mode():
        result = pipeline(
            prompt,
            num_inference_steps=int(inference_steps),
            guidance_scale=float(guidance_scale),
            height=RESOLUTION,
            width=RESOLUTION,
            generator=generator,
        )
    return result.images[0]


def generate_ldm_image(
    label: str,
    image_seed: int,
    inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    runtime = get_ldm_runtime(
        sample_steps=int(inference_steps),
        guidance_scale=float(guidance_scale),
        seed=int(image_seed),
    )
    sample_seed = runtime.tf.constant([int(image_seed), 0], dtype=runtime.tf.int32)
    class_id = runtime.tf.constant(LDM_CLASS_IDS[label], dtype=runtime.tf.int32)

    with MODEL_LOCK:
        image_tensor = runtime.sampler(class_id, sample_seed)
        image_np = image_tensor[0].numpy()

    image = ldm_image_to_pil(runtime, image_np)
    del image_tensor
    del image_np
    gc.collect()
    return image


def emit_worker_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def worker_generate_images(
    model_choice: str,
    label: str,
    image_count: int,
    base_seed: int,
    inference_steps: int,
    guidance_scale: float,
    session_dir: Path,
) -> None:
    model_defaults = MODEL_DEFAULTS[model_choice]
    conditioning = prompt_for_selection(model_choice, label)
    session_dir.mkdir(parents=True, exist_ok=True)

    emit_worker_event(
        {
            "event": "status",
            "status": f"Caricamento di {model_defaults['status_name']}...",
        }
    )

    try:
        for index in range(image_count):
            image_seed = base_seed + index
            if model_choice == LDM_MODEL_CHOICE:
                image = generate_ldm_image(
                    label,
                    image_seed,
                    int(inference_steps),
                    float(guidance_scale),
                )
            else:
                image = generate_sd_image(
                    conditioning,
                    image_seed,
                    int(inference_steps),
                    float(guidance_scale),
                )

            destination = session_dir / f"{LABEL_KEYS[label]}_{index + 1:02d}.png"
            save_image(
                image,
                destination,
                model_choice,
                label,
                conditioning,
                image_seed,
                int(inference_steps),
                float(guidance_scale),
            )
            emit_worker_event(
                {
                    "event": "image",
                    "path": str(destination),
                    "caption": f"{label} - seed {image_seed}",
                    "status": (
                        f"Generate {index + 1}/{image_count} immagini con "
                        f"{model_defaults['status_name']}..."
                    ),
                }
            )

        emit_worker_event(
            {
                "event": "complete",
                "status": f"Completato. File salvati in `{display_path(session_dir)}`",
            }
        )
    finally:
        release_loaded_models()


def load_gallery_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def log_tail(log_path: Path, max_chars: int = 1800) -> str:
    if not log_path.exists():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    content = content.strip()
    return content[-max_chars:] if len(content) > max_chars else content


def generate_images(
    model_choice: str,
    label: str,
    image_count: int,
    requested_seed: int,
    inference_steps: int,
    guidance_scale: float,
):
    """Generate in a short-lived worker process to release VRAM after each run."""
    model_defaults = MODEL_DEFAULTS[model_choice]
    image_count = max(1, min(int(image_count), MAX_IMAGES))
    base_seed = -1 if requested_seed is None else int(requested_seed)
    if base_seed < 0:
        base_seed = secrets.randbelow(2**31)

    gallery_items = []
    session_dir = make_output_session(model_choice, label, base_seed)
    log_path = session_dir / "worker.log"
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-generate",
        "--model-choice",
        model_choice,
        "--label",
        label,
        "--image-count",
        str(image_count),
        "--base-seed",
        str(base_seed),
        "--inference-steps",
        str(int(inference_steps)),
        "--guidance-scale",
        str(float(guidance_scale)),
        "--session-dir",
        str(session_dir),
    ]

    release_loaded_models()
    yield (
        gallery_items,
        f"Avvio processo isolato per {model_defaults['status_name']}...",
        base_seed,
    )

    process: subprocess.Popen[str] | None = None
    worker_error: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                worker_command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("stdout del worker non disponibile")

            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log_file.write(f"[stdout] {line}\n")
                    log_file.flush()
                    continue

                event_type = event.get("event")
                if event_type == "status":
                    yield gallery_items, str(event.get("status", "")), base_seed
                elif event_type == "image":
                    image_path = Path(str(event["path"]))
                    gallery_items.append(
                        (load_gallery_image(image_path), str(event.get("caption", "")))
                    )
                    yield gallery_items, str(event.get("status", "")), base_seed
                elif event_type == "complete":
                    yield gallery_items, str(event.get("status", "")), base_seed
                elif event_type == "error":
                    worker_error = str(event.get("message", "Errore worker"))

            return_code = process.wait()

        if return_code != 0 or worker_error is not None:
            details = worker_error or f"worker terminato con codice {return_code}"
            tail = log_tail(log_path)
            if tail:
                details = f"{details}\n\nLog: `{display_path(log_path)}`\n```text\n{tail}\n```"
            else:
                details = f"{details}\n\nLog: `{display_path(log_path)}`"
            yield gallery_items, f"Errore durante la generazione: {details}", base_seed
    except Exception as exc:
        yield gallery_items, f"Errore durante la generazione: `{exc}`", base_seed
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        release_loaded_models()


def clear_gallery():
    return [], "Galleria pulita. I file già generati restano salvati."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MammoDiffusion Studio") as demo:
        gr.HTML(
            """
            <div id="md-header">
                <h1>MammoDiffusion Studio</h1>
                <p>Generazione condizionata con il diffusore SD 02 e il diffusore from scratch 06</p>
            </div>
            """
        )

        gallery = gr.Gallery(
            label=None,
            value=[],
            columns=4,
            rows=2,
            height=570,
            object_fit="contain",
            preview=True,
            show_label=False,
            elem_id="main-gallery",
        )

        with gr.Row(elem_id="control-panel"):
            prompt_preview = gr.Textbox(
                value=prompt_for_selection(DEFAULT_MODEL_CHOICE, "Positiva"),
                lines=2,
                max_lines=3,
                interactive=False,
                show_label=False,
                container=False,
                elem_id="prompt-preview",
            )
            generate_button = gr.Button(
                "Generate",
                variant="primary",
                elem_id="generate-button",
            )

        with gr.Row(elem_id="options-panel"):
            model_choice = gr.Radio(
                choices=list(MODEL_DEFAULTS),
                value=DEFAULT_MODEL_CHOICE,
                label="Modello",
            )
            label = gr.Radio(
                choices=list(PROMPTS),
                value="Positiva",
                label="Etichetta",
            )
            image_count = gr.Slider(
                minimum=1,
                maximum=MAX_IMAGES,
                value=4,
                step=1,
                label="Numero di immagini",
            )
            clear_button = gr.Button("Pulisci galleria", variant="secondary")

        with gr.Accordion("Impostazioni avanzate", open=False):
            with gr.Row():
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed iniziale",
                    info="-1 sceglie un seed casuale; le immagini successive usano seed + 1.",
                )
                used_seed = gr.Number(
                    value=None,
                    precision=0,
                    label="Seed usato",
                    interactive=False,
                )
                inference_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=SD_DEFAULT_STEPS,
                    step=1,
                    label="Inference steps",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=SD_DEFAULT_GUIDANCE,
                    step=0.5,
                    label="Guidance scale",
                )

        status = gr.Markdown(
            "Pronto. Il modello verrà caricato alla prima generazione.",
            elem_id="status-box",
        )
        gr.HTML(
            """
            <p class="md-warning">
                Demo sperimentale per ricerca. Le immagini generate non sono destinate
                alla diagnosi o all'uso clinico.
            </p>
            """
        )

        model_choice.change(
            fn=defaults_for_model,
            inputs=[model_choice, label],
            outputs=[prompt_preview, inference_steps, guidance_scale, status],
            queue=False,
        )
        label.change(
            fn=prompt_for_selection,
            inputs=[model_choice, label],
            outputs=prompt_preview,
            queue=False,
        )
        generate_button.click(
            fn=generate_images,
            inputs=[
                model_choice,
                label,
                image_count,
                seed,
                inference_steps,
                guidance_scale,
            ],
            outputs=[gallery, status, used_seed],
        )
        clear_button.click(
            fn=clear_gallery,
            outputs=[gallery, status],
            queue=False,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--worker-generate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-choice", choices=list(MODEL_DEFAULTS), help=argparse.SUPPRESS)
    parser.add_argument("--label", choices=list(PROMPTS), help=argparse.SUPPRESS)
    parser.add_argument("--image-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--base-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--inference-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--guidance-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--session-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_worker_generate(args: argparse.Namespace) -> int:
    required = {
        "model_choice": args.model_choice,
        "label": args.label,
        "image_count": args.image_count,
        "base_seed": args.base_seed,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "session_dir": args.session_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        emit_worker_event(
            {
                "event": "error",
                "message": "Argomenti worker mancanti: " + ", ".join(missing),
            }
        )
        return 2

    try:
        worker_generate_images(
            model_choice=args.model_choice,
            label=args.label,
            image_count=int(args.image_count),
            base_seed=int(args.base_seed),
            inference_steps=int(args.inference_steps),
            guidance_scale=float(args.guidance_scale),
            session_dir=Path(args.session_dir),
        )
        return 0
    except Exception as exc:
        emit_worker_event({"event": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    args = parse_args()
    if args.worker_generate:
        raise SystemExit(run_worker_generate(args))

    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
        theme=THEME,
        css=CSS,
    )
