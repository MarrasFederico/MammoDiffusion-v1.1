"""Local Gradio demo for the MammoDiffusion fine-tuned generator."""

from __future__ import annotations

import argparse
import secrets
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL.PngImagePlugin import PngInfo


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "20260607_sd21_rsna_mlo_512"
BASE_MODEL_DIR = (
    EXPERIMENT_DIR / "pretrained_model" / "stable-diffusion-2-1-base"
)
CHECKPOINT_DIR = EXPERIMENT_DIR / "model" / "checkpoint-3000"
OUTPUT_DIR = APP_DIR / "outputs"

DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 7.5
RESOLUTION = 512
MAX_IMAGES = 12

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

PIPELINE: StableDiffusionPipeline | None = None
PIPELINE_LOCK = threading.Lock()


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


def validate_model_paths() -> None:
    required = [
        BASE_MODEL_DIR / "model_index.json",
        CHECKPOINT_DIR / "unet" / "config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Modello fine-tuned non trovato. File richiesti mancanti:\n" + details
        )


def get_pipeline() -> StableDiffusionPipeline:
    """Load the selected validation checkpoint once, on first generation."""
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    with PIPELINE_LOCK:
        if PIPELINE is not None:
            return PIPELINE

        validate_model_paths()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        unet = UNet2DConditionModel.from_pretrained(
            str(CHECKPOINT_DIR / "unet"),
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            str(BASE_MODEL_DIR),
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

        PIPELINE = pipeline
        return PIPELINE


def prompt_for_label(label: str) -> str:
    return PROMPTS[label]


def make_output_session(label: str, base_seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = OUTPUT_DIR / f"{timestamp}_{LABEL_KEYS[label]}_seed-{base_seed}"
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
    label: str,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
) -> None:
    metadata = PngInfo()
    metadata.add_text("class", LABEL_KEYS[label])
    metadata.add_text("prompt", prompt)
    metadata.add_text("seed", str(seed))
    metadata.add_text("checkpoint", CHECKPOINT_DIR.name)
    metadata.add_text("inference_steps", str(steps))
    metadata.add_text("guidance_scale", str(guidance_scale))
    image.save(destination, pnginfo=metadata)


def generate_images(
    label: str,
    image_count: int,
    requested_seed: int,
    inference_steps: int,
    guidance_scale: float,
):
    """Generate sequentially and stream each completed image to the gallery."""
    prompt = PROMPTS[label]
    image_count = max(1, min(int(image_count), MAX_IMAGES))
    base_seed = -1 if requested_seed is None else int(requested_seed)
    if base_seed < 0:
        base_seed = secrets.randbelow(2**31)

    gallery_items = []
    yield gallery_items, "Caricamento del modello fine-tuned...", base_seed

    try:
        pipeline = get_pipeline()
        device = pipeline.device.type
        session_dir = make_output_session(label, base_seed)

        with PIPELINE_LOCK, torch.inference_mode():
            for index in range(image_count):
                image_seed = base_seed + index
                generator = torch.Generator(device=device).manual_seed(image_seed)

                result = pipeline(
                    prompt,
                    num_inference_steps=int(inference_steps),
                    guidance_scale=float(guidance_scale),
                    height=RESOLUTION,
                    width=RESOLUTION,
                    generator=generator,
                )
                image = result.images[0]
                destination = session_dir / f"{LABEL_KEYS[label]}_{index + 1:02d}.png"
                save_image(
                    image,
                    destination,
                    label,
                    prompt,
                    image_seed,
                    int(inference_steps),
                    float(guidance_scale),
                )
                gallery_items.append((image, f"{label} · seed {image_seed}"))
                yield (
                    gallery_items,
                    f"Generate {index + 1}/{image_count} immagini...",
                    base_seed,
                )

        yield (
            gallery_items,
            f"Completato. File salvati in `{display_path(session_dir)}`",
            base_seed,
        )
    except Exception as exc:
        yield gallery_items, f"Errore durante la generazione: `{exc}`", base_seed


def clear_gallery():
    return [], "Galleria pulita. I file già generati restano salvati."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MammoDiffusion Studio") as demo:
        gr.HTML(
            """
            <div id="md-header">
                <h1>MammoDiffusion Studio</h1>
                <p>Generazione condizionata con Stable Diffusion 2.1 fine-tuned · checkpoint-3000</p>
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
                value=PROMPTS["Positiva"],
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
                    value=DEFAULT_STEPS,
                    step=1,
                    label="Inference steps",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=DEFAULT_GUIDANCE,
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

        label.change(
            fn=prompt_for_label,
            inputs=label,
            outputs=prompt_preview,
            queue=False,
        )
        generate_button.click(
            fn=generate_images,
            inputs=[label, image_count, seed, inference_steps, guidance_scale],
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
        theme=THEME,
        css=CSS,
    )
