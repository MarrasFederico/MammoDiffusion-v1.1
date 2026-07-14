#!/usr/bin/env python3
"""Optional local GPU construction/forward smoke; it creates no scientific certificate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "notebooks/utility"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("maxvit512", "mammofm"), required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.gpu is not None and os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("use either --gpu or CUDA_VISIBLE_DEVICES, not both")
    if args.gpu is None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("select a GPU with --gpu or CUDA_VISIBLE_DEVICES")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(UTILITY))
    from classifier_architecture_adapters import get_adapter
    from downstream_protocol import load_protocol
    policy = load_protocol(ROOT)["architectures"][args.architecture]
    adapter = get_adapter(args.architecture, policy, ROOT)
    model = adapter.build_model(pretrained=True, seed=17)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU smoke requires CUDA")
    model.eval().to(device)
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 512, 512), device=device))
    print(json.dumps({"status": "smoke_ready", "architecture": args.architecture,
                      "model_type": type(model).__name__, "certificate_created": False,
                      "output_shape": list(output.shape),
                      "memory_profile": adapter.estimate_memory_profile()}, indent=2))


if __name__ == "__main__":
    main()
