#!/usr/bin/env python3
"""Run one manually selected downstream training/validation job with checkpoint/resume."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "notebooks/utility"


def configure_gpu(gpu: int | None) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None and visible:
        raise ValueError("use either --gpu or CUDA_VISIBLE_DEVICES, not both")
    if gpu is None and not visible:
        raise ValueError("select a GPU with --gpu or CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("maxvit512", "mammofm"), required=True)
    parser.add_argument("--condition", choices=("real_only", "real_augmented", "real_plus_best_finetuned_positive", "real_plus_best_fromscratch_positive"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 73), required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--plan", action="store_true", help="resolve and display the job without training")
    parser.add_argument("--tiny", action="store_true", help="dependency-free synthetic fixture mode; never for scientific results")
    parser.add_argument("--allow-retrain", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    configure_gpu(args.gpu)
    sys.path.insert(0, str(UTILITY))
    from classifier_experiment_runner import plan, run_auto
    function = plan if args.plan else run_auto
    kwargs = {} if args.plan else {"tiny": args.tiny, "allow_retrain": args.allow_retrain, "resume": not args.no_resume}
    result = function(ROOT, args.architecture, args.condition, args.seed, **kwargs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
