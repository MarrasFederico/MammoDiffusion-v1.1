#!/usr/bin/env bash
# Package a lightweight, code+test+config audit subset of the repository: everything needed to
# import notebooks/utility, run the three unit test suites, and inspect the locked classifier
# pipeline artifacts -- without any heavy model checkpoints or generated images. Intended to let
# someone verify the code/tests without cloning or shipping the full (multi-GB) repository.
#
# Usage: scripts/package_code_audit.sh [output_path]
#   output_path defaults to dist/code_audit_package.tar.gz (relative to the repo root).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-dist/code_audit_package.tar.gz}"
if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$REPO_ROOT/$OUTPUT"
fi
mkdir -p "$(dirname "$OUTPUT")"

cd "$REPO_ROOT"

# Minimum set required for the code+tests to be importable and runnable, per the packaging spec:
# README.md, configs/, notebooks/, results/final_evaluation/, tests/. data/processed/metadata/
# is also included (small CSVs only, ~1MB) because even DRY_RUN=True notebook cells read the
# canonical test.csv for a row count -- without it the locked notebooks can't be dry-run from the
# package at all.
INCLUDE_PATHS=(
    README.md
    configs
    notebooks
    results/final_evaluation
    tests
    scripts
    requirements.txt
    data/processed/metadata
)

for path in "${INCLUDE_PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "package_code_audit.sh: missing expected path '$path'" >&2
        exit 1
    fi
done

tar --create --gzip \
    --file "$OUTPUT" \
    --exclude=".git" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    --exclude="notebooks/pretrained_model" \
    --exclude="*.safetensors" \
    --exclude="*.keras" \
    --exclude="*.pt" \
    --exclude="*.pth" \
    --exclude="*.ckpt" \
    --exclude="*.bin" \
    "${INCLUDE_PATHS[@]}"

echo "Wrote $OUTPUT"
du -h "$OUTPUT"
