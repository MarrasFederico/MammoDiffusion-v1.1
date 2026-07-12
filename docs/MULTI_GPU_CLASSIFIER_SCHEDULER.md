# Multi-GPU classifier scheduler

`notebooks/utility/classifier_gpu_scheduler.py` identifies GPUs by **name** (logged alongside
their UUID), never by CUDA index. This matters concretely on the target machine: `nvidia-smi`
reports index 0 as the RTX 3060 and index 1 as the RTX 5060 Ti — the reverse of how the two cards
are casually labelled "GPU 0 / GPU 1" in planning documents. A job is launched with
`CUDA_VISIBLE_DEVICES` pinned to a single physical device resolved by UUID at dispatch time; it
never sees both GPUs.

## Targets (soft caps, not guarantees)

```json
{"rtx_5060_ti_16gb": {"target_max_jobs": 3, "reserve_vram_mb": 1800},
 "rtx_3060_12gb": {"target_max_jobs": 2, "reserve_vram_mb": 1500}}
```

A job is admitted only if `free_vram >= estimated_peak * 1.20 + reserve` **and** the GPU's slot
count is under its target. `estimated_peak` comes from `results/runtime_profiles/
classifier_vram_profiles.json` (written by `scripts/profile_classifier_vram.py`, a real
forward/backward probe per architecture) when available, otherwise a conservative
per-resource-profile fallback (light=3000MB, medium=6000MB, heavy=11000MB, exclusive=15000MB)
that deliberately under-packs rather than risks systematic OOM. Verified against the real
machine in dry-run: with no VRAM profile yet recorded, the scheduler currently admits only 1-2
jobs total from a 336-job Stage 1 plan — this is the safe default, not a bug; running the VRAM
probe first is what unlocks real concurrency.

`HOST_MAX_CONCURRENT_JOBS = 5` caps total jobs across both GPUs regardless of free VRAM, to avoid
oversubscribing dataloader workers/CPU threads (default 3 workers, 4 threads per job).

## Resource profiles

Derived per architecture from `classifier_training_protocols.json`'s
`resource_profile_by_phase` (the heaviest declared phase governs the whole job): ResNet-50 is
`medium` (light head-training phase, medium fine-tuning phase), MaxViT-512 is `heavy`, Mammo-FM
and RAD-DINO are `medium`. An `exclusive` job only ever starts on a fully idle eligible GPU.

## OOM handling

`notebooks/utility/classifier_gpu_scheduler.OomState`: first OOM halves the physical batch size
and doubles gradient accumulation (effective batch size is asserted unchanged), then retries
once; a second OOM stops retrying and forces the job to `exclusive`. Every adjustment is recorded
in `.history`, never applied silently.

## Verification performed this session

- Real `nvidia-smi` query against the target machine: both GPUs correctly classified by name.
- Dry-run of `scripts/run_classifier_experiment_matrix.py` against the real 336-job Stage 1
  matrix: correct admission decisions, zero subprocesses launched.
- 13 unit tests with mocked GPU/VRAM data covering 3-slot/2-slot admission, VRAM rejection,
  exclusive-profile gating, host job cap, release-then-readmit, priority ordering, and the OOM
  state machine (`tests/test_classifier_gpu_scheduler.py`).
- `scripts/profile_classifier_vram.py` itself was **not** executed — it performs real GPU
  training work and is listed as a follow-up command, not part of this session's verification.
