# Legacy recovery report

The legacy source was found at `/mnt/MammoDiffusion/Versione vecchia` (the prompt's expected words
were reversed). It was audited read-only. `legacy_recovery_inventory.csv` records every examined
file with target, byte size, SHA-256, provenance and decision; the Markdown companion summarizes
the classifications.

No `baseline_resnet50_final_best.keras`, `real_synth_resnet50_final_best.keras`, or equivalent
ResNet50 finalist checkpoint was found in either tree. Consequently 01a/01b remain blocked, no
checkpoint was copied, no historical metric was rebound to a different model, and no signature was
invented. The legacy tree does contain large generator checkpoints and two full older SD2.1 runs,
but their names alone are insufficient provenance and current generator artefacts are newer and
complete. They remain `legacy_unverified` and were not copied.

The pre-change safety snapshot is outside the repository in the newest sibling directory named
`pre_sol_recovery_<timestamp>`. It contains Git status/diff, both file listings and critical legacy
checksums. The legacy source itself was never modified.
