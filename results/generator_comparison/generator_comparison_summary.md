# Confronto generatori — sintesi

Generatori con dati classe positiva: 8/8.
Generatori con confronto due-classi completo: 4.
Generatori con metriche incomplete (esclusi dalla media, mai stimati): ['01_sd21_baseline_50steps', '06_ldm_extra1361_fromscratch', '07_ldm_sdvae_extra1361'].

Nessun vincitore downstream e' dichiarato qui: la selezione dei generatori per Stage 2 avviene esclusivamente su validation dei classificatori (scripts/finalize_validation_stage.py --stage 1), mai da FID/IS/PRDC da soli e mai dal test set.

Tabelle: results/generator_comparison/tables/*.{csv,json}.
