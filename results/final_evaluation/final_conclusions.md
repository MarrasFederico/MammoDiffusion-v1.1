# Conclusioni finali — stato corrente della pipeline

La selezione scientifica è congelata esclusivamente sul validation, ma **non è operativamente completa**. Non viene dichiarato alcun vincitore e non sono state lette metriche test per questa decisione.

Inferenze locked ancora da eseguire: ['raddino_04a_real_only', 'raddino_04b_real_synth', 'mammofm_03d_real_augmented', 'maxvit512_02c_real_synth_full', 'mammofm_03b_real_synth_finetuned', 'maxvit512_02j_real_aug_synth_finetuned', 'mammofm_03a_real_only', 'maxvit512_02a_real_only']. Predizioni Mammo-FM legacy che richiedono reinferenza o accettazione esplicita: ['mammofm_03d_real_augmented', 'mammofm_03b_real_synth_finetuned', 'mammofm_03a_real_only'].

## Blocker

- `raddino_04a_real_only`: test_inference_not_yet_run
- `raddino_04b_real_synth`: test_inference_not_yet_run
- `mammofm_03d_real_augmented`: unverified_prediction_provenance
- `resnet50_01b_real_synth_partial`: Vincitore validation ResNet, ma checkpoint non presente e predizioni test legacy senza provenance completa
- `maxvit512_02c_real_synth_full`: test_inference_not_yet_run
- `mammofm_03b_real_synth_finetuned`: unverified_prediction_provenance
- `maxvit512_02j_real_aug_synth_finetuned`: test_inference_not_yet_run
- `mammofm_03a_real_only`: unverified_prediction_provenance
- `resnet50_01a_real_only`: Checkpoint dichiarato dal notebook ma non presente nel repository; il CSV test legacy non ha provenance sufficiente
- `maxvit512_02a_real_only`: test_inference_not_yet_run

Il confronto finale non può ancora coprire correttamente ResNet-50: il finalista validation ResNet non ha un checkpoint recuperabile. Le predizioni Mammo-FM normalizzate non sono trattate come native.

## Limiti metodologici

Il test era già stato osservato durante lo sviluppo. I futuri risultati devono riportare CI, confronti paired e Holm; nessuna differenza puntuale sarà presentata come vittoria senza supporto statistico. Serve inoltre conferma esterna o grouped cross-validation.
