# Limitations of the MammoDiffusion v1.1 whole-image study

This document defines the boundary of the claims supported by release v1.1.0. It is not a list of
failed objectives: the release contributes a complete and auditable synthesis-to-classification
pipeline. The limitations explain why its favorable MaxViT signal remains a hypothesis for future
work rather than evidence of a general clinical benefit.

## 1. Constructed, positive-enriched analytical cohort

The canonical cohort contains 2,916 patients: 486 with `cancer = 1` and 2,430 with `cancer = 0`.
Preprocessing retains selected positive patients and caps the negative-to-positive ratio at 5:1.
The resulting 16.7% positive fraction is therefore an experimental sampling choice, not the source
dataset prevalence and not a screening-population prevalence estimate. Training contains only 340
positive patients; validation and test contain 73 each.

## 2. One primary dataset and no external validation

All canonical downstream results come from one RSNA-derived cohort. No independent institution,
vendor, acquisition protocol, population, or dataset is used for external validation. Domain
generalization and clinical transportability are not established.

## 3. Historically reused test cohort

The same 438-patient test split is retained from the earlier V1 prototype. The
current code prevents generator selection and operating-point optimization from using test data,
but earlier project development had less strict separation and could consult test-derived outputs.
The final test results are therefore a controlled evaluation on a held-out project split, not a
fresh independent replication cohort.

## 4. Restricted image representation

The study uses one selected 512×512 grayscale MLO image per patient. It does not model the usual
bilateral, CC-plus-MLO clinical reading context, cannot measure multiview consistency, and may lose
fine detail through resizing and padding. Results do not directly apply to full-resolution DICOM or
multiview diagnostic workflows.

## 5. Endpoint is cancer versus non-cancer, not lesion versus no lesion

The target is the RSNA `cancer` label. A label of zero does not prove the absence of benign,
suspicious, or unlabeled findings. The manifests have no lesion bounding boxes, segmentations,
position, size, morphology, pathology, or histology fields adequate for lesion-level validation.
Calling the task “malignant finding versus no lesion” would overstate the annotation.

## 6. Whole-image synthesis entangles anatomy, acquisition, and pathology

The generators synthesize the entire mammogram. They can change anatomy, acquisition appearance,
background texture, padding, and preprocessing characteristics together with any cancer-associated
content. This design cannot isolate whether downstream changes arise from pathological information
or from global synthetic-domain changes.

## 7. Synthetic and preprocessing shortcuts remain possible

A classifier may exploit generator texture, denoising signatures, frequency artifacts, borders,
orientation normalization, contrast transformations, or other source cues rather than pathology.
Duplicate and train-reference memorization checks reduce specific risks but do not rule out broader
synthetic-origin or preprocessing shortcuts.

## 8. Generator proxy metrics do not establish downstream or clinical validity

KID, FID, PRDC, RAD-DINO features, diversity, duplicate rates, and nearest-neighbour diagnostics
measure different proxies. Passing the configured eligibility gates means only that a candidate can
enter this project's ranking. It does not prove lesion presence, label adherence, radiological
plausibility, diagnostic safety, or downstream benefit. The favorable downstream condition is not
simply the candidate with a clinically validated lesion signal.

## 9. No systematic blinded radiologist study

There is no adequately powered, blinded, multi-reader evaluation of anatomy, acquisition
plausibility, lesion presence, lesion type, or diagnostic realism. Model embeddings and classifier
responses are not substitutes for qualified radiological assessment.

## 10. Architecture-dependent downstream effect

The G07 condition raises MaxViT-512's point PR-AUC but not Mammo-FM's. The second architecture is
evaluated on the same patients and is not an independent dataset replication. No formal
architecture-by-condition interaction test was performed. The available evidence therefore does
not support a classifier-independent augmentation effect.

## 11. Augmentation dose and composition are confounded

Real-only training has 2,041 images. Traditional augmentation adds 1,020 transformed positive
images (3,061 total), whereas each synthetic condition adds 1,361 positives (3,402 total). The
synthetic conditions balance the training labels at 1,701/1,701; traditional augmentation yields
1,701/1,360. Comparisons against real-only or traditional augmentation therefore combine the
augmentation method with dataset size, class balance, sample reuse, and exposure. G02 versus G07
uses the same nominal synthetic dose and isolates generator family more cleanly, but not every other
comparison isolates synthesis modality alone. No dose-response experiment was run.

## 12. Maximum training budget is fixed; actual compute is not equal

All jobs share a maximum of 6,400 optimizer updates and common within-architecture rules, but early
stopping ended 23 of 24 runs before that cap. Actual runs span 2,750 to 6,400 optimizer updates.
Dataset sizes also differ by condition, so the same update count would not imply the same number of
effective passes over unique images. The protocol controls an upper budget and stopping rule, not
identical realized compute.

## 13. Statistical evidence is exploratory

The repository defines eight primary PR-AUC comparisons and applies Holm adjustment. The study was
not formally preregistered. Values stored as `p_value_two_sided` are empirical bootstrap tail areas:
twice the fraction of paired, class-stratified bootstrap differences on the opposite side of zero
from the observed direction. They are not permutation or bootstrap-null p-values. Percentile
intervals quantify resampling uncertainty within this selected cohort but do not remove selection,
development-history, or external-validity limitations. No adjusted comparison is significant at
0.05.

## 14. Nominal target specificity does not transfer exactly

The threshold associated with target specificity 0.90 is selected on validation and then frozen.
On test, the code reports the achieved specificity at that threshold; it need not equal or exceed
0.90 because of sampling variation, score distribution shift, and discrete thresholds. These values
must not be reported as test sensitivity “at exactly 90% specificity” without the achieved test
specificity beside them.

## 15. Interpretability is qualitative and not lesion localization

The classifier notebooks save Grad-CAM and Integrated Gradients panels for selected validation
cases. There are no lesion masks with which to compute attribution overlap, and no quantitative
lesion-localization endpoint. The panels can generate hypotheses about model attention but cannot
show that a classifier used the cancer finding, nor can they validate synthetic lesion placement.

## 16. Sustainability estimates have a narrow boundary

The retained analysis covers generator workflows, not classifier training. Energy is reconstructed
as `elapsed_seconds × 0.170 kW` for one GPU because the recorded CodeCarbon energy and CO2 fields are
not trusted for the RTX 5060 Ti. It is not a wall-socket measurement, lifecycle analysis, or carbon
estimate and does not enter generator selection.

## 17. Portability depends on separately archived assets

The public Git repository excludes image data, checkpoints, local model encoders, and other heavy
runtime artifacts. Mammo-FM weights are subject to a separate academic license and cannot be
redistributed. Some frozen manifests and results contain absolute paths from the original
workstation; utilities reroot recognized project paths, but those legacy strings remain provenance,
not portable locations. A clone supports audit and lightweight report regeneration, while full
training and generation require the documented external/local assets.

## Consequence for the next research step

The most direct response is not merely a larger whole-image generator. A lesion-aware inpainting
design can preserve real target-domain anatomy outside a controlled region, expose lesion masks for
quantitative attribution and preservation tests, add sham-inpainting controls, and test whether the
augmentation effect becomes more consistent across classifiers and external datasets. That is a
scientific motivation, not a claim that inpainting will necessarily solve these limitations.
