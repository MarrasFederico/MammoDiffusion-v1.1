# Historical evolution from internal V1 to internal V2

> **Naming note.** V1 and V2 in this document are historical methodological phases of the
> whole-image project. The combined study is frozen publicly as **MammoDiffusion v1.1.0**. These
> references are retained for provenance and must not be confused with a separate lesion-aware
> `MammoDiffusion-v2` successor.

## 1. Project continuity

The internal V2 phase is the direct methodological evolution of the original internal V1
pipeline. Both ask whether class-conditioned, whole-image synthetic mammograms can affect
downstream cancer classification relative to real-only training and traditional augmentation.

Both phases use the same canonical RSNA-derived cohort of 2,916 patients: 486 with `cancer = 1` and
2,430 with `cancer = 0`. These are cancer-positive and non-cancer labels; zero does not establish a
lesion-free image. The one-in-five positive composition is constructed by retaining selected
positive patients and capping negatives at a 5:1 ratio, so it is not source or screening
prevalence. Patient-level train/validation/test splits contain 2,041/437/438 cases and 340/73/73
positives. Each patient contributes one selected 512×512 grayscale MLO image.

The shared patient, image, label, and split keys create continuity, but also an important caveat:
the final test cohort is historically reused rather than a fresh independent replication set.

## 2. What internal V1 established

Internal V1 built the first complete prototype around the RSNA Screening Mammography Breast Cancer
Detection data from a derived 512-pixel PNG release. Its preprocessing selected MLO views, retained
one image per patient, normalized image orientation and intensity, produced padded 512×512 grayscale
images, and created patient-level partitions.

For conditional whole-image generation, it investigated a Stable Diffusion 2.1 fine-tuning path and
a latent diffusion model trained from scratch with a VAE and U-Net. It studied RAW and filtered
generations and reported established image-generation proxies including FID, Inception Score,
precision, recall, density, and coverage.

The downstream study used an ImageNet-initialized ResNet-50 classifier with real-only,
real-plus-synthetic, and synthetic-only conditions, plus traditional-augmentation comparisons. It
also tracked computational sustainability. This first phase connected preprocessing, generation,
filtering, classification, evaluation, and resource accounting in one end-to-end prototype.

## 3. Why the internal V2 phase was developed

The original prototype demonstrated feasibility, but its principal classifier evidence depended on
one seed and one main architecture. It lacked a systematic ensemble analysis and a uniform
multiplicity adjustment across the main downstream contrasts. Threshold selection and
development-time consultation of test-derived outputs were also less strictly separated than in the
final pipeline. These limits do not invalidate the earlier experiments; they restrict their
generalizability and mean the reused test cohort cannot be treated as untouched independent
confirmation.

The original generator evaluation also relied mainly on generic Inception features. Duplicate and
memorization controls were narrower, RAW and filtered pools were not always compared under one
uniform protocol, and some comparisons changed multiple factors. Provenance, cache invalidation,
resume behavior, and executable safety checks were less explicit.

Internal V2 froze a 24-run classifier matrix, retained keyed predictions, separated generator
selection and threshold choice from test evaluation, and applied one analysis framework across a
declared comparison family. The study was not formally preregistered; this freeze makes the final
repository auditable but is not a claim of prospective registration.

## 4. Methodological evolution

| Area | Internal V1 | Internal V2, frozen in release v1.1 |
|---|---|---|
| Classifier architectures | One principal ImageNet-initialized ResNet-50 | MaxViT-512 and Mammo-FM |
| Random seeds | Principal results based on seed 42 | Seeds 17, 42, and 73 |
| Training conditions | Real-only, real+synthetic, synthetic-only; additional traditional augmentation | Real-only, real+traditional augmentation, real+G02 positives, real+G07 positives |
| Run matrix | Primarily single-run comparisons | 2 architectures × 4 conditions × 3 seeds = 24 completed runs |
| Ensembles | None in the principal comparison | Eight mean-probability ensembles |
| Decision thresholds | Validation-derived with less uniform workflow separation | Validation-selected and applied unchanged to test |
| Uncertainty | Not uniform for all principal effects | Patient-level paired, class-stratified bootstrap |
| Comparison family | No single configured family-wise correction | Holm adjustment across eight repository-defined comparisons |
| Generator representation | Generic Inception features | RAD-DINO primary ranking representation; Inception secondary |
| Generator metrics | FID, IS, precision, recall, density, coverage | KID, descriptive FID, PRDC, diversity, technical validity, and similarity diagnostics |
| Synthetic pools | Sizes and rules varied | Equalized 1,361-image official FILTERED pools |
| Duplicate/memorization checks | More limited | Exact/perceptual duplicate and train-reference similarity diagnostics |
| Generator selection | Choices embedded in individual workflows | Validation-only G02/G07 selection by family |
| Test isolation | Earlier development could consult test-derived outputs | Generator selection and threshold optimization exclude test data |
| Configuration | Mainly notebook-local | Versioned classifier/generator protocols, registry, and selection |
| Execution safety | More dependent on notebook/local state | Keyed artifacts, cache invalidation, checkpoint/resume guards, explicit phase flags |
| Tests | No comparable scientific contract suite | Model-free fixtures and static checks for protocol and safety invariants |

The larger run count is useful because the comparison is structured, not because scale alone proves
the result. Multiple seeds expose training variability and mean-probability ensembles reduce
dependence on a single run. Validation-frozen thresholds prevent the current test workflow from
optimizing operating points on test. The two architectures reveal whether a pattern is consistent
across representations, although using the same cohort is not an independent replication and no
formal interaction test is fitted.

The generator benchmark likewise separates proxy endpoints. RAD-DINO is radiology-oriented but not
mammography-specific expert assessment. KID, PRDC, diversity, duplicate, and memorization outputs
describe different properties and do not establish localized cancer content. G02 represents the
fine-tuned family and G07 the from-scratch family.

## 5. Interpretation of the final results

For MaxViT-512, the real-only test ensemble has PR-AUC 0.4128 and the G07 condition 0.5230. The
point delta is +0.1102. In 2,000 paired class-stratified patient bootstrap iterations, the mean
G07-minus-real-only difference is +0.1052 with a 95% percentile interval of [+0.0247, +0.1827].
The empirical two-sided bootstrap tail area is 0.011 and its Holm-adjusted value is 0.088. The null
is not rejected at the configured 0.05 level after adjustment.

Mammo-FM does not reproduce this pattern. Its real-only and G07 point PR-AUC values are 0.3241 and
0.3161. The bootstrap mean difference is −0.0083 with interval [−0.0529, +0.0342], tail area 0.708,
and Holm-adjusted value 1.000. The point estimate is slightly negative and the interval permits both
modest benefit and modest harm.

For the within-architecture generator-family contrasts, G07 minus G02 has bootstrap mean +0.0530
for MaxViT (interval [−0.0067, +0.1149], tail area 0.089, adjusted 0.623) and −0.0087 for Mammo-FM
(interval [−0.0646, +0.0408], tail area 0.755, adjusted 1.000). No comparison in the eight-member
family is rejected after Holm adjustment.

The stored `p_value_two_sided` is twice the proportion of bootstrap differences on the opposite side
of zero from the observed direction. It is an empirical tail-area summary, not a permutation or
null-centered bootstrap-test p-value. This and the absence of formal preregistration support an
exploratory, estimation-first interpretation.

## 6. Why scores are not direct replacements

The shared cohort does not make aggregate scores from the two internal phases interchangeable. The
architecture changes from ResNet-50 to MaxViT-512 and Mammo-FM; the final phase uses three seeds and
ensembles; and condition definitions, threshold policy, analysis, and generator representations
differ. PR-AUC is threshold-independent; frozen thresholds govern operating-point metrics and test
discipline rather than the numerical PR-AUC difference.

Training exposure also differs among the final conditions. Traditional augmentation adds 1,020
positive transforms, while G02 and G07 each add 1,361 positives. Those changes alter training-set
size and balance, and early stopping yields 2,750–6,400 actual optimizer updates despite a common
6,400 maximum. Comparisons against real-only do not isolate augmentation modality from dose,
composition, exposure, and realized compute.

## 7. Remaining scientific limits

Only one MLO image per patient is used, with no CC view, bilateral context, or external cohort.
Global cancer labels provide no lesion box or mask, so neither generated pathology nor attribution
localization can be verified quantitatively. The saved Grad-CAM and Integrated Gradients panels are
qualitative. There is no systematic blinded radiologist evaluation.

Whole-image generation jointly synthesizes anatomy, acquisition appearance, background, and any
pathological signal. A classifier can potentially learn generator texture, preprocessing artifacts,
or synthetic-domain shortcuts. Passing generator gates or obtaining favorable feature metrics does
not rule out these mechanisms, and distribution similarity is not equivalent to downstream or
clinical validity.

The operating point targeted to validation specificity 0.90 is frozen before test. Its achieved
test specificity can differ from 0.90 and must be reported rather than implied. Results cannot be
assumed to transfer to other institutions, devices, populations, views, or diagnostic workflows.

See [LIMITATIONS.md](LIMITATIONS.md) for the complete claim boundary.

## 8. Final interpretation

Release v1.1.0 strengthens the original whole-image prototype into a more controlled and
reproducible study. Its clearest downstream observation is a favorable but non-conclusive MaxViT
signal that is not reproduced by Mammo-FM and is not significant after the configured Holm
adjustment. This is neither proof that synthetic data generally help nor evidence that the project
failed. It is a precise boundary on the claim and a motivation to test lesion-aware synthesis that
preserves real anatomy and exposes localized pathology controls.
