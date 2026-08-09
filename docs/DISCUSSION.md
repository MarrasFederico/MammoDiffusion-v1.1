# Discussion

## What MammoDiffusion v1.1 establishes

MammoDiffusion v1.1 establishes that a whole-image mammography synthesis study can be organized as
an auditable chain from patient-level preprocessing through generator benchmarking, synthetic-pool
selection, multi-seed downstream training, validation-frozen operating points, saved predictions,
paired resampling, and deterministic report regeneration. This infrastructure is a substantive
result: the generator choice, test isolation rules, prediction keys, and uncertainty calculations
can be inspected without rerunning expensive model training.

The release also produces a scientifically useful negative constraint. Whole-image synthetic
augmentation does not have one stable effect across the two tested classifier representations.
That observation narrows the claim and identifies the next experiment more clearly than a result
from a single classifier would.

## Interpreting the downstream signal

On the held-out 438-patient project test split, MaxViT-512 PR-AUC changes from 0.4128 with real-only
training to 0.5230 with G07 positives. The point delta is +0.1102. Paired class-stratified bootstrap
resampling gives a mean delta of +0.1052 and a percentile interval of [+0.0247, +0.1827]. The
empirical two-sided bootstrap tail area is 0.011; after Holm adjustment across the eight configured
comparisons it is 0.088.

The corresponding Mammo-FM point estimates are 0.3241 and 0.3161. Its point delta is −0.0079 and
bootstrap mean delta −0.0083, with interval [−0.0529, +0.0342], tail area 0.708, and Holm-adjusted
value 1.000. Thus the MaxViT observation is a favorable architecture-specific signal, while the
Mammo-FM result is compatible with modest benefit or harm and has a slightly negative point
estimate.

The two architectures are complementary representations, but their comparison is not an
independent replication: they use the same cohort, labels, and split. The study also does not fit a
formal architecture-by-condition interaction. “Architecture-dependent” is therefore a descriptive
summary of the observed pattern, not a quantified interaction effect.

## Statistical scope

The repository calls eight PR-AUC contrasts primary and applies one Holm family. This family is
defined in the frozen executable protocol, but the study was not formally preregistered. The stored
`p_value_two_sided` values are calculated from the observed bootstrap distribution rather than a
null-centered permutation or bootstrap distribution. They should be described as empirical
two-sided tail areas, not as conventional confirmatory p-values.

This distinction does not make the bootstrap uninformative. Pairing preserves patient-level
correlation between conditions, class-stratified resampling avoids class-degenerate samples, and
percentile intervals show sampling variability within the cohort. It does mean that the exact
tail-area and adjusted values should not carry more evidential weight than the design supports.
Most importantly, no Holm-adjusted comparison is below 0.05.

## Why MaxViT and Mammo-FM may differ

Several mechanisms are compatible with the observed divergence:

- the architectures and their pretraining may respond differently to texture, frequency content,
  global context, or preprocessing;
- synthetic positives may alter optimization or class balance in ways that interact with the
  representation;
- whole-image generation may introduce source cues that are useful to one classifier but ignored
  or penalized by another;
- the small number of positive test patients may make architecture-specific point estimates
  unstable;
- training stopped at different update counts under the common early-stopping rule.

The present experiment cannot choose among these explanations. Grad-CAM and Integrated Gradients
panels are qualitative and lack lesion masks, so they cannot demonstrate that the MaxViT gain came
from attention to valid pathological content.

## Generator quality is not equivalent to downstream utility

G02 and G07 were selected using a RAD-DINO-KID-first hierarchy with distribution, diversity,
duplicate, technical-validity, and detected-memorization diagnostics. Those measurements are useful
quality-control proxies, but they do not establish localized cancer content or clinical realism.
Passing the gates means that a candidate is eligible for this project's ranking; it is not a
radiological certificate.

The downstream comparison reinforces this separation. A synthetic pool can be close to a reference
distribution yet have no stable classifier benefit, while a classifier change can arise from
global domain cues rather than pathology. Generator evaluation should therefore retain both proxy
metrics and downstream tests without treating either as sufficient evidence of clinical validity.

## Dataset composition and compute

The comparison matrix controls architecture, seeds, maximum update budget, optimizer family, loss,
and selection rules within each architecture. It does not equalize every causal factor. Traditional
augmentation adds 1,020 images, whereas each synthetic condition adds 1,361. The resulting training
set size and positive balance differ, and 23 of 24 runs terminate early between 2,750 and 6,250
updates; one reaches the 6,400-update cap. Accordingly, real-only versus augmented contrasts mix
augmentation content with dose, balance, exposure, and realized compute. The equal-dose G02-versus-
G07 contrast is cleaner for generator family, but it also yields no adjusted finding.

## Why this is not a failed study

The project delivers a complete whole-image baseline, exposes rather than hides non-replication,
and identifies which claim is not yet supported. It would be misleading to convert the MaxViT
signal into a general success claim, but it would be equally misleading to treat the lack of
cross-classifier consistency as a valueless result. The design shows that classifier choice is a
material component of synthetic-data evaluation.

## Scientific motivation for lesion-aware work

Whole-image synthesis jointly generates anatomy, acquisition characteristics, background, and any
pathological signal. A lesion-aware successor can test a more specific mechanism: retain a real
target-domain mammogram and modify only a known region. Hard compositing can enforce zero pixel
change outside the mask; sham inpainting can test whether the classifier detects an editing
signature; known masks can support quantitative attribution overlap and localization; and external
datasets can test whether pathology transfers without importing their global acquisition domain.

These controls directly target the principal ambiguity of v1.1, but they are hypotheses for future
evaluation. Lesion inpainting could still introduce boundary artifacts, unrealistic morphology, or
new shortcuts. It must be compared with real-only, traditional augmentation, external real pooling,
whole-image synthesis, and sham editing at matched augmentation doses, with patient-level splits,
external validation, and blinded radiological review where feasible.

See [LIMITATIONS.md](LIMITATIONS.md) for the complete claim boundary and
[PROTOCOL.md](PROTOCOL.md) for the executable-study details.
