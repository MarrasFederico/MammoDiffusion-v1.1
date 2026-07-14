# Experimental protocol summary

RQ1 is answered by the validation-only unified generator benchmark. The real reference count is all available validation positives; 1,361 is the synthetic pool target. KID and PRDC use balanced repeated subsampling without replacement, FID is secondary, and train memorization is separate from validation similarity.

RQ2 and RQ3 use MaxViT-512 and Mammo-FM across four conditions and seeds 17, 42 and 73. The 24 experiments share a fixed optimizer-update budget within architecture. Checkpointing, early stopping and scheduling use validation PR-AUC.

The three-seed mean-probability ensembles and condition comparisons use validation only and patient-level statistics. Eight primary PR-AUC comparisons receive Holm correction. Final evaluation is optional, guarded by a visible Boolean and readiness checklist, and documented by a plain protocol snapshot. Historical reuse of the prior internal test is disclosed.
