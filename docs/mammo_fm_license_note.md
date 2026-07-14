# Mammo-FM academic-license compliance

Mammo-FM model weights are governed by the repository's **Custom Academic License for Model Weights**, not by this project's source-code license. Before every use, authorized researchers must read the current full license at <https://huggingface.co/batmanLab/Mammo-FM/blob/main/LICENSE>.

For this academic study, the operational rules are:

- non-commercial academic research only, by permitted academic/non-profit users;
- no clinical, diagnostic, treatment, medical-decision, product, service, revenue-generating, industry-research, hosted API, SaaS, or other remote-access use;
- no redistribution of original weights, fine-tuned weights, partial weights, derived checkpoints, archives, or derivative models;
- no distillation, compression, extraction, imitation, or transfer intended to create or improve another model;
- derivatives remain private, academic-only, and subject to the same license;
- obtain prior written permission from the licensor for any exception.

The public repository may contain loading/training code, configurations, hashes, aggregate non-sensitive outputs, and instructions for authorized users. It must not contain Mammo-FM checkpoints. Set `MAMMOFM_LOCAL_CHECKPOINT_PATH` to an authorized local file; do not add that file to Git.

Required acknowledgment:

> This work uses Mammo-FM developed by the authors of Mammo-FM, Boston University.

Required publication citation:

```bibtex
@article{ghosh2025mammo,
  title={Mammo-FM: Breast-specific foundational model for Integrated Mammographic Diagnosis, Prognosis, and Reporting},
  author={Ghosh, Shantanu and Joshi, Vedant Parthesh and Syed, Rayan and Kassem, Aya and Varshney, Abhishek and Basak, Payel and Dai, Weicheng and Gichoya, Judy Wawira and Trivedi, Hari M and Banerjee, Imon and others},
  journal={arXiv preprint arXiv:2512.00198},
  year={2025}
}
```

License/model card: <https://huggingface.co/batmanLab/Mammo-FM>. Paper: <https://arxiv.org/abs/2512.00198>.
