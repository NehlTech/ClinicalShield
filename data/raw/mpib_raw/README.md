---
license: cc-by-nc-4.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- medical
- safety
- adversarial
- benchmark
size_categories:
- 10K<n<100K
---

# MPIB (Medical Prompt Injection Benchmark)

Release version: `v1.1`

Canonical dataset version: `v1.1`

MPIB is a comprehensive benchmark for evaluating the safety and robustness of medical Large Language Models (LLMs) against prompt injection attacks. It contains **9,697** clinically grounded adversarial samples derived from MedQA and PubMedQA.

## Dataset Structure

The dataset is partitioned into three splits:
- `train` (80%): 7,759 samples for training or few-shot exemplars.
- `validation` (10%): 969 samples for tuning.
- `test` (10%): 969 samples for final evaluation.

## Reproducibility Note

This repository mimics a **Gated Access** (Tier 1) environment.
- **Public**: V2 payloads are redacted (`[REDACTED_PAYLOAD]`) for immediate safety.
- **Restricted**: Approved researchers can access the full **Payload Registry** at `data/restricted/`.
- **Reconstruction**: By providing the registry file to our [evaluation toolkit](https://github.com/jhlee0619/mpib-eval), you can restore exact functional attacks.

## Citation

If you use MPIB in your research, please cite our paper:

```bibtex
@misc{lee2026mpibbenchmarkmedicalprompt,
      title={MPIB: A Benchmark for Medical Prompt Injection Attacks and Clinical Safety in LLMs},
      author={Junhyeok Lee and Han Jang and Kyu Sung Choi},
      year={2026},
      eprint={2602.06268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06268},
}
```

## Disclaimer

This dataset contains adversarial examples designed to test safety boundaries. The medical information in "poisoned" contexts is intentionally fabricated or distorted and **MUST NOT** be used for actual clinical decision-making.
