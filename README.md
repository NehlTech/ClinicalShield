# ClinicalShield

**A Multi-Layer Defence Framework for Detecting and Neutralizing Indirect Prompt Injection Attacks in RAG-Based Clinical Decision Support Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Venue: Scientific Reports](https://img.shields.io/badge/Venue-Scientific%20Reports-red.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Paper:** ClinicalShield: A Multi-Layer Defence Framework for Detecting and Neutralizing Indirect Prompt Injection Attacks in RAG-Based Clinical Decision Support Systems
> **Authors:** Adu-Boahene Bright, Oliver Kornyo, Seiba Alhassan, Siddique Abubakr Muntaka, Yaw Afriyie
> **Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Ghana
> **Correspondence:** baduboahene@st.knust.edu.gh

---

## Overview

Clinical Decision Support Systems (CDSSs) increasingly combine Large Language Models with Retrieval-Augmented Generation (RAG) pipelines, retrieving documents from pharmacological databases, clinical guidelines and adverse event registries before generating recommendations. This architecture introduces a critical vulnerability: every retrieved document is a potential attack surface.

**Indirect Prompt Injection** exploits this by embedding malicious instructions inside documents that the RAG retriever is likely to surface — without any direct access to the model, the user session, or the hospital system. A single poisoned drug label can cause a clinical AI to recommend a contraindicated medication, suppress an allergy warning, or override a dosing guideline.

ClinicalShield is the first defence framework built specifically for this threat in a clinical setting. It processes every retrieved document through five sequential modules that detect encoded and disguised payloads, verify document intent, remove adversarial content while preserving clinical information, and verify drug-property claims against FDA labelling — all without discarding the clinical content the physician needs.

---

## Key Results

| Metric | Value |
|--------|-------|
| Detection accuracy | 0.9726 ± 0.0123 |
| Detection recall | 0.9427 ± 0.0281 |
| False positive rate | 0.0011 ± 0.0024 |
| AUC-ROC | 0.9987 ± 0.0011 |
| Encoding recovery (all 5 formats) | 100% |
| Adversarial variant recovery (PyRIT suite) | 73.8% |
| Clinical entity preservation (CEPS) | 0.6066 |
| End-to-end attack success (poisoned) | 23.0% |
| End-to-end attack success (after sanitisation) | 6.0% |
| Relative ASR reduction | **73.9%** (McNemar *p* = 5.4 × 10⁻⁹) |

All detection figures are mean ± standard deviation across five independent training runs on a 701-passage held-out partition. The corpus contains 5,353 passages from 336 drugs across 1,604 FDA label sections.

---

## Architecture

```
Query (q) ──┐
            ├──► Module 1: Encoding-Aware Ingestion
RAG KB ─────┘    (canonicalisation → format decoding)
                         │
                         ▼
              Module 2: PubMedBERT Detection
              (document + sentence-level scores)
                    │           │
              Attack?          No
                 │              └──► Safe clinical output → LLM
                Yes
                 │
              ┌──┴───────────────────────────┐
              │  Module 3: Intent Verification│
              │  (cosine similarity + markers)│
              └──────────────────────────────┘
                         │
                         ▼
              Module 4: Utility-Preserving Sanitisation
              (sentence-level removal · CEPS validation)
                         │
              Module 5: Clinical Fact Verification (optional)
              (drug claims · openFDA API)
                         │
                         ▼
                    Policy Layer
                ┌────┬────────────────┐
             Release  Sanitise    Withhold
                │     & release      │
                ▼        ▼           ▼
          Safe output    Safe    Blocked
```

### Module descriptions

| Module | Function |
|--------|----------|
| **Module 1** | Canonicalises disguised text (invisible characters, homoglyphs, letter spacing, cipher text) then decodes five obfuscation formats: Base64, hexadecimal, Unicode escapes, URL encoding, Leetspeak. Auto-flags encoded inputs for downstream processing. |
| **Module 2** | Fine-tunes PubMedBERT for binary injection detection. Attention weights localise the adversarial span within the document. Exposes per-sentence scores consumed by Module 4. |
| **Module 3** | Computes cosine similarity between the clinical query and each retrieved document using PubMedBERT embeddings. Checks for 21 override marker phrases. Flags documents whose intent diverges from the query. |
| **Module 4** | Removes adversarial content at sentence level using Module 2's scores, override-marker detection, and entity-aware protection. Validates the result with the Clinical Entity Preservation Score (CEPS). |
| **Module 5** | Extracts drug-property claims from the sanitised document and checks them against openFDA structured labelling. Flags contradictions and unverifiable claims separately. *(Optional)* |

---

## Clinical Entity Preservation Score (CEPS)

CEPS is a dual metric introduced in this work to measure sanitisation quality. It pairs entity preservation with adversarial removal so that a sanitiser which removes nothing scores zero rather than scoring perfectly.

$$\text{CEPS} = \frac{2PR}{P + R}$$

where *P* is the fraction of clinical entities retained after sanitisation and *R* is the fraction of payload sentences removed. Both components are reported separately throughout the paper.

---

## Repository Structure

```
ClinicalShield/
├── src/
│   └── clinicalshield/
│       ├── module1.py          # Canonicalisation + encoding detection
│       ├── module2.py          # PubMedBERT detection engine
│       ├── module3.py          # Semantic intent verification
│       ├── module4.py          # Utility-preserving sanitisation
│       ├── module5.py          # Clinical fact verification
│       └── pipeline.py         # Assembled five-module pipeline
├── notebooks/
│   ├── 01_corpus_construction.ipynb
│   ├── 02_attack_generation.ipynb
│   ├── 03_leakage_safe_split.ipynb
│   ├── 04_module1_ingestion.ipynb
│   ├── 05_module2_detection.ipynb
│   ├── 06_module3_intent.ipynb
│   ├── 07_module4_sanitisation.ipynb
│   ├── 08_baselines_transfer.ipynb
│   ├── 09_end_to_end_asr.ipynb
│   ├── 10_mpib_evaluation.ipynb
│   ├── 11_ablation.ipynb
│   ├── 12_aggregation.ipynb
│   └── 13_figures_tables.ipynb
├── data/
│   ├── dataset/                # openFDA corpus + leakage-safe splits
│   └── results/                # All results files read by the manuscript
├── figures/                    # All paper figures (generated by NB13)
├── tests/
│   └── test_modules.py         # 11 automated tests (all passing)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/NehlTech/ClinicalShield.git
cd ClinicalShield
git checkout v2-revision

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Requirements:** Python 3.10, PyTorch ≥ 2.0, HuggingFace Transformers ≥ 4.35, spaCy ≥ 3.7. Full pinned versions are in `requirements.txt`.

---

## Quickstart

```python
from clinicalshield.pipeline import ClinicalShieldPipeline

pipeline = ClinicalShieldPipeline()

query    = "What is the recommended dose of warfarin for atrial fibrillation?"
document = "...retrieved drug label text..."

result = pipeline.process(query, document)

print(result["decision"])          # "release" | "sanitise_and_release" | "withhold"
print(result["safe_context"])      # sanitised document text
print(result["ceps"])              # CEPS score
print(result["attack_detected"])   # True / False
```

---

## Reproducing the Results

The thirteen notebooks are designed to be run in order on Google Colab with an A100 GPU. Each notebook reads from `data/` and writes to `data/results/`. NB13 reads all results files and writes every figure and table in the paper.

```
NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07
→ NB08 → NB09 → NB10 → NB11 → NB12 → NB13
```

All numbers in the manuscript are generated by this pipeline and read from `data/results/all_paper_numbers.json`. None is typed by hand.

---

## Running the Tests

```bash
cd src
python -m pytest ../tests/test_modules.py -v
```

All 11 tests should pass. The test suite confirms that all five modules load correctly, process input, and return outputs in the expected format.

---

## Dataset

The corpus is constructed from openFDA drug labelling retrieved through the public API. It is reproducible from NB01 using the drug list recorded in `data/dataset/drug_list.json`. No proprietary data is used.

The Medical Prompt Injection Benchmark (MPIB) is subject to its authors' access terms and is not redistributed here. To reproduce the MPIB evaluation in NB10, request the benchmark from its original authors and place the payload file at `data/mpib/mpib_payloads.jsonl`.

---

## Citation

If you use ClinicalShield in your research, please cite:

```bibtex
@article{aduboahene2026clinicalshield,
  title   = {ClinicalShield: A Multi-Layer Defence Framework for Detecting
             and Neutralizing Indirect Prompt Injection Attacks in RAG-Based
             Clinical Decision Support Systems},
  author  = {Adu-Boahene, Bright and Kornyo, Oliver and Alhassan, Seiba
             and Muntaka, Siddique Abubakr and Afriyie, Yaw},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

The authors thank the Department of Computer Science at Kwame Nkrumah University of Science and Technology (KNUST) for institutional support. The openFDA API is provided by the U.S. Food and Drug Administration. PubMedBERT is developed by Microsoft Research.
