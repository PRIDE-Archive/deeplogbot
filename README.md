# DeepLogBot

[![Python](https://img.shields.io/pypi/pyversions/deeplogbot)](https://pypi.org/project/deeplogbot/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Semi-supervised and LLM-refined traffic classification for scientific data repository download logs.

## Overview

DeepLogBot classifies download traffic from scientific data repositories into three categories:

- **Independent users** -- Individual researchers with natural download patterns
- **Bots** -- Automated scrapers, crawlers, and coordinated bot farms
- **Download hubs** -- Legitimate institutional mirrors, reanalysis pipelines, and data aggregators

Applied to the PRIDE Archive (159.3M download records, 2021--2025), DeepLogBot classified 71,133 geographic locations with 92.2% accuracy on a held-out test set evaluated against blind multi-LLM consensus labels.

## Classification Pipeline

DeepLogBot uses a four-phase semi-supervised pipeline:

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Heuristic Seed Selection** | Identifies high-confidence bot/user/hub seeds using behavioral heuristics (3-tier organic, 6-signal bot, structural hub) |
| 2 | **Blind Multi-LLM Annotation** | 1,153 locations annotated blindly by Claude and Qwen3; 934 consensus labels split into train (67%) and test (33%) sets, injected as gold-standard seeds |
| 3 | **Fusion Meta-learner** | GradientBoosting classifier (200 estimators, Platt calibration) trained on 36 behavioral features |
| 4 | **Hub Protection & Finalization** | Structural rules prevent institutional locations from being misclassified as bots; insufficient-evidence filter + final boolean labels |

Gold-standard training labels improved accuracy from 62.1% (heuristic seeds only) to 92.2%, with per-class F1 gains: bot 0.734 to 0.946, hub 0.687 to 0.933, organic 0.305 to 0.849.

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, scipy, duckdb, pyyaml

## Usage

### Command Line

```bash
# Deep classification (recommended)
deeplogbot -i data.parquet -o output/ -m deep

# Rule-based classification (faster, no ML training)
deeplogbot -i data.parquet -o output/ -m rules

# With sampling for large datasets
deeplogbot -i data.parquet -o output/ -m deep --sample-size 1000000
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input parquet file | Required |
| `-o, --output-dir` | Output directory | `output/bot_analysis` |
| `-m, --classification-method` | `rules` or `deep` | `rules` |
| `-s, --sample-size` | Sample N records | None (use all) |
| `-p, --provider` | Log provider | `ebi` |

### Python API

```python
from deeplogbot import run_bot_annotator

results = run_bot_annotator(
    input_parquet='data.parquet',
    output_dir='output/',
    classification_method='deep'
)
```

## Project Structure

```
deeplogbot/
├── main.py                      # CLI entry point and pipeline
├── config.py                    # Configuration loading
├── config.yaml                  # Pipeline config
├── features/                    # Feature extraction
│   └── providers/ebi/           # EBI/PRIDE-specific extractors
├── models/
│   └── classification/
│       ├── deep_architecture.py # 4-phase pipeline orchestrator
│       ├── seed_selection.py    # Heuristic seed identification
│       ├── fusion.py            # GradientBoosting meta-learner
│       ├── post_classification.py # Hub protection & finalization
│       ├── rules.py             # Rule-based classifier
│       └── feature_validation.py
├── reports/                     # Output generation
└── utils/

data/
├── gold_standard_labels.csv     # Gold-standard labels with train/test split
├── llm_corrections.csv          # LLM-annotated seed corrections
└── prompts/                     # LLM annotation prompts

scripts/
├── classify_full_dataset.py     # Run classification on full dataset
├── retrain_with_llm_labels.py   # LLM-augmented retraining experiment
├── blind_llm_annotation.py      # Blind multi-LLM annotation pipeline
├── compute_llm_consensus.py     # Compute consensus from LLM annotations
├── build_enrichment_cache.py    # Build location enrichment cache
├── run_full_analysis.py         # Analysis pipeline
├── generate_figures.py          # Main manuscript figures
└── generate_supp_figures.py     # Supplementary figures
```

## Configuration

Main configuration is in `deeplogbot/config.yaml`:

```yaml
# Gold-standard labels (preferred, used by default)
gold_standard:
  path: "data/gold_standard_labels.csv"
  location_column: "geo_location"
  label_column: "label"
  split_column: "split"

# Hub protection rules
classification:
  hub_protection:
    research_institution:
      max_users: 1000
      min_downloads_per_user: 200
      min_years_span: 3
```

## Input Format

The input parquet must contain at minimum:

| Column | Description |
|--------|-------------|
| `accession` | Dataset accession (e.g., `PXD000001`) |
| `geo_location` | Geographic coordinate string |
| `country` | Country name |
| `year` | Download year |
| `date` | Download date |

## Output

| Column | Description |
|--------|-------------|
| `is_bot` | Bot classification flag |
| `is_hub` | Download hub classification flag |
| `is_organic` | Independent user classification flag |
| `classification_confidence` | Confidence score (0-1) |

Reports generated: `bot_detection_report.txt`, `location_analysis.csv`.

## Citation

If you use DeepLogBot in your research, please cite:

> Hewapathirana S, Bai J, Bandla C, et al. *Tracking dataset reuse in proteomics: semi-supervised and LLM-refined analysis of PRIDE download statistics.* (2025)

## License

Apache 2.0
