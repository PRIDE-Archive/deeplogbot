# DeepLogBot

[![PyPI version](https://img.shields.io/pypi/v/deeplogbot)](https://pypi.org/project/deeplogbot/)
[![Python](https://img.shields.io/pypi/pyversions/deeplogbot)](https://pypi.org/project/deeplogbot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bot detection and traffic classification for scientific data repository download logs.

## Overview

DeepLogBot classifies download traffic from scientific data repositories into three categories:

- **Organic users** -- Human researchers with natural download patterns
- **Bots** -- Automated scrapers, crawlers, and coordinated bot farms
- **Download hubs** -- Legitimate institutional mirrors, reanalysis pipelines, and data aggregators

Applied to the PRIDE Archive (159M download records, 2021--2025), DeepLogBot classified 71,133 geographic locations with 85.6% accuracy on an independent held-out test set.

## Classification Pipeline

DeepLogBot uses a five-phase semi-supervised pipeline refined by LLM annotations:

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Heuristic Seed Selection** | Identifies high-confidence bot/organic/hub seeds using behavioral heuristics (3-tier organic, 6-signal bot, structural hub) |
| 2 | **LLM Seed Refinement** | Injects LLM-annotated corrections (from `data/llm_corrections.csv`) to fix systematic seed errors |
| 3 | **Fusion Meta-learner** | GradientBoosting classifier (200 estimators, Platt calibration) trained on 33 behavioral features |
| 4 | **Hub Protection** | Structural rules prevent institutional locations from being misclassified as bots |
| 5 | **Finalization** | Insufficient-evidence filter + final boolean labels |

The LLM refinement step (Phase 2) improved accuracy from 67.2% to 85.6%, with the largest gains in organic F1 (0.575 to 0.837) and bot F1 (0.684 to 0.874).

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

### Retraining with LLM corrections

To evaluate the LLM-augmented retraining (baseline vs retrained comparison):

```bash
python scripts/retrain_with_llm_labels.py
```

This splits the 1,153 LLM-labeled locations into train/test, runs baseline and LLM-augmented classification, and prints accuracy comparisons.

## Project Structure

```
deeplogbot/
├── main.py                      # CLI entry point and pipeline
├── config.py                    # Configuration loading
├── config.yaml                  # Pipeline config (incl. LLM corrections path)
├── features/                    # Feature extraction
│   └── providers/ebi/           # EBI/PRIDE-specific extractors
├── models/
│   └── classification/
│       ├── deep_architecture.py # 5-phase pipeline orchestrator
│       ├── seed_selection.py    # Heuristic seed identification
│       ├── fusion.py            # GradientBoosting meta-learner
│       ├── post_classification.py # Hub protection & finalization
│       ├── rules.py             # Rule-based classifier
│       └── feature_validation.py
├── reports/                     # Output generation
└── utils/

data/
└── llm_corrections.csv          # 1,153 LLM-annotated seed corrections

scripts/
├── classify_full_dataset.py     # Run classification on full dataset
├── retrain_with_llm_labels.py   # LLM-augmented retraining experiment
├── run_full_analysis.py         # Analysis pipeline
├── generate_figures.py          # Main manuscript figures
└── generate_supp_figures.py     # Supplementary figures
```

## Configuration

Main configuration is in `deeplogbot/config.yaml`:

```yaml
# LLM seed corrections (auto-loaded during deep classification)
llm_corrections:
  path: "data/llm_corrections.csv"
  label_column: "claude_evaluation"
  location_column: "geo_location"
  weight: 0.95

# Hub protection rules
classification:
  hub_protection:
    research_institution:
      max_users: 1000
      min_downloads_per_user: 200
      min_years_span: 3
```

Set `llm_corrections` to `null` to disable LLM seed injection and use heuristic-only seeds.

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
| `is_organic` | Organic user classification flag |
| `classification_confidence` | Confidence score (0-1) |

Reports generated: `bot_detection_report.txt`, `location_analysis.csv`, and optionally an interactive HTML report.

## License

MIT
