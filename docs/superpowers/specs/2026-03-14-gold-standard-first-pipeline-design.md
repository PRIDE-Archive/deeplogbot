# Gold-Standard-First Classification Pipeline (v9)

## Problem Statement

The current DeepLogBot pipeline (v8) uses a semi-supervised approach where heuristic seed selection identifies ~30K training examples, augmented by 625 gold-standard LLM consensus labels injected as high-confidence seeds. The gold standard is drowned 50:1 by heuristic organic seeds, preventing the meta-learner from learning the true decision boundary — particularly for small, single-year locations where behavioral features overlap between bots and organic users.

**Evidence of failure**: 2025 saw 61,438 new locations (vs ~2,400/year baseline in 2021-2024). Of these, ~37K are classified as "user" but the gold standard indicates 78% of the labeled subset are actually bots. The meta-learner assigns prob_organic > 0.99 to these locations because heuristic organic seeds dominate the feature space where small bots live.

**Root cause**: The gold standard is used as a supplement to heuristic seeds, not as the primary training data. The heuristic seeds define the decision boundary; the gold standard merely nudges it.

## Data Preparation

### Creating `data/gold_standard_labels.csv`

Source: `data/blind_consensus_final.csv` (934 locations with columns: geo_location, city, country, label, source).

The train/test split is created programmatically:
1. Load `blind_consensus_final.csv`
2. Match each location to its `validation_zone` from the original stratified sampling (stored in `data/llm_corrections.csv`, column `validation_zone`)
3. Perform stratified split by `validation_zone`: 67% train (625) / 33% test (309), using `random_state=42` for reproducibility
4. Save as `data/gold_standard_labels.csv` with columns: `geo_location`, `label`, `source`, `split`

This is a one-time derivation step, executed as part of the first implementation task. The resulting file is checked into the repo.

## Design

### Architecture: 4-Phase Pipeline

The v8 pipeline had 5 phases in code (Seed Selection → LLM Injection → Fusion → Hub Protection → Finalization), documented as 4 phases in some places because LLM injection was considered part of seed selection. The v9 pipeline cleanly reduces to 4 phases by removing phases 1-2 entirely and replacing them with gold-standard supervised training:

1. **Feature Extraction** (unchanged): 33 behavioral features per location
2. **Gold-Standard Supervised Training** (new): Train GradientBoosting exclusively on 934 LLM consensus labels
3. **Hub Protection** (unchanged): Structural override rules for institutional locations
4. **Finalization** (unchanged): Insufficient evidence marking, boolean derivation

The key change: heuristic seed selection is **removed from the training loop**. The gold standard becomes the sole source of supervised labels.

### Phase 2: Gold-Standard Supervised Training

**Data**:
- Load `data/gold_standard_labels.csv` (934 locations: 593 bot, 134 hub, 207 organic)
- Pre-split: 625 train / 309 test (stratified by validation zone, random_state=42)
- Labels come from blind multi-LLM consensus (Claude Opus 4.6 + Qwen3-30B-A3B, Cohen's κ = 0.535)

**Model**:
- GradientBoostingClassifier (200 estimators, max depth 5, learning rate 0.1, subsample 0.8, min samples leaf 10)
- Class weights computed via `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` and passed as `sample_weight` to `fit()` (GradientBoostingClassifier does not have a native `class_weight` parameter). This replaces the per-sample seed confidence weights from v8 — all gold-standard labels have equal base confidence.
- Feature standardization via StandardScaler
- Platt calibration via `CalibratedClassifierCV` with `cv='prefit'` (calibrate on training data). This matches the v8 approach and avoids the statistical fragility of cross-validated calibration with only ~90 hub training samples.
- Same 33 behavioral features as v8

**Prediction**:
- Apply trained model to all 71K locations
- Output: prob_organic, prob_bot, prob_hub + predicted class
- Low-confidence flag: max probability < 0.5 → needs_review = True

### Phase 3: Hub Protection (unchanged)

Structural hub protection rules act as post-classification safety net. Rules from config.yaml:
- High DL/user mirror (>500 DL/user, ≤200 users)
- Few users high DL (≤100 users, >100 DL/user)
- Protocol-based (Aspera >0.3 or Globus >0.1)
- Research institution (≤1000 users, >200 DL/user, ≥3 years)
- Large research hub (>200 DL/user, ≥3 years)

### Phase 4: Finalization (unchanged)

- Locations with <3 total downloads → insufficient_evidence
- Boolean columns derived: is_bot, is_hub, is_organic

## Configuration

### config.yaml changes

Replace the `llm_corrections` section with:

```yaml
gold_standard:
  path: "data/gold_standard_labels.csv"
  label_column: "label"
  location_column: "geo_location"
  split_column: "split"

# Training mode: "gold_standard" (v9) or "semi_supervised" (v8 legacy)
training_mode: "gold_standard"
```

The `training_mode` toggle allows falling back to the v8 semi-supervised pipeline (heuristic seeds + LLM injection) without code changes, for comparison or if the gold-standard approach underperforms on a specific metric.

## File Changes

### Modified files

| File | Change |
|------|--------|
| `deeplogbot/models/classification/deep_architecture.py` | Replace 5-phase pipeline with 4-phase. Remove calls to `select_seeds()` and `inject_llm_seeds()` when `training_mode=gold_standard`. Add gold-standard loading and direct supervised training. Keep v8 code path behind `training_mode=semi_supervised` toggle. |
| `deeplogbot/models/classification/fusion.py` | Update `train_meta_learner()` to accept a gold-standard DataFrame (geo_location + label) as an alternative to seed DataFrames. Replace per-sample seed confidence weights with `compute_class_weight('balanced')` passed via `sample_weight`. |
| `deeplogbot/config.yaml` | Add `gold_standard` section and `training_mode` toggle. Keep `llm_corrections` for backward compatibility. |
| `deeplogbot/main.py` | Update `run_bot_annotator()` to read `training_mode` from config and pass gold-standard path or seeds accordingly. |
| `scripts/classify_full_dataset.py` | Pass gold standard config instead of llm_corrections. |
| `scripts/retrain_with_llm_labels.py` | Update to compare no-gold-standard baseline vs gold-standard-trained model. |

### New files

| File | Purpose |
|------|---------|
| `data/gold_standard_labels.csv` | Clean gold standard: geo_location, label, source, split. Derived from `blind_consensus_final.csv` + `llm_corrections.csv` (for validation_zone). |

### Unchanged files

| File | Reason |
|------|--------|
| `seed_selection.py` | Kept for Rules method, v8 fallback, and backward compatibility. |
| `post_classification.py` | Hub protection unchanged. |
| `rules.py` | Independent method, unchanged. |

## Validation Plan

1. **Held-out accuracy**: 309-location test set accuracy (currently 84.8% with diluted training — should improve)
2. **Per-regime accuracy**: Report accuracy separately for small/single-year (229 gold-standard labels) vs large/multi-year (705 labels) to monitor overfitting in the small regime
3. **Temporal sanity**: 2025 organic new-location count should be ~2,400 (matching 2021-2024 baseline), not 61K
4. **Download sanity**: 2025 user downloads should show modest growth over 2024, not 9x
5. **Rules vs Deep comparison**: Recompute S6.6 tables against same 934 gold standard
6. **needs_review count**: Report how many of the 71K locations are flagged with max probability < 0.5
7. **Manual spot-check**: Sample 50 reclassified locations to verify

## Manuscript Impact

- Pipeline description: "five-phase" → "four-phase" (seed selection removed from training)
- Phase 2 becomes "Gold-Standard Supervised Training" instead of "LLM Seed Refinement"
- All downstream numbers (bot/hub/user percentages, temporal trends, country distributions) will change
- All figures need regeneration after reclassification
- Discussion section needs updating (bot traffic percentage, hub counts, etc.)

## Why This Works

The gold standard contains 229 labels in the problematic small/single-year regime. In this regime, `unique_users` alone separates bot from organic with >90% accuracy (all 74 organic have ≤10 users; 86% of 155 bots have >10 users). The current meta-learner can't learn this because 30K heuristic organic seeds include thousands of locations with 10-50 users, blurring the boundary. Training exclusively on the gold standard gives the model an undiluted view of the true decision boundary.

**Overfitting risk**: With 625 training examples and 33 features, the model could overfit. Mitigation: (a) the held-out 309 test set catches overfitting, (b) per-regime accuracy reporting detects if the model memorizes small-location labels, (c) GradientBoosting with max_depth=5 and min_samples_leaf=10 provides regularization, (d) temporal and download sanity checks catch unreasonable aggregate outcomes.
