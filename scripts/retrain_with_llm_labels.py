#!/usr/bin/env python3
"""Retrain DeepLogBot with LLM-annotated seed corrections and measure improvement.

Splits 1,153 LLM-labeled locations into train/test (stratified by zone),
injects train labels as high-confidence seeds, retrains the meta-learner,
and compares baseline vs retrained accuracy on the held-out test set.

Usage:
    python scripts/retrain_with_llm_labels.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeplogbot.models.classification.deep_architecture import (
    classify_locations_deep,
    BEHAVIORAL_FEATURE_COLS,
)
from deeplogbot.utils import logger


LABEL_MAP = {'organic': 0, 'bot': 1, 'hub': 2}
LABEL_NAMES = {0: 'organic', 1: 'bot', 2: 'hub'}
BEHAVIOR_TYPE_MAP = {'user': 'organic', 'bot': 'bot', 'hub': 'hub'}


def load_data(output_dir: str):
    """Load location analysis and LLM validation data."""
    loc_path = os.path.join(output_dir, 'location_analysis.csv')
    val_path = os.path.join(project_root, 'data', 'llm_corrections.csv')

    logger.info(f"Loading location analysis from {loc_path}")
    df_locations = pd.read_csv(loc_path)
    logger.info(f"  {len(df_locations):,} locations loaded")

    logger.info(f"Loading LLM corrections from {val_path}")
    df_validation = pd.read_csv(val_path)
    logger.info(f"  {len(df_validation):,} LLM-labeled locations loaded")

    return df_locations, df_validation


def split_train_test(df_validation: pd.DataFrame, test_size: float = 0.33,
                     random_state: int = 42):
    """Stratified train/test split by validation_zone."""
    zones = df_validation['validation_zone'].values
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df_validation, zones))

    train = df_validation.iloc[train_idx].copy()
    test = df_validation.iloc[test_idx].copy()

    logger.info(f"  Train: {len(train)} locations")
    logger.info(f"  Test:  {len(test)} locations")

    # Log class distribution
    for name, subset in [('Train', train), ('Test', test)]:
        dist = subset['claude_evaluation'].value_counts()
        logger.info(f"    {name}: {dict(dist)}")

    return train, test


def run_classification(df_locations: pd.DataFrame,
                       llm_corrections: pd.DataFrame = None,
                       disable_llm_autoload: bool = False,
                       label: str = "baseline"):
    """Run the full classification pipeline, optionally with LLM corrections.

    Args:
        df_locations: Full location DataFrame.
        llm_corrections: Explicit LLM corrections to inject. If None and
            disable_llm_autoload is False, auto-loads from config.yaml.
        disable_llm_autoload: If True, pass empty DataFrame to prevent
            auto-loading from config (used for baseline comparison).
        label: Label for logging.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running classification: {label}")
    logger.info(f"{'='*70}")

    df = df_locations.copy()
    feature_cols = [c for c in BEHAVIORAL_FEATURE_COLS if c in df.columns]

    # Pass empty DataFrame to explicitly disable auto-loading from config
    corrections = llm_corrections
    if disable_llm_autoload and corrections is None:
        corrections = pd.DataFrame()

    start = time.time()
    classified_df, _ = classify_locations_deep(
        df,
        feature_columns=feature_cols,
        llm_corrections=corrections,
    )
    elapsed = time.time() - start
    logger.info(f"  Classification completed in {elapsed:.1f}s")

    return classified_df


def evaluate_on_test(classified_df: pd.DataFrame, test_set: pd.DataFrame):
    """Evaluate classifier predictions against LLM labels on the test set."""
    # Join predictions back to test set via geo_location
    test_geos = set(test_set['geo_location'].values)
    pred_rows = classified_df[classified_df['geo_location'].isin(test_geos)].copy()

    # Map behavior_type to standard labels
    pred_rows['pred_label'] = pred_rows['behavior_type'].map(BEHAVIOR_TYPE_MAP)

    # Merge with LLM labels
    merged = test_set[['geo_location', 'claude_evaluation']].merge(
        pred_rows[['geo_location', 'pred_label', 'classification_confidence']],
        on='geo_location',
        how='inner',
    )

    if len(merged) < len(test_set):
        logger.warning(f"  Only matched {len(merged)}/{len(test_set)} test locations")

    # Confusion matrix
    cm = Counter()
    for _, row in merged.iterrows():
        cm[(row['pred_label'], row['claude_evaluation'])] += 1

    # Metrics
    labels = ['bot', 'hub', 'organic']
    total = len(merged)
    agree = sum(1 for _, r in merged.iterrows()
                if r['pred_label'] == r['claude_evaluation'])
    accuracy = agree / total if total > 0 else 0

    metrics = {'accuracy': accuracy, 'total': total, 'agree': agree}
    per_class = {}

    for lbl in labels:
        tp = cm.get((lbl, lbl), 0)
        fp = sum(cm.get((other, lbl), 0) for other in labels if other != lbl)
        fn = sum(cm.get((lbl, other), 0) for other in labels if other != lbl)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[lbl] = {'precision': p, 'recall': r, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

    metrics['per_class'] = per_class
    metrics['confusion_matrix'] = {f"{k[0]}->{k[1]}": v for k, v in cm.items()}

    return metrics


def print_comparison(baseline: dict, retrained: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs LLM-Augmented Retraining")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Retrained':>12} {'Delta':>12}")
    print("-" * 66)

    b_acc = baseline['accuracy']
    r_acc = retrained['accuracy']
    print(f"{'Overall accuracy':<30} {b_acc:>11.1%} {r_acc:>11.1%} {r_acc-b_acc:>+11.1%}")

    for lbl in ['bot', 'hub', 'organic']:
        b = baseline['per_class'][lbl]
        r = retrained['per_class'][lbl]
        print(f"\n  {lbl.upper()}")
        for metric in ['precision', 'recall', 'f1']:
            bv = b[metric]
            rv = r[metric]
            print(f"    {metric:<26} {bv:>11.3f} {rv:>11.3f} {rv-bv:>+11.3f}")

    # Confusion matrices side by side
    labels = ['bot', 'hub', 'organic']
    for name, data in [('BASELINE', baseline), ('RETRAINED', retrained)]:
        print(f"\n  {name} Confusion Matrix (classifier rows x LLM cols):")
        print(f"    {'':>12} {'bot':>8} {'hub':>8} {'organic':>8}")
        for clf_lbl in labels:
            vals = [data['confusion_matrix'].get(f"{clf_lbl}->{llm_lbl}", 0)
                    for llm_lbl in labels]
            print(f"    {clf_lbl:>12} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")


def main():
    output_dir = 'output/full_deep_v8'

    # Load data
    df_locations, df_validation = load_data(output_dir)

    # Split
    train, test = split_train_test(df_validation)

    # Run baseline (no LLM corrections — disable auto-load from config)
    df_baseline = run_classification(df_locations, disable_llm_autoload=True, label="BASELINE")
    baseline_metrics = evaluate_on_test(df_baseline, test)
    logger.info(f"  Baseline accuracy: {baseline_metrics['accuracy']:.1%}")

    # Run retrained (with LLM train corrections)
    df_retrained = run_classification(df_locations, llm_corrections=train, label="LLM-AUGMENTED")
    retrained_metrics = evaluate_on_test(df_retrained, test)
    logger.info(f"  Retrained accuracy: {retrained_metrics['accuracy']:.1%}")

    # Compare
    print_comparison(baseline_metrics, retrained_metrics)

    # Count net reclassifications across all 71K locations
    if 'behavior_type' in df_baseline.columns and 'behavior_type' in df_retrained.columns:
        changed = (df_baseline['behavior_type'] != df_retrained['behavior_type']).sum()
        logger.info(f"\n  Net reclassifications across all locations: {changed:,}")

    # Save results
    results = {
        'created': datetime.now().isoformat(),
        'train_size': len(train),
        'test_size': len(test),
        'baseline': baseline_metrics,
        'retrained': retrained_metrics,
        'accuracy_delta': retrained_metrics['accuracy'] - baseline_metrics['accuracy'],
    }

    results_path = os.path.join(output_dir, 'llm_retraining_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n  Results saved: {results_path}")


if __name__ == '__main__':
    main()
