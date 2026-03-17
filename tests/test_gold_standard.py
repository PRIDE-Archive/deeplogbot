"""Tests for gold-standard-first pipeline."""

import numpy as np
import pandas as pd
import pytest
from deeplogbot.models.classification.fusion import (
    train_meta_learner,
    train_meta_learner_gold_standard,
    predict_with_confidence,
    LABEL_ORGANIC, LABEL_BOT, LABEL_HUB,
)


def _make_synthetic_gold_standard(n_bot=60, n_hub=15, n_organic=25):
    """Create synthetic gold standard data with separable classes."""
    rng = np.random.RandomState(42)

    # Bots: many users, low DL/user
    bot_features = np.column_stack([
        rng.uniform(50, 200, n_bot),   # unique_users (high)
        rng.uniform(1, 5, n_bot),      # downloads_per_user (low)
        rng.uniform(0.2, 0.4, n_bot),  # working_hours_ratio (low)
        rng.uniform(0.3, 0.5, n_bot),  # night_activity_ratio (high)
        rng.uniform(2.5, 3.5, n_bot),  # hourly_entropy (high)
    ])

    # Hubs: few users, high DL/user
    hub_features = np.column_stack([
        rng.uniform(1, 20, n_hub),      # unique_users (low)
        rng.uniform(200, 1000, n_hub),  # downloads_per_user (high)
        rng.uniform(0.3, 0.6, n_hub),  # working_hours_ratio (moderate)
        rng.uniform(0.1, 0.3, n_hub),  # night_activity_ratio (low)
        rng.uniform(1.5, 2.5, n_hub),  # hourly_entropy (moderate)
    ])

    # Organic: few users, moderate DL/user
    organic_features = np.column_stack([
        rng.uniform(1, 10, n_organic),    # unique_users (low)
        rng.uniform(3, 30, n_organic),    # downloads_per_user (moderate)
        rng.uniform(0.3, 0.6, n_organic), # working_hours_ratio (high)
        rng.uniform(0.1, 0.4, n_organic), # night_activity_ratio (low)
        rng.uniform(0.5, 1.5, n_organic), # hourly_entropy (low)
    ])

    X = np.vstack([bot_features, hub_features, organic_features])
    y = np.array([LABEL_BOT] * n_bot + [LABEL_HUB] * n_hub + [LABEL_ORGANIC] * n_organic)
    return X, y


def test_train_meta_learner_gold_standard():
    """Gold-standard training produces a working model with class weights."""
    X, y = _make_synthetic_gold_standard()
    model, scaler = train_meta_learner_gold_standard(X, y)

    assert model is not None
    # scaler is None when embedded in Pipeline (avoids calibration leakage)

    # Predict on training data — should get most right
    labels, confidences, probas = predict_with_confidence(model, scaler, X)
    accuracy = (labels == y).mean()
    assert accuracy > 0.7, f"Training accuracy too low: {accuracy:.3f}"
    assert probas.shape == (len(X), 3)
    assert all(c > 0 for c in confidences)


def test_gold_standard_class_weights():
    """Class weights should boost minority classes (hub, organic)."""
    X, y = _make_synthetic_gold_standard(n_bot=60, n_hub=10, n_organic=20)
    model, scaler = train_meta_learner_gold_standard(X, y)

    labels, _, _ = predict_with_confidence(model, scaler, X)

    # Hub (minority) should still be predicted for hub examples
    hub_examples = X[y == LABEL_HUB]
    hub_preds, _, _ = predict_with_confidence(model, scaler, hub_examples)
    hub_recall = (hub_preds == LABEL_HUB).mean()
    assert hub_recall > 0.3, f"Hub recall too low: {hub_recall:.3f}"


def test_original_train_meta_learner_still_works():
    """v8 train_meta_learner with sample weights still works (backward compat)."""
    X, y = _make_synthetic_gold_standard()
    weights = np.ones(len(y)) * 0.8
    model, scaler = train_meta_learner(X, y, weights=weights)

    labels, confidences, probas = predict_with_confidence(model, scaler, X)
    assert len(labels) == len(y)
