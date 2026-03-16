"""Fusion meta-learner for classification.

Combines behavioral features using gradient boosting with
confidence-weighted training and Platt-calibrated probabilities.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Labels for 3-class classification
LABEL_ORGANIC = 0
LABEL_BOT = 1
LABEL_HUB = 2
LABEL_NAMES = {0: 'organic', 1: 'bot', 2: 'hub'}


def prepare_fusion_features(df: pd.DataFrame,
                            behavioral_cols: list = None) -> np.ndarray:
    """Assemble feature matrix for the meta-learner.

    Args:
        df: DataFrame with location features
        behavioral_cols: List of behavioral feature column names to include

    Returns:
        (n_samples, n_features) feature matrix
    """
    if not behavioral_cols:
        raise ValueError("No features provided for fusion")

    available = [c for c in behavioral_cols if c in df.columns]
    if not available:
        raise ValueError("No features provided for fusion")

    feat_df = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    return feat_df.values


def train_meta_learner(X_train: np.ndarray, y_train: np.ndarray,
                       weights: np.ndarray = None) -> tuple:
    """Train confidence-weighted gradient boosting meta-learner.

    Args:
        X_train: Training features from seed sets
        y_train: Labels (0=organic, 1=bot, 2=hub)
        weights: Per-sample confidence weights

    Returns:
        (calibrated_model, scaler) tuple
    """
    scaler = StandardScaler()
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    # Wrap scaler + model in Pipeline so CalibratedClassifierCV
    # re-fits the scaler inside each CV fold, avoiding data leakage
    pipe = Pipeline([('scaler', scaler), ('clf', base_model)])
    pipe.fit(X_clean, y_train, clf__sample_weight=weights)

    # Calibrate probabilities (Platt scaling with proper CV)
    try:
        calibrated = CalibratedClassifierCV(pipe, cv=5, method='sigmoid')
        calibrated.fit(X_clean, y_train)
        logger.info("  Meta-learner trained with Platt calibration (cv=5)")
        return calibrated, None
    except Exception as e:
        logger.warning(f"  Calibration failed ({e}), using uncalibrated model")
        return pipe, None


def predict_with_confidence(model, scaler: StandardScaler,
                            X: np.ndarray) -> tuple:
    """Predict labels with calibrated confidence scores.

    Args:
        model: Trained (calibrated) meta-learner
        scaler: Fitted StandardScaler
        X: Feature matrix (n_samples, n_features)

    Returns:
        (labels, confidences, probabilities) tuple:
          labels: (n_samples,) int labels (0/1/2)
          confidences: (n_samples,) confidence of the prediction (0-1)
          probabilities: (n_samples, 3) class probabilities
    """
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None:
        X_clean = scaler.transform(X_clean)

    probas = model.predict_proba(X_clean)

    # Ensure all 3 classes are represented in output
    if probas.shape[1] < 3:
        # Pad with zeros for missing classes
        full_probas = np.zeros((len(X_clean), 3))
        classes = model.classes_ if hasattr(model, 'classes_') else np.arange(probas.shape[1])
        for i, c in enumerate(classes):
            if c < 3:
                full_probas[:, c] = probas[:, i]
        probas = full_probas

    labels = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    return labels, confidences, probas


def get_feature_importances(model, feature_names: list = None) -> pd.DataFrame:
    """Extract feature importances from the meta-learner.

    Args:
        model: Trained meta-learner (or CalibratedClassifierCV wrapper)
        feature_names: Optional list of feature names

    Returns:
        DataFrame with feature importance rankings
    """
    # Unwrap CalibratedClassifierCV / Pipeline if needed
    base = model
    if hasattr(model, 'estimator'):
        base = model.estimator
    elif hasattr(model, 'calibrated_classifiers_'):
        base = model.calibrated_classifiers_[0].estimator

    # Unwrap Pipeline to get the classifier step
    if hasattr(base, 'named_steps'):
        base = base.named_steps.get('clf', base)

    if not hasattr(base, 'feature_importances_'):
        logger.warning("Model does not support feature importances")
        return pd.DataFrame()

    importances = base.feature_importances_
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    imp_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances,
    }).sort_values('importance', ascending=False)

    return imp_df


def train_meta_learner_gold_standard(X_train: np.ndarray,
                                      y_train: np.ndarray) -> tuple:
    """Train meta-learner using gold-standard labels with class balancing.

    Unlike train_meta_learner() which uses per-sample confidence weights
    from heuristic seeds, this uses inverse-frequency class weights computed
    from the gold-standard label distribution.

    Args:
        X_train: Training features from gold-standard labeled locations
        y_train: Labels (0=organic, 1=bot, 2=hub)

    Returns:
        (calibrated_model, scaler) tuple
    """
    from sklearn.utils.class_weight import compute_class_weight

    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute balanced class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_map = dict(zip(classes, class_weights))
    sample_weights = np.array([weight_map[label] for label in y_train])

    logger.info(f"  Class weights: {', '.join(f'{LABEL_NAMES.get(c, c)}={w:.3f}' for c, w in weight_map.items())}")

    scaler = StandardScaler()
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    # Wrap scaler + model in Pipeline so CalibratedClassifierCV
    # re-fits the scaler inside each CV fold, avoiding data leakage
    pipe = Pipeline([('scaler', scaler), ('clf', base_model)])
    pipe.fit(X_clean, y_train, clf__sample_weight=sample_weights)

    # Calibrate probabilities (Platt scaling with proper CV)
    try:
        calibrated = CalibratedClassifierCV(pipe, cv=5, method='sigmoid')
        calibrated.fit(X_clean, y_train)
        logger.info("  Gold-standard meta-learner trained with Platt calibration (cv=5)")
        return calibrated, None
    except Exception as e:
        logger.warning(f"  Calibration failed ({e}), using uncalibrated model")
        return pipe, None
