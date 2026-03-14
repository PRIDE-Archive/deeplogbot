"""Learned deep classification for bot / hub / organic detection.

Supports two training modes (set via config.yaml ``training_mode``):

**gold_standard** (v9, default):
  1. Feature Extraction – 33 behavioral features per location.
  2. Gold-Standard Supervised Training – GradientBoosting trained on
     934 blind multi-LLM consensus labels with class balancing.
  3. Hub Protection – rule-based structural override.
  4. Finalize – insufficient evidence filter + boolean columns.

**semi_supervised** (v8 legacy):
  1. Heuristic Seed Selection – organic / bot / hub seeds from rules.
  2. LLM Seed Refinement – optional injection of LLM corrections.
  3. Fusion Meta-learner – gradient-boosted classifier on seeds.
  4. Hub Protection – rule-based structural override.
  5. Finalize – insufficient evidence filter + boolean columns.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from ...utils import logger
from .post_classification import (
    apply_hub_protection,
    log_prediction_summary,
    log_hierarchical_summary,
)

from .seed_selection import select_organic_seed, select_bot_seed, select_hub_seed
from .fusion import (
    LABEL_ORGANIC, LABEL_BOT, LABEL_HUB,
    prepare_fusion_features,
    train_meta_learner,
    train_meta_learner_gold_standard,
    predict_with_confidence,
    get_feature_importances,
)


# ---------------------------------------------------------------------------
# Core behavioural features used as fusion inputs
# ---------------------------------------------------------------------------

BEHAVIORAL_FEATURE_COLS = [
    'unique_users', 'downloads_per_user', 'total_downloads',
    'working_hours_ratio', 'night_activity_ratio', 'hourly_entropy',
    'burst_pattern_score', 'user_coordination_score',
    'spike_ratio', 'fraction_latest_year', 'year_over_year_cv',
    'years_span', 'protocol_legitimacy_score',
    'aspera_ratio', 'globus_ratio',
    'regularity_score', 'file_diversity_ratio',
    'bot_composite_score', 'user_scarcity_score',
    'download_concentration', 'temporal_irregularity',
    'request_velocity', 'access_regularity',
    'weekend_weekday_imbalance',
    'user_entropy', 'user_gini_coefficient',
    'single_download_user_ratio', 'power_user_ratio',
    'session_duration_cv', 'inter_session_regularity',
    'momentum_score', 'recent_activity_ratio',
    'unique_projects',
]


# ---------------------------------------------------------------------------
# LLM seed correction helpers
# ---------------------------------------------------------------------------

def load_llm_corrections_from_config() -> Optional[pd.DataFrame]:
    """Load LLM corrections from the path specified in config.yaml.

    Returns:
        DataFrame with LLM corrections, or None if not configured/found.
    """
    from ...config import APP_CONFIG

    llm_cfg = APP_CONFIG.get('llm_corrections')
    if not llm_cfg or not llm_cfg.get('path'):
        return None

    path = llm_cfg['path']
    # Resolve relative paths from project root
    if not os.path.isabs(path):
        # Walk up from this file to find project root (contains config.yaml)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(project_root, path)

    if not os.path.exists(path):
        logger.warning(f"    LLM corrections file not found: {path}")
        return None

    df = pd.read_csv(path)
    label_col = llm_cfg.get('label_column', 'claude_evaluation')
    loc_col = llm_cfg.get('location_column', 'geo_location')

    if label_col not in df.columns or loc_col not in df.columns:
        logger.warning(f"    LLM corrections missing required columns: {label_col}, {loc_col}")
        return None

    logger.info(f"    Loaded {len(df)} LLM corrections from {path}")
    return df


def inject_llm_seeds(
    df: pd.DataFrame,
    organic_seed: pd.DataFrame,
    bot_seed: pd.DataFrame,
    hub_seed: pd.DataFrame,
    llm_corrections: pd.DataFrame,
    label_column: str = 'claude_evaluation',
    location_column: str = 'geo_location',
    weight: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inject LLM-annotated corrections into seed sets.

    For each correction, removes the location from any existing seed set
    and adds it to the seed set matching the LLM label.

    Args:
        df: Full location DataFrame (to look up indices).
        organic_seed, bot_seed, hub_seed: Current seed DataFrames.
        llm_corrections: DataFrame with LLM labels.
        label_column: Column name containing LLM labels.
        location_column: Column name containing geo_location identifiers.
        weight: Seed confidence weight for LLM corrections.

    Returns:
        Updated (organic_seed, bot_seed, hub_seed) tuple.
    """
    valid_labels = {'organic', 'bot', 'hub'}
    n_override, n_added = 0, 0

    if location_column not in df.columns:
        logger.warning(f"    Cannot inject LLM seeds: '{location_column}' not in df")
        return organic_seed, bot_seed, hub_seed

    # Build geo_location → df index lookup for fast matching
    geo_to_idx = {}
    for idx, geo in df[location_column].items():
        if geo not in geo_to_idx:
            geo_to_idx[geo] = idx

    seed_sets = {'organic': organic_seed, 'bot': bot_seed, 'hub': hub_seed}

    for _, corr in llm_corrections.iterrows():
        geo = corr.get(location_column)
        llm_label = corr.get(label_column)
        if geo is None or llm_label not in valid_labels:
            continue

        loc_idx = geo_to_idx.get(geo)
        if loc_idx is None:
            continue

        # Remove from any existing seed set
        for name in list(seed_sets.keys()):
            if loc_idx in seed_sets[name].index:
                if llm_label != name:
                    n_override += 1
                seed_sets[name] = seed_sets[name].drop(loc_idx)

        # Add to the correct seed set
        row = df.loc[[loc_idx]].copy()
        row['seed_confidence'] = weight
        seed_sets[llm_label] = pd.concat([seed_sets[llm_label], row])
        n_added += 1

    # Deduplicate indices (in case of repeated geo_locations)
    for name in seed_sets:
        seed_sets[name] = seed_sets[name][~seed_sets[name].index.duplicated(keep='last')]

    logger.info(f"    LLM corrections: {n_added} injected, {n_override} overrides")
    logger.info(f"    Updated seeds: {len(seed_sets['organic'])} organic, "
                f"{len(seed_sets['bot'])} bot, {len(seed_sets['hub'])} hub")

    return seed_sets['organic'], seed_sets['bot'], seed_sets['hub']


# ---------------------------------------------------------------------------
# Gold-standard label loading
# ---------------------------------------------------------------------------

def load_gold_standard() -> Optional[pd.DataFrame]:
    """Load gold-standard labels from config.

    Returns:
        DataFrame with columns [geo_location, label, split], or None.
    """
    from ...config import APP_CONFIG

    gs_cfg = APP_CONFIG.get('gold_standard')
    if not gs_cfg or not gs_cfg.get('path'):
        return None

    path = gs_cfg['path']
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(project_root, path)

    if not os.path.exists(path):
        logger.warning(f"    Gold standard file not found: {path}")
        return None

    df = pd.read_csv(path)
    logger.info(f"    Loaded {len(df)} gold-standard labels from {path}")
    return df


# ===================================================================
# Main entry point
# ===================================================================

def classify_locations_deep(
    df: pd.DataFrame,
    feature_columns: List[str],
    compute_feature_importance: bool = False,
    feature_importance_output_dir: Optional[str] = None,
    input_parquet: Optional[str] = None,
    conn=None,
    llm_corrections: Optional[pd.DataFrame] = None,
    training_mode: str = 'gold_standard',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Classify locations using the learned deep pipeline.

    Args:
        df: DataFrame with location features.
        feature_columns: Available feature column names.
        compute_feature_importance: Log and optionally save feature importances.
        feature_importance_output_dir: Directory to save importance CSVs.
        input_parquet: Path to raw parquet (unused, kept for API compat).
        conn: DuckDB connection (unused, kept for API compat).
        llm_corrections: Optional DataFrame with LLM-annotated seed corrections.
            If None, attempts to load from config.yaml llm_corrections.path.
        training_mode: 'gold_standard' (v9, train on consensus labels only)
            or 'semi_supervised' (v8, heuristic seeds + LLM injection).

    Returns:
        (classified_df, empty_cluster_df) matching the legacy contract.
    """
    if training_mode == 'gold_standard':
        logger.info("=" * 70)
        logger.info("GOLD-STANDARD CLASSIFICATION (v9)")
        logger.info("  Pipeline: Gold-Standard Training → Hub Protection → Finalize")
        logger.info("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("SEMI-SUPERVISED CLASSIFICATION (v8)")
        logger.info("  Pipeline: Seed → LLM Refinement → Fusion → Hub Protection")
        logger.info("=" * 70)

    n_locations = len(df)
    logger.info(f"  Locations: {n_locations:,}")

    # ------------------------------------------------------------------
    # 0. Initialize output columns
    # ------------------------------------------------------------------
    df['user_category'] = 'normal'
    df['classification_confidence'] = 0.0
    df['needs_review'] = False
    df['behavior_type'] = 'organic'
    df['automation_category'] = None
    df['is_bot_neural'] = False
    df['is_protected_hub'] = False

    # Ensure total_downloads exists
    if 'total_downloads' not in df.columns or df['total_downloads'].eq(0).all():
        if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
            df['total_downloads'] = df['unique_users'] * df['downloads_per_user']

    # ------------------------------------------------------------------
    # Build feature matrix for ALL locations
    # ------------------------------------------------------------------
    available_behavioral = [c for c in BEHAVIORAL_FEATURE_COLS if c in df.columns]
    extra = [c for c in feature_columns
             if c in df.columns and c not in available_behavioral
             and c != 'time_series_features_present']
    all_feature_cols = list(dict.fromkeys(available_behavioral + extra))
    logger.info(f"    Using {len(all_feature_cols)} behavioural features")

    X_fusion_all = prepare_fusion_features(df, behavioral_cols=all_feature_cols)

    if training_mode == 'gold_standard':
        # ============================================================
        # v9: Gold-standard supervised training
        # ============================================================
        from ...config import APP_CONFIG

        logger.info("\n  Phase 1: Loading gold-standard labels ...")
        gs_df = load_gold_standard()
        if gs_df is None:
            raise ValueError(
                "Gold standard file not found. Set training_mode='semi_supervised' "
                "or provide gold_standard.path in config.yaml"
            )

        gs_cfg = APP_CONFIG.get('gold_standard', {})
        loc_col = gs_cfg.get('location_column', 'geo_location')
        label_col = gs_cfg.get('label_column', 'label')
        split_col = gs_cfg.get('split_column', 'split')

        # Match gold-standard locations to df indices
        label_map = {'bot': LABEL_BOT, 'hub': LABEL_HUB, 'organic': LABEL_ORGANIC}
        train_gs = gs_df[gs_df[split_col] == 'train']

        geo_to_idx = {}
        for idx, geo in df[loc_col].items():
            if geo not in geo_to_idx:
                geo_to_idx[geo] = idx

        train_indices = []
        train_labels = []
        for _, row in train_gs.iterrows():
            loc_idx = geo_to_idx.get(row[loc_col])
            if loc_idx is not None:
                train_indices.append(df.index.get_loc(loc_idx))
                train_labels.append(label_map[row[label_col]])

        train_indices = np.array(train_indices)
        train_labels = np.array(train_labels)

        n_bot = (train_labels == LABEL_BOT).sum()
        n_hub = (train_labels == LABEL_HUB).sum()
        n_org = (train_labels == LABEL_ORGANIC).sum()
        logger.info(f"    Training set: {len(train_indices)} gold-standard labels "
                    f"({n_org} organic, {n_bot} bot, {n_hub} hub)")

        logger.info("\n  Phase 2: Training gold-standard meta-learner ...")
        X_train = X_fusion_all[train_indices]
        meta_model, meta_scaler = train_meta_learner_gold_standard(X_train, train_labels)

    else:
        # ============================================================
        # v8: Semi-supervised (heuristic seeds + LLM injection)
        # ============================================================
        logger.info("\n  Phase 1: Seed selection ...")
        organic_seed = select_organic_seed(df)
        bot_seed = select_bot_seed(df)
        hub_seed = select_hub_seed(df)

        # Resolve seed overlaps: hub > bot > organic priority
        bot_hub_overlap = bot_seed.index.intersection(hub_seed.index)
        if len(bot_hub_overlap) > 0:
            bot_seed = bot_seed.drop(bot_hub_overlap)
            logger.info(f"    Removed {len(bot_hub_overlap)} bot seeds that overlap with hub seeds")

        bot_organic_overlap = bot_seed.index.intersection(organic_seed.index)
        if len(bot_organic_overlap) > 0:
            organic_seed = organic_seed.drop(bot_organic_overlap)
            logger.info(f"    Removed {len(bot_organic_overlap)} organic seeds that overlap with bot seeds")

        logger.info(f"    Seeds: {len(organic_seed)} organic, {len(bot_seed)} bot, {len(hub_seed)} hub")

        # LLM seed refinement
        if llm_corrections is None:
            llm_corrections = load_llm_corrections_from_config()
        if llm_corrections is not None and len(llm_corrections) > 0:
            logger.info("\n  Phase 2: LLM seed refinement ...")
            from ...config import APP_CONFIG
            llm_cfg = APP_CONFIG.get('llm_corrections', {})
            organic_seed, bot_seed, hub_seed = inject_llm_seeds(
                df, organic_seed, bot_seed, hub_seed, llm_corrections,
                label_column=llm_cfg.get('label_column', 'claude_evaluation'),
                location_column=llm_cfg.get('location_column', 'geo_location'),
                weight=llm_cfg.get('weight', 0.95),
            )

        # Assemble training data from seeds
        seed_indices, seed_labels, seed_weights = [], [], []
        for idx in organic_seed.index:
            seed_indices.append(df.index.get_loc(idx))
            seed_labels.append(LABEL_ORGANIC)
            seed_weights.append(organic_seed.loc[idx, 'seed_confidence'])
        for idx in bot_seed.index:
            seed_indices.append(df.index.get_loc(idx))
            seed_labels.append(LABEL_BOT)
            seed_weights.append(bot_seed.loc[idx, 'seed_confidence'])
        for idx in hub_seed.index:
            seed_indices.append(df.index.get_loc(idx))
            seed_labels.append(LABEL_HUB)
            seed_weights.append(hub_seed.loc[idx, 'seed_confidence'])

        seed_indices = np.array(seed_indices)
        seed_labels = np.array(seed_labels)
        seed_weights = np.array(seed_weights)

        logger.info(f"\n  Phase 3: Training fusion meta-learner ...")
        logger.info(f"    Training set: {len(seed_indices)} seeds "
                    f"({(seed_labels == LABEL_ORGANIC).sum()} organic, "
                    f"{(seed_labels == LABEL_BOT).sum()} bot, "
                    f"{(seed_labels == LABEL_HUB).sum()} hub)")

        X_train = X_fusion_all[seed_indices]
        meta_model, meta_scaler = train_meta_learner(X_train, seed_labels, weights=seed_weights)

    # ------------------------------------------------------------------
    # Predict on ALL locations (shared by both modes)
    # ------------------------------------------------------------------
    labels, confidences, probas = predict_with_confidence(
        meta_model, meta_scaler, X_fusion_all,
    )

    # Log feature importances
    if compute_feature_importance:
        feat_names = list(all_feature_cols)
        imp_df = get_feature_importances(meta_model, feat_names)
        if not imp_df.empty:
            logger.info("\n  Top 15 feature importances:")
            for _, row in imp_df.head(15).iterrows():
                logger.info(f"    {row['feature']:40s} {row['importance']:.4f}")
            if feature_importance_output_dir:
                os.makedirs(feature_importance_output_dir, exist_ok=True)
                imp_df.to_csv(os.path.join(feature_importance_output_dir,
                                           'fusion_importances.csv'), index=False)

    # ------------------------------------------------------------------
    # Map fusion labels → hierarchical classification
    # ------------------------------------------------------------------
    phase_label = "Phase 3" if training_mode == 'gold_standard' else "Phase 4"
    logger.info(f"\n  {phase_label}: Mapping predictions + Hub protection ...")

    df['classification_confidence'] = confidences
    df['prob_organic'] = probas[:, LABEL_ORGANIC]
    df['prob_bot'] = probas[:, LABEL_BOT]
    df['prob_hub'] = probas[:, LABEL_HUB]

    # User (organic)
    organic_mask = labels == LABEL_ORGANIC
    df.loc[df.index[organic_mask], 'user_category'] = 'normal'
    df.loc[df.index[organic_mask], 'behavior_type'] = 'user'
    df.loc[df.index[organic_mask], 'automation_category'] = None

    # Refine → independent_user for small locations
    if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
        indep_mask = (
            organic_mask &
            (df['unique_users'].values <= 10) &
            (df['downloads_per_user'].values <= 5)
        )
        df.loc[df.index[indep_mask], 'user_category'] = 'independent_user'

    # Bot
    bot_mask = labels == LABEL_BOT
    df.loc[df.index[bot_mask], 'user_category'] = 'bot'
    df.loc[df.index[bot_mask], 'behavior_type'] = 'bot'
    df.loc[df.index[bot_mask], 'automation_category'] = 'bot'
    df.loc[df.index[bot_mask], 'is_bot_neural'] = True

    # Hub
    hub_mask = labels == LABEL_HUB
    df.loc[df.index[hub_mask], 'user_category'] = 'download_hub'
    df.loc[df.index[hub_mask], 'behavior_type'] = 'hub'
    df.loc[df.index[hub_mask], 'automation_category'] = 'legitimate_automation'

    # Flag low-confidence predictions for review
    df.loc[confidences < 0.5, 'needs_review'] = True

    log_prediction_summary(df, labels, confidences)

    # Hub protection (structural override)
    logger.info(f"\n  {phase_label}b: Hub protection ...")
    df = apply_hub_protection(df)

    # ------------------------------------------------------------------
    # Derive boolean columns & finalize
    # ------------------------------------------------------------------
    final_phase = "Phase 4" if training_mode == 'gold_standard' else "Phase 5"
    logger.info(f"\n  {final_phase}: Deriving final labels ...")

    if 'total_downloads' in df.columns:
        insufficient_mask = df['total_downloads'] < 3
        df.loc[insufficient_mask, 'behavior_type'] = 'insufficient_evidence'
        df.loc[insufficient_mask, 'automation_category'] = None

    df['is_bot'] = df['automation_category'] == 'bot'
    df['is_hub'] = df['automation_category'] == 'legitimate_automation'
    df['is_organic'] = df['behavior_type'] == 'user'
    df['is_download_hub'] = df['is_hub']  # Legacy alias
    log_hierarchical_summary(df)

    cluster_df = pd.DataFrame()
    return df, cluster_df
