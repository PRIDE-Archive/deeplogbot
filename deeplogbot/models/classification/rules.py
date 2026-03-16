"""Rule-based classification for bot and download hub detection.

This module classifies locations into three categories: bot, hub, or organic.
Stage 1 separates organic from automated traffic. Stage 2 distinguishes
bots from legitimate automation (hubs) within the automated locations.
Stage 3 applies hub protection rules (shared with the deep pipeline).

Configuration is read from `config.yaml`.
"""

import numpy as np
import pandas as pd

from ...utils import logger
from ...config import (
    get_behavior_type_rules,
    get_automation_category_rules,
    get_taxonomy_info,
    get_classification_config,
)


def derive_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive is_bot, is_hub, is_organic boolean columns from classification."""
    if 'automation_category' in df.columns:
        df['is_bot'] = df['automation_category'] == 'bot'
        df['is_hub'] = df['automation_category'] == 'legitimate_automation'
    else:
        df['is_bot'] = False
        df['is_hub'] = False

    if 'behavior_type' in df.columns:
        df['is_organic'] = df['behavior_type'] == 'organic'
    else:
        df['is_organic'] = ~df['is_bot'] & ~df['is_hub']

    return df


# =====================================================================
# Confidence scoring
# =====================================================================

def _compute_rule_confidence(df: pd.DataFrame) -> pd.Series:
    """Compute a heuristic confidence score for each rule-based classification.

    The score is based on how many pattern signals agree with the final label
    and how extreme the feature values are relative to the thresholds.
    Range: 0.0 -- 1.0.
    """
    confidence = pd.Series(0.5, index=df.index)

    # Bot confidence boosters
    bot_mask = df['automation_category'] == 'bot'
    if bot_mask.any() and 'unique_users' in df.columns:
        # More users + lower DL/user = higher bot confidence
        user_signal = np.clip(np.log10(df.loc[bot_mask, 'unique_users'] + 1) / 5, 0, 1)
        dl_signal = np.clip(1 - df.loc[bot_mask, 'downloads_per_user'] / 200, 0, 1)
        confidence.loc[bot_mask] = 0.4 + 0.3 * user_signal + 0.3 * dl_signal

    # Hub confidence boosters
    hub_mask = df['automation_category'] == 'legitimate_automation'
    if hub_mask.any() and 'downloads_per_user' in df.columns:
        dl_signal = np.clip(np.log10(df.loc[hub_mask, 'downloads_per_user'] + 1) / 4, 0, 1)
        confidence.loc[hub_mask] = 0.4 + 0.6 * dl_signal

    # Organic confidence: working hours + low user count
    organic_mask = df['behavior_type'] == 'organic'
    if organic_mask.any():
        signals = []
        if 'working_hours_ratio' in df.columns:
            signals.append(np.clip(df.loc[organic_mask, 'working_hours_ratio'], 0, 1))
        if 'unique_users' in df.columns:
            signals.append(np.clip(1 - np.log10(df.loc[organic_mask, 'unique_users'] + 1) / 4, 0, 1))
        if signals:
            avg = sum(signals) / len(signals)
            confidence.loc[organic_mask] = 0.3 + 0.7 * avg

    return np.clip(confidence, 0.0, 1.0)


# =====================================================================
# Pattern matching
# =====================================================================

_warned_missing_fields: set = set()


def _match_pattern(df: pd.DataFrame, pattern: dict) -> pd.Series:
    """Match a single pattern against the DataFrame.

    Returns a boolean Series indicating which rows match the pattern.
    Logs a warning (once per field) when a feature column referenced
    by a pattern is missing from the DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    for field, spec in pattern.items():
        if field in ("id", "description", "parent"):
            continue

        if field not in df.columns:
            if field not in _warned_missing_fields:
                logger.warning(f"Pattern field '{field}' not in DataFrame — "
                               f"pattern '{pattern.get('id', '?')}' will not match")
                _warned_missing_fields.add(field)
            mask &= False
            continue

        col = df[field]
        if isinstance(spec, dict):
            min_val = spec.get("min", None)
            max_val = spec.get("max", None)
            if min_val is not None:
                mask &= col >= min_val
            if max_val is not None:
                mask &= col <= max_val
        else:
            mask &= col == spec

    return mask


def _match_any_pattern(df: pd.DataFrame, patterns: list) -> pd.Series:
    """Match any pattern from a list (OR logic)."""
    if not patterns:
        return pd.Series(False, index=df.index)

    overall = pd.Series(False, index=df.index)
    for pattern in patterns:
        overall |= _match_pattern(df, pattern)
    return overall


# =====================================================================
# Hierarchical Classification Functions
# =====================================================================

def classify_behavior_type(df: pd.DataFrame) -> pd.DataFrame:
    """Level 1: Classify locations as ORGANIC or AUTOMATED.

    Adds 'behavior_type' column with values: 'organic' or 'automated'.
    """
    behavior_rules = get_behavior_type_rules()

    df['behavior_type'] = 'unknown'

    organic_patterns = behavior_rules.get('organic', {}).get('patterns', [])
    automated_patterns = behavior_rules.get('automated', {}).get('patterns', [])

    automated_mask = _match_any_pattern(df, automated_patterns)
    organic_mask = _match_any_pattern(df, organic_patterns)

    # Automated takes precedence if both match
    df.loc[organic_mask & ~automated_mask, 'behavior_type'] = 'organic'
    df.loc[automated_mask, 'behavior_type'] = 'automated'

    # Fallback defaults from config
    fallback = get_classification_config().get('fallback_thresholds', {})
    organic_max_users = fallback.get('organic_max_users', 100)

    unknown_mask = df['behavior_type'] == 'unknown'
    if unknown_mask.any():
        n_unknown = unknown_mask.sum()
        default_organic = unknown_mask & (df['unique_users'] <= organic_max_users)
        default_automated = unknown_mask & (df['unique_users'] > organic_max_users)
        df.loc[default_organic, 'behavior_type'] = 'organic'
        df.loc[default_automated, 'behavior_type'] = 'automated'
        logger.info(f"  Fallback classification applied to {n_unknown:,} locations "
                    f"(threshold: unique_users <= {organic_max_users})")

    return df


def classify_automation_category(df: pd.DataFrame) -> pd.DataFrame:
    """Level 2: Classify AUTOMATED locations as BOT or LEGITIMATE_AUTOMATION.

    Only applies to locations where behavior_type == 'automated'.
    """
    automation_rules = get_automation_category_rules()

    df['automation_category'] = None

    automated_mask = df['behavior_type'] == 'automated'
    if not automated_mask.any():
        return df

    automated_df = df[automated_mask]

    bot_patterns = automation_rules.get('bot', {}).get('patterns', [])
    legitimate_patterns = automation_rules.get('legitimate_automation', {}).get('patterns', [])

    bot_mask = _match_any_pattern(automated_df, bot_patterns)
    legitimate_mask = _match_any_pattern(automated_df, legitimate_patterns)

    # Legitimate automation takes precedence if both match
    df.loc[automated_df[bot_mask & ~legitimate_mask].index, 'automation_category'] = 'bot'
    df.loc[automated_df[legitimate_mask].index, 'automation_category'] = 'legitimate_automation'

    # Fallback heuristics from config
    fallback = get_classification_config().get('fallback_thresholds', {})
    bot_min_users = fallback.get('bot_min_users', 500)
    bot_max_dl_per_user = fallback.get('bot_max_dl_per_user', 100)
    hub_min_dl_per_user = fallback.get('hub_min_dl_per_user', 100)

    unclassified = automated_mask & df['automation_category'].isna()
    if unclassified.any():
        heuristic_bot = (unclassified
                         & (df['unique_users'] > bot_min_users)
                         & (df['downloads_per_user'] < bot_max_dl_per_user))
        heuristic_legitimate = (unclassified
                                & (df['downloads_per_user'] >= hub_min_dl_per_user))
        df.loc[heuristic_bot, 'automation_category'] = 'bot'
        df.loc[heuristic_legitimate, 'automation_category'] = 'legitimate_automation'

    # Remaining automated locations default to bot
    remaining = automated_mask & df['automation_category'].isna()
    n_remaining = remaining.sum()
    if n_remaining > 0:
        df.loc[remaining, 'automation_category'] = 'bot'
        logger.info(f"  {n_remaining:,} automated locations defaulted to bot")

    return df


def classify_locations_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """Classify locations as bot, hub, or organic using config-driven rules.

    Stage 1: Separate organic from automated traffic.
    Stage 2: Distinguish bots from hubs (legitimate automation).
    Stage 3: Apply hub protection (shared structural rules).

    Returns DataFrame with is_bot, is_hub, is_organic, classification_confidence.
    """
    # Ensure expected columns exist
    if "total_downloads" not in df.columns:
        df["total_downloads"] = df["unique_users"] * df["downloads_per_user"]

    taxonomy = get_taxonomy_info()
    logger.info(f"Using taxonomy: {taxonomy.get('name', 'default')} "
                f"v{taxonomy.get('version', '1.0')}")

    # Stage 1
    logger.info("Stage 1: Classifying organic vs automated...")
    df = classify_behavior_type(df)

    # Stage 2
    logger.info("Stage 2: Classifying bot vs hub (legitimate automation)...")
    df = classify_automation_category(df)

    # Derive boolean columns
    df = derive_boolean_columns(df)

    # Stage 3: Hub protection (same rules used by the deep pipeline)
    from .post_classification import apply_hub_protection
    logger.info("Stage 3: Applying hub protection rules...")
    df = apply_hub_protection(df)
    # Update booleans after hub protection overrides
    df['is_bot'] = df['automation_category'] == 'bot'
    df['is_hub'] = df['automation_category'] == 'legitimate_automation'
    df['is_organic'] = df['behavior_type'] == 'organic'

    # Confidence scoring
    df['classification_confidence'] = _compute_rule_confidence(df)

    _log_classification_summary(df)
    return df


def _log_classification_summary(df: pd.DataFrame) -> None:
    """Log a summary of the hierarchical classification results."""
    total = len(df)

    logger.info("\nHierarchical Classification Summary:")
    logger.info("=" * 50)

    logger.info("\nLevel 1 - Behavior Type:")
    for bt in ['organic', 'automated']:
        count = (df['behavior_type'] == bt).sum()
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {bt.upper()}: {count:,} locations ({pct:.1f}%)")

    automated_count = (df['behavior_type'] == 'automated').sum()
    if automated_count > 0:
        logger.info("\nLevel 2 - Automation Category (within AUTOMATED):")
        for ac in ['bot', 'legitimate_automation']:
            count = (df['automation_category'] == ac).sum()
            pct = count / automated_count * 100 if automated_count > 0 else 0
            logger.info(f"  {ac.upper()}: {count:,} locations ({pct:.1f}% of automated)")

    if 'is_bot' in df.columns:
        bot_count = df['is_bot'].sum()
        hub_count = df['is_hub'].sum() if 'is_hub' in df.columns else 0
        organic_count = df['is_organic'].sum() if 'is_organic' in df.columns else total - bot_count - hub_count
        protected = df['is_protected_hub'].sum() if 'is_protected_hub' in df.columns else 0
        logger.info(f"\nFinal: {bot_count:,} bot, {hub_count:,} hub "
                    f"({protected:,} protected), {organic_count:,} organic")
