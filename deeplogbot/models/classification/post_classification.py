"""Post-classification steps shared across classification methods.

Includes hub protection (structural override) and logging summaries.
These are rule-based refinements applied *after* the learned pipeline
produces initial labels.
"""

import numpy as np
import pandas as pd

from ...utils import logger
from ...config import get_hub_protection_rules

from .fusion import LABEL_NAMES


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _has_required_columns(df: pd.DataFrame, *columns: str) -> bool:
    """Check if DataFrame has all required columns."""
    return all(col in df.columns for col in columns)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_prediction_summary(df: pd.DataFrame, labels: np.ndarray,
                           confidences: np.ndarray) -> None:
    """Log fusion prediction summary."""
    n = len(df)
    for lbl, name in LABEL_NAMES.items():
        mask = labels == lbl
        count = mask.sum()
        mean_conf = confidences[mask].mean() if count > 0 else 0
        logger.info(f"    {name.upper():8s}: {count:,} ({count/n*100:.1f}%), "
                    f"mean confidence {mean_conf:.3f}")

    low_conf = (confidences < 0.5).sum()
    logger.info(f"    Low confidence (<0.5): {low_conf:,} ({low_conf/n*100:.1f}%)")


def log_hierarchical_summary(df: pd.DataFrame) -> None:
    """Log hierarchical classification summary."""
    total = len(df)
    if total == 0:
        return

    logger.info("\n  " + "=" * 60)
    logger.info("  HIERARCHICAL CLASSIFICATION SUMMARY")
    logger.info("  " + "=" * 60)

    logger.info("\n  Classification:")
    for bt in ['user', 'bot', 'hub', 'insufficient_evidence']:
        count = (df['behavior_type'] == bt).sum()
        pct = count / total * 100
        if count > 0:
            dl = df.loc[df['behavior_type'] == bt, 'total_downloads'].sum() if 'total_downloads' in df.columns else 0
            logger.info(f"    {bt.upper():25s}: {count:>7,} ({pct:5.1f}%) — {dl:>14,.0f} DL")

    # Final category counts
    if 'is_bot' in df.columns:
        bot_count = df['is_bot'].sum() if 'is_bot' in df.columns else 0
        hub_count = df['is_hub'].sum() if 'is_hub' in df.columns else 0
        organic_count = df['is_organic'].sum() if 'is_organic' in df.columns else total - bot_count - hub_count
        logger.info(f"\n  Final: {bot_count:,} bot, {hub_count:,} hub, {organic_count:,} organic")


# ---------------------------------------------------------------------------
# Hub protection
# ---------------------------------------------------------------------------

def apply_hub_protection(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strict hub protection rules.

    Definite hub patterns should NEVER be classified as bots.
    Uses structural signals (few users, very high DL/user, legitimate protocols)
    that are reliable regardless of the learned model's output.
    """
    hub_rules = get_hub_protection_rules()

    if 'is_protected_hub' not in df.columns:
        df['is_protected_hub'] = False

    definite_hub_mask = pd.Series(False, index=df.index)

    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        behavioral_rules = hub_rules.get('behavioral_exclusion', {})

        # Behavioural exclusion: don't protect if clearly bot-like
        behavioral_exclusion = pd.Series(False, index=df.index)
        if _has_required_columns(df, 'working_hours_ratio', 'night_activity_ratio'):
            behavioral_exclusion = (
                (df['working_hours_ratio'] < behavioral_rules.get('max_working_hours_ratio', 0.1)) &
                (df['night_activity_ratio'] > behavioral_rules.get('min_night_activity_ratio', 0.7))
            )

        # High single-project concentration exclusion: locations where >80%
        # of downloads come from one project are CI/CD, not legitimate hubs
        concentration_exclusion = pd.Series(False, index=df.index)
        if 'top_project_concentration' in df.columns:
            concentration_exclusion = df['top_project_concentration'] > 0.8

        # Rule 1: Protocol-based hub detection (aspera/globus = institutional tools)
        protocol_hub = pd.Series(False, index=df.index)
        if _has_required_columns(df, 'aspera_ratio', 'globus_ratio'):
            protocol_hub = (df['aspera_ratio'] > 0.3) | (df['globus_ratio'] > 0.1)

        # Rule 2: Extreme mirrors — very high DL/user with few users
        high_dl_rule = hub_rules.get('high_dl_per_user', {})
        extreme_mirror = (
            (df['downloads_per_user'] > high_dl_rule.get('min_downloads_per_user', 500)) &
            (df['unique_users'] <= high_dl_rule.get('max_users', 200))
        )

        # Combine: protocol hubs + extreme mirrors, excluding bot-like and CI/CD
        definite_hub_mask = (
            protocol_hub | extreme_mirror
        ) & ~behavioral_exclusion & ~concentration_exclusion

    df.loc[definite_hub_mask, 'is_protected_hub'] = True
    df.loc[definite_hub_mask, 'is_bot_neural'] = False
    df.loc[definite_hub_mask, 'behavior_type'] = 'hub'
    df.loc[definite_hub_mask, 'automation_category'] = 'legitimate_automation'
    if 'user_category' in df.columns:
        df.loc[definite_hub_mask, 'user_category'] = 'download_hub'

    n_protected = definite_hub_mask.sum()
    if n_protected > 0:
        logger.info(f"    Hub protection: {n_protected:,} locations protected")

    # ------------------------------------------------------------------
    # Suspicious hub demotion: reclassify "hubs" with concentrated project
    # downloads as bots. Legitimate mirrors download broadly; locations
    # where a few projects dominate traffic are scrapers or CI/CD, not hubs.
    # ------------------------------------------------------------------
    hub_mask = df['behavior_type'] == 'hub'
    suspicious_hub = pd.Series(False, index=df.index)

    if _has_required_columns(df, 'top3_project_concentration', 'project_hhi'):
        # Standard threshold: top-3 projects > 50% AND HHI > 0.05
        standard_suspicious = (
            hub_mask &
            (df['top3_project_concentration'] > 0.5) &
            (df['project_hhi'] > 0.05)
        )
        # Volume-aware threshold: for high-volume locations (>100K downloads),
        # lower the bar to 45% — concentrated downloads at massive scale
        # are inconsistent with legitimate mirroring behavior
        high_volume_suspicious = pd.Series(False, index=df.index)
        if 'total_downloads' in df.columns:
            high_volume_suspicious = (
                hub_mask &
                (df['total_downloads'] > 100_000) &
                (df['top3_project_concentration'] > 0.45) &
                (df['project_hhi'] > 0.05)
            )
        suspicious_hub = standard_suspicious | high_volume_suspicious
    elif _has_required_columns(df, 'top3_project_concentration'):
        suspicious_hub = hub_mask & (df['top3_project_concentration'] > 0.5)

    n_demoted = suspicious_hub.sum()
    if n_demoted > 0:
        df.loc[suspicious_hub, 'behavior_type'] = 'bot'
        df.loc[suspicious_hub, 'automation_category'] = 'bot'
        df.loc[suspicious_hub, 'is_bot_neural'] = True
        df.loc[suspicious_hub, 'is_protected_hub'] = False
        if 'user_category' in df.columns:
            df.loc[suspicious_hub, 'user_category'] = 'bot'
        logger.info(f"    Suspicious hub demotion: {n_demoted:,} hubs → bot "
                    f"(concentrated project downloads)")

    return df


