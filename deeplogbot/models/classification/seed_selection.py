"""Seed selection for learned bot detection.

Selects high-confidence organic and bot locations from behavioral features
to train the meta-learner. Uses a 3-tier system for organic locations,
multi-signal heuristics for bot locations, and structural patterns for hubs.

No hard-coded thresholds are exported -- all thresholds are internal to
seed selection and are only used to identify training examples, not to
classify production data.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum downloads to be a reliable seed (below this, features are noisy)
MIN_SEED_DOWNLOADS = 20


def select_organic_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence organic locations for meta-learner training.

    Uses a 3-tier system:
      Tier A: Individual researchers (very high confidence)
      Tier B: Active researchers (high confidence)
      Tier C: Research groups (moderate confidence)

    Each tier gets a confidence weight used during training.

    Args:
        df: DataFrame with location features

    Returns:
        DataFrame subset with added 'seed_confidence' column (0-1)
    """
    def has(*cols):
        return all(c in df.columns for c in cols)

    # ------------------------------------------------------------------
    # Tier A: Individual researchers -- very high confidence
    # ------------------------------------------------------------------
    tier_a = pd.Series(True, index=df.index)
    if has('unique_users'):
        tier_a &= df['unique_users'] <= 10
    if has('downloads_per_user'):
        tier_a &= df['downloads_per_user'] <= 5
    if has('total_downloads'):
        tier_a &= df['total_downloads'].between(MIN_SEED_DOWNLOADS, 200)
    if has('working_hours_ratio'):
        tier_a &= df['working_hours_ratio'] > 0.3
    if has('night_activity_ratio'):
        tier_a &= df['night_activity_ratio'] < 0.5
    if has('years_span'):
        tier_a &= df['years_span'] >= 2

    # ------------------------------------------------------------------
    # Tier B: Active researchers -- high confidence
    # ------------------------------------------------------------------
    tier_b = pd.Series(True, index=df.index)
    if has('unique_users'):
        tier_b &= df['unique_users'] <= 50
    if has('downloads_per_user'):
        tier_b &= df['downloads_per_user'] <= 10
    if has('total_downloads'):
        tier_b &= df['total_downloads'].between(MIN_SEED_DOWNLOADS, 1000)
    if has('working_hours_ratio'):
        tier_b &= df['working_hours_ratio'] > 0.25
    if has('hourly_entropy'):
        tier_b &= df['hourly_entropy'] > 1.5
    if has('burst_pattern_score'):
        tier_b &= df['burst_pattern_score'] < 0.5
    # Exclude tier A (already captured)
    tier_b &= ~tier_a

    # ------------------------------------------------------------------
    # Tier C: Research groups -- moderate confidence
    # ------------------------------------------------------------------
    tier_c = pd.Series(True, index=df.index)
    if has('unique_users'):
        tier_c &= df['unique_users'] <= 200
    if has('downloads_per_user'):
        tier_c &= df['downloads_per_user'] <= 20
    if has('total_downloads'):
        tier_c &= df['total_downloads'] >= MIN_SEED_DOWNLOADS
    if has('user_coordination_score'):
        tier_c &= df['user_coordination_score'] < 0.3
    if has('protocol_legitimacy_score'):
        tier_c &= df['protocol_legitimacy_score'] > 0.3
    # Exclude locations that appeared only in the latest year with many users
    # (these are likely distributed bot-farm locations, not research groups)
    if has('fraction_latest_year', 'unique_users'):
        tier_c &= ~((df['fraction_latest_year'] > 0.9) & (df['unique_users'] > 50))
    # Require multi-year activity for larger groups
    if has('years_span', 'unique_users'):
        tier_c &= ~((df['years_span'] < 2) & (df['unique_users'] > 30))
    # Exclude tiers A and B
    tier_c &= ~tier_a & ~tier_b

    # Combine with confidence weights
    seed_mask = tier_a | tier_b | tier_c
    seed_df = df.loc[seed_mask].copy()

    seed_df['seed_confidence'] = 0.0
    seed_df.loc[tier_a[seed_mask].values, 'seed_confidence'] = 1.0
    seed_df.loc[tier_b[seed_mask].values, 'seed_confidence'] = 0.7
    seed_df.loc[tier_c[seed_mask].values, 'seed_confidence'] = 0.4

    logger.info(f"Organic seed: {tier_a.sum()} Tier A, {tier_b.sum()} Tier B, "
                f"{tier_c.sum()} Tier C = {len(seed_df)} total")
    return seed_df


def _is_hub_like(df: pd.DataFrame) -> pd.Series:
    """Identify locations with hub-like patterns that should NOT be bot seeds.

    Hub pattern: high downloads_per_user over multiple years. These are
    institutional mirrors/bulk-downloaders, not bot farms.
    """
    hub_like = pd.Series(False, index=df.index)
    if 'downloads_per_user' in df.columns and 'years_span' in df.columns:
        hub_like = (df['downloads_per_user'] > 200) & (df['years_span'] >= 3)
    return hub_like


def select_bot_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence bot locations for meta-learner training.

    Uses strong behavioral signals that are very unlikely to be organic.
    Requires minimum download volume for reliable feature estimation and
    excludes hub-like patterns (high DL/user over years).

    Args:
        df: DataFrame with location features

    Returns:
        DataFrame subset with added 'seed_confidence' column (0-1)
    """
    def has(*cols):
        return all(c in df.columns for c in cols)

    # Minimum volume: seeds must have enough data for features to be meaningful
    volume_filter = pd.Series(True, index=df.index)
    if has('total_downloads'):
        volume_filter = df['total_downloads'] >= MIN_SEED_DOWNLOADS

    # Exclude hub-like locations from bot seeds
    hub_like = _is_hub_like(df)

    # Strong bot signal: many users with low DL/user (distributed bot farm)
    bot_farm = pd.Series(False, index=df.index)
    if has('unique_users', 'downloads_per_user'):
        bot_farm = (df['unique_users'] > 5000) & (df['downloads_per_user'] < 50)

    # Medium bot signal: moderate users with very low DL/user (distributed bot network)
    # These are the "long tail" bot locations with 500-5000 users doing 3-10 DL each
    distributed_bot = pd.Series(False, index=df.index)
    if has('unique_users', 'downloads_per_user'):
        distributed_bot = (
            (df['unique_users'] > 500) &
            (df['downloads_per_user'] < 15) &
            (df['downloads_per_user'] > 2)  # exclude very low activity
        )
        # Require uniform temporal pattern (bots don't follow circadian rhythm)
        if has('working_hours_ratio'):
            distributed_bot &= df['working_hours_ratio'] < 0.38
        # Require activity concentrated in latest year (sudden appearance)
        if has('fraction_latest_year'):
            distributed_bot &= df['fraction_latest_year'] > 0.8

    # Strong bot signal: extreme nocturnal + no working hours
    # Require meaningful activity to avoid tiny locations that happen to be at night
    nocturnal = pd.Series(False, index=df.index)
    if has('night_activity_ratio', 'working_hours_ratio'):
        nocturnal = (
            (df['night_activity_ratio'] > 0.8) &
            (df['working_hours_ratio'] < 0.1) &
            volume_filter
        )

    # Strong bot signal: massive user count (coordinated)
    coordinated = pd.Series(False, index=df.index)
    if has('unique_users', 'downloads_per_user'):
        coordinated = (df['unique_users'] > 10000) & (df['downloads_per_user'] < 20)

    # Strong bot signal: scraper (accesses huge % of all datasets)
    scraper = pd.Series(False, index=df.index)
    if has('unique_projects'):
        scraper = df['unique_projects'] > 15000

    # Strong bot signal: explosive year-over-year growth with many users
    yoy_explosion = pd.Series(False, index=df.index)
    if has('spike_ratio', 'unique_users', 'fraction_latest_year'):
        yoy_explosion = (
            (df['spike_ratio'] > 50) &        # >50x growth vs previous years
            (df['unique_users'] > 200) &
            (df['fraction_latest_year'] > 0.95)  # almost all activity in latest year
        )

    bot_mask = (
        (bot_farm | distributed_bot | nocturnal | coordinated | scraper | yoy_explosion)
        & volume_filter   # all seeds need minimum downloads
        & ~hub_like       # never use hub-like locations as bot seeds
    )
    bot_df = df.loc[bot_mask].copy()

    # Assign confidence based on signal strength
    bot_df['seed_confidence'] = 0.7  # base confidence
    if has('unique_users'):
        bot_df.loc[bot_df['unique_users'] > 10000, 'seed_confidence'] = 0.9
    bot_df.loc[bot_farm[bot_mask].values & nocturnal[bot_mask].values, 'seed_confidence'] = 0.95
    # Pure distributed bots (no stronger signal) get moderate confidence
    pure_distributed = (
        distributed_bot[bot_mask].values
        & ~bot_farm[bot_mask].values
        & ~nocturnal[bot_mask].values
        & ~coordinated[bot_mask].values
        & ~scraper[bot_mask].values
    )
    bot_df.loc[pure_distributed, 'seed_confidence'] = 0.6

    logger.info(f"Bot seed: {(bot_farm & volume_filter & ~hub_like).sum()} bot farm, "
                f"{(distributed_bot & volume_filter & ~hub_like).sum()} distributed, "
                f"{(nocturnal & volume_filter & ~hub_like).sum()} nocturnal, "
                f"{(coordinated & volume_filter & ~hub_like).sum()} coordinated, "
                f"{(scraper & volume_filter & ~hub_like).sum()} scraper, "
                f"{(yoy_explosion & volume_filter & ~hub_like).sum()} yoy explosion = {len(bot_df)} total")
    return bot_df


def select_hub_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence hub locations.

    Hubs: high DL/user sustained over multiple years, legitimate protocols.
    Includes both small mirrors (few users, very high DL/user) and larger
    institutional hubs (moderate users, high DL/user, multi-year).

    Args:
        df: DataFrame with location features

    Returns:
        DataFrame subset with added 'seed_confidence' column (0-1)
    """
    def has(*cols):
        return all(c in df.columns for c in cols)

    # Classic small mirror: few users, very high DL/user
    small_mirror = pd.Series(False, index=df.index)
    if has('downloads_per_user', 'unique_users'):
        small_mirror = (df['downloads_per_user'] > 500) & (df['unique_users'] < 100)

    # Institutional hub: moderate users, high DL/user, multi-year
    institutional = pd.Series(False, index=df.index)
    if has('downloads_per_user', 'unique_users', 'years_span'):
        institutional = (
            (df['downloads_per_user'] > 200) &
            (df['unique_users'] <= 1000) &
            (df['years_span'] >= 3)
        )

    hub_mask = small_mirror | institutional

    # Exclude nocturnal behavior (likely bot, not hub)
    if has('working_hours_ratio', 'night_activity_ratio'):
        bot_behavior = (df['working_hours_ratio'] < 0.1) & (df['night_activity_ratio'] > 0.7)
        hub_mask &= ~bot_behavior

    # Require minimum downloads
    if has('total_downloads'):
        hub_mask &= df['total_downloads'] >= MIN_SEED_DOWNLOADS

    hub_df = df.loc[hub_mask].copy()
    hub_df['seed_confidence'] = 0.8

    # Boost confidence for protocol-verified hubs
    if has('aspera_ratio', 'globus_ratio'):
        protocol_verified = (hub_df['aspera_ratio'] > 0.3) | (hub_df['globus_ratio'] > 0.1)
        hub_df.loc[protocol_verified, 'seed_confidence'] = 0.95

    # Boost confidence for multi-year institutional hubs
    if 'years_span' in hub_df.columns:
        long_running = hub_df['years_span'] >= 4
        hub_df.loc[long_running & (hub_df['seed_confidence'] < 0.9), 'seed_confidence'] = 0.85

    logger.info(f"Hub seed: {small_mirror.sum()} small mirrors, "
                f"{(institutional & ~small_mirror).sum()} institutional = "
                f"{len(hub_df)} total (after exclusions)")
    return hub_df
