"""
src/signals/factors.py
──────────────────────
Quantitative factor signals:
  1. Cross-sectional Momentum (12-1 month)
  2. Short-term Reversal (1-month, volume-filtered)
  3. Cross-asset features (sector relative strength, beta, market cap rank)

All signals are:
  - Computed with no look-ahead bias
  - Cross-sectionally ranked (rank-transformed) at each date
  - Scaled to [-1, 1] (long top, short bottom)
"""

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Cross-sectional rank transform
# ─────────────────────────────────────────────

def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    At each date, rank-transform values across tickers.
    Output in [-1, 1]: top ticker = +1, bottom = -1.
    """
    ranked = df.rank(axis=1, pct=True)   # [0, 1]
    return 2 * ranked - 1               # [-1, 1]


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score cross-sectionally at each date."""
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


# ─────────────────────────────────────────────
#  1. Momentum (12-1 month)
# ─────────────────────────────────────────────

def momentum_signal(
    close: pd.DataFrame,
    lookback_long: int = 252,
    lookback_skip: int = 21,
) -> pd.DataFrame:
    """
    Cross-sectional momentum: 12-month return excluding last month.
    At time T: return from T-252 to T-21.
    Ranked cross-sectionally. Positive = outperformer.

    Economic rationale: stocks that outperformed over the past year
    (excluding last month) continue to outperform over the next 1-3 months.
    Documented by Jegadeesh & Titman (1993), robust globally.
    """
    log_close = np.log(close)
    # Return from T-lookback_long to T-lookback_skip
    long_ret  = log_close.shift(lookback_skip) - log_close.shift(lookback_long)
    return cross_sectional_rank(long_ret)


# ─────────────────────────────────────────────
#  2. Short-term Reversal
# ─────────────────────────────────────────────

def reversal_signal(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 21,
    volume_threshold: float = 1.5,
    volume_window: int = 20,
) -> pd.DataFrame:
    """
    1-month reversal with volume filter.
    Negative last-month return + high volume = strong reversal candidate.

    Economic rationale: short-term mean reversion is driven by
    over-reaction + liquidity demand. High-volume down moves are most
    likely to revert as selling pressure exhausts.
    """
    log_close = np.log(close)
    ret_1m  = log_close - log_close.shift(lookback)
    neg_ret = -ret_1m   # flip sign: low return → high score

    # Volume filter: compute abnormal volume flag
    avg_vol     = volume.rolling(volume_window, min_periods=10).mean()
    vol_ratio   = volume / (avg_vol + 1e-10)
    high_vol_flag = (vol_ratio >= volume_threshold).astype(float)

    # Score = reversal magnitude * volume flag
    # (if volume filter disabled, just return neg_ret ranked)
    score = neg_ret * high_vol_flag
    # Fall back to unfiltered for low-volume stocks (at least some signal)
    score = score.where(score != 0, neg_ret * 0.5)

    return cross_sectional_rank(score)


# ─────────────────────────────────────────────
#  3. Cross-asset / sector features
# ─────────────────────────────────────────────

def sector_relative_strength(
    close: pd.DataFrame,
    sector_map: dict[str, str],   # {ticker: sector}
    window: int = 63,
) -> pd.DataFrame:
    """
    Stock return relative to its sector median return over window days.
    Positive = outperforming sector peers.
    """
    ret = close.pct_change(window)
    result = ret.copy() * np.nan

    for sector in set(sector_map.values()):
        tickers_in_sector = [t for t, s in sector_map.items() if s == sector and t in ret.columns]
        if not tickers_in_sector:
            continue
        sector_ret = ret[tickers_in_sector]
        sector_median = sector_ret.median(axis=1)
        for ticker in tickers_in_sector:
            result[ticker] = ret[ticker] - sector_median

    return cross_sectional_rank(result)


def market_beta(
    close: pd.DataFrame,
    market_close: pd.Series,   # e.g. Nifty 50 index
    window: int = 126,
) -> pd.DataFrame:
    """
    Rolling beta of each stock to the market index.
    Used as a risk factor, not a signal directly.
    """
    stock_ret  = np.log(close / close.shift(1))
    market_ret = np.log(market_close / market_close.shift(1))

    betas = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for ticker in close.columns:
        cov   = stock_ret[ticker].rolling(window).cov(market_ret)
        m_var = market_ret.rolling(window).var()
        betas[ticker] = cov / (m_var + 1e-12)

    return betas


def size_rank(
    volume: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Rolling market cap proxy (price × volume momentum) cross-sectional rank.
    Large cap = high rank. Used as a factor in the ML model.
    """
    market_cap_proxy = close * volume.rolling(window).mean()
    return cross_sectional_rank(market_cap_proxy)


# ─────────────────────────────────────────────
#  4. Multi-horizon return features (for ML)
# ─────────────────────────────────────────────

def multi_horizon_returns(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns over multiple past horizons — used as ML features.
    All are PAST returns at time T (no lookahead).
    """
    log_close = np.log(close)
    horizons = {
        "ret_1d":  log_close - log_close.shift(1),
        "ret_5d":  log_close - log_close.shift(5),
        "ret_21d": log_close - log_close.shift(21),
        "ret_63d": log_close - log_close.shift(63),
        "ret_126d": log_close - log_close.shift(126),
        "ret_252d": log_close - log_close.shift(252),
    }
    # Cross-sectionally rank each
    return {k: cross_sectional_rank(v) for k, v in horizons.items()}


# ─────────────────────────────────────────────
#  Build all factor features
# ─────────────────────────────────────────────

def build_factor_features(
    close:      pd.DataFrame,
    high:       pd.DataFrame,
    low:        pd.DataFrame,
    volume:     pd.DataFrame,
    sector_map: dict[str, str],
    cfg:        dict,
) -> pd.DataFrame:
    """
    Compute all factor features and return a tall (date, ticker) DataFrame.
    """
    signal_cfg = cfg["signals"]

    # Momentum
    mom = momentum_signal(
        close,
        lookback_long=signal_cfg["momentum"]["lookback_long"],
        lookback_skip=signal_cfg["momentum"]["lookback_skip"],
    ).rename("momentum_signal")

    # Reversal
    rev = reversal_signal(
        close, volume,
        lookback=signal_cfg["reversal"]["lookback"],
        volume_threshold=signal_cfg["reversal"]["volume_threshold"],
    ).rename("reversal_signal")

    # Sector relative strength
    sec_rs = sector_relative_strength(close, sector_map).rename("sector_rs")

    # Multi-horizon returns
    mhr = multi_horizon_returns(close)

    # Size rank
    sz = size_rank(volume, close).rename("size_rank")

    # Stack all to long format
    frames = {
        "momentum_signal": mom.stack(),
        "reversal_signal": rev.stack(),
        "sector_rs":       sec_rs.stack(),
        "size_rank":       sz.stack(),
    }
    for k, v in mhr.items():
        frames[k] = v.stack()

    result = pd.DataFrame(frames)
    result.index.names = ["date", "ticker"]
    return result


def compute_forward_returns(
    close: pd.DataFrame,
    horizon: int = 21,
) -> pd.DataFrame:
    """
    TARGET variable for ML training.
    forward_return at T = return from T+1 to T+horizon+1.
    We shift BACKWARDS in time (future into the record).
    This column is ONLY used during the training window — never in
    production. The train/test split in ml_signal.py enforces this.
    """
    log_close = np.log(close)
    fwd = (log_close.shift(-horizon) - log_close.shift(-1))  # T+1 to T+horizon+1
    # Cross-sectionally rank
    return cross_sectional_rank(fwd)
