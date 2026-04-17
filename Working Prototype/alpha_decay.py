"""
src/signals/alpha_decay.py
──────────────────────────
Alpha decay analysis for each signal:
  - Information Coefficient (IC): rank correlation of signal vs forward return
  - ICIR: IC / std(IC) — signal-to-noise ratio
  - IC half-life: how quickly predictive power decays
  - Cumulative IC plot data

This is a KEY differentiator in the competition — most teams skip this.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

log = logging.getLogger(__name__)


def compute_ic_series(
    signal: pd.DataFrame,       # wide: dates × tickers
    forward_returns: pd.DataFrame,  # wide: dates × tickers
    method: str = "spearman",
) -> pd.Series:
    """
    Compute daily Information Coefficient (IC).
    IC at date T = rank correlation between signal(T) and forward_return(T).

    IMPORTANT: forward_return at T must use data from T+1 onwards.
    This is only valid inside the training window. Never use in production.
    """
    ics = {}
    common_dates = signal.index.intersection(forward_returns.index)

    for date in common_dates:
        s = signal.loc[date].dropna()
        f = forward_returns.loc[date].dropna()
        common = s.index.intersection(f.index)
        if len(common) < 10:
            continue
        if method == "spearman":
            ic, _ = stats.spearmanr(s[common], f[common])
        else:
            ic = s[common].corr(f[common])
        ics[date] = ic

    return pd.Series(ics, name="ic")


def compute_ic_stats(ic_series: pd.Series, signal_name: str = "") -> dict:
    """
    Compute IC summary statistics.
    IC > 0.05 is considered economically significant.
    ICIR > 0.5 suggests a consistently predictive signal.
    """
    ic = ic_series.dropna()
    mean_ic  = ic.mean()
    std_ic   = ic.std()
    icir     = mean_ic / (std_ic + 1e-10)
    pct_positive = (ic > 0).mean()

    # t-test: is mean IC significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(ic, 0)

    stats_dict = {
        "signal":       signal_name,
        "mean_ic":      round(mean_ic, 4),
        "std_ic":       round(std_ic, 4),
        "icir":         round(icir, 4),
        "pct_positive": round(pct_positive, 3),
        "t_stat":       round(t_stat, 3),
        "p_value":      round(p_value, 4),
        "n_obs":        len(ic),
        "is_significant": p_value < 0.05 and mean_ic > 0,
    }
    log.info(f"[{signal_name}] Mean IC={mean_ic:.4f}, ICIR={icir:.4f}, p={p_value:.4f}")
    return stats_dict


def compute_ic_half_life(ic_series: pd.Series) -> float:
    """
    Fit AR(1) to cumulative IC to estimate alpha decay half-life in days.
    half_life = -log(2) / log(|phi|) where phi is the AR(1) coefficient.
    """
    ic = ic_series.dropna()
    if len(ic) < 30:
        return np.nan

    # AR(1): IC_t = phi * IC_{t-1} + epsilon
    y = ic.values[1:]
    x = ic.values[:-1]
    phi = np.cov(x, y)[0, 1] / (np.var(x) + 1e-10)

    if abs(phi) >= 1 or phi <= 0:
        return np.nan

    half_life = -np.log(2) / np.log(abs(phi))
    return round(half_life, 1)


def compute_rolling_ic(
    signal:          pd.DataFrame,
    forward_returns: pd.DataFrame,
    window:          int = 63,
    method:          str = "spearman",
) -> pd.Series:
    """Rolling IC to visualise alpha stability over time."""
    ic_daily = compute_ic_series(signal, forward_returns, method)
    return ic_daily.rolling(window, min_periods=window // 2).mean()


def compute_ic_decay_by_horizon(
    signal: pd.DataFrame,
    close:  pd.DataFrame,
    horizons: list[int] = [1, 5, 10, 21, 42, 63],
) -> pd.DataFrame:
    """
    Compute IC of a signal against forward returns at multiple horizons.
    Shows how quickly the signal loses predictive power (alpha decay).
    Returns DataFrame: index=horizon, cols=[mean_ic, icir]
    """
    log_close = np.log(close)
    results = []

    for h in horizons:
        fwd = (log_close.shift(-h) - log_close.shift(-1))
        # Cross-sectional rank
        fwd_ranked = fwd.rank(axis=1, pct=True) * 2 - 1

        ic_series = compute_ic_series(signal, fwd_ranked)
        stats_d = compute_ic_stats(ic_series, f"horizon_{h}d")
        stats_d["horizon_days"] = h
        results.append(stats_d)

    return pd.DataFrame(results).set_index("horizon_days")


def run_alpha_decay_report(
    signals: dict[str, pd.DataFrame],   # signal_name → wide signal df
    close:   pd.DataFrame,
    forward_returns_21d: pd.DataFrame,  # pre-computed 21d forward returns
    output_dir: str = "data/reports",
) -> pd.DataFrame:
    """
    Run full alpha decay report for all signals.
    Returns a summary DataFrame suitable for the competition report.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    decay_rows   = []

    for name, sig in signals.items():
        # IC stats at 21-day horizon
        ic_series = compute_ic_series(sig, forward_returns_21d)
        stats_d   = compute_ic_stats(ic_series, name)
        half_life = compute_ic_half_life(ic_series)
        stats_d["half_life_days"] = half_life
        summary_rows.append(stats_d)

        # IC by horizon
        decay_df = compute_ic_decay_by_horizon(sig, close, [1, 5, 10, 21, 42, 63])
        decay_df["signal"] = name
        decay_rows.append(decay_df.reset_index())

        # Save rolling IC
        rolling_ic = compute_rolling_ic(sig, forward_returns_21d)
        rolling_ic.to_csv(os.path.join(output_dir, f"rolling_ic_{name}.csv"))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "alpha_decay_summary.csv"), index=False)

    decay_all = pd.concat(decay_rows, ignore_index=True)
    decay_all.to_csv(os.path.join(output_dir, "ic_by_horizon.csv"), index=False)

    log.info(f"Alpha decay report saved to {output_dir}")
    return summary


def compute_ic_weighted_ensemble(
    signals:         dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    lookback_days:   int = 126,
    rebalance_dates: list[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    IC-weighted ensemble: weight each signal by its recent IC.
    At each rebalance date, estimate IC over the past `lookback_days`.
    Returns a wide DataFrame of composite scores.
    """
    if rebalance_dates is None:
        all_dates = list(signals.values())[0].index
        rebalance_dates = all_dates

    composite_scores = []

    for date in rebalance_dates:
        hist_start = date - pd.Timedelta(days=lookback_days)

        weights = {}
        for name, sig in signals.items():
            # IC over lookback window
            sig_window = sig.loc[hist_start:date]
            fwd_window = forward_returns.loc[hist_start:date]
            ic_series  = compute_ic_series(sig_window, fwd_window)
            mean_ic    = ic_series.mean() if len(ic_series) > 10 else 0
            weights[name] = max(mean_ic, 0)  # floor at 0 (don't use negative-IC signals)

        total_w = sum(weights.values())
        if total_w == 0:
            # Equal weight fallback
            weights = {k: 1/len(signals) for k in signals}
            total_w = 1.0

        # Composite score at this date
        scores = pd.Series(0.0, index=list(signals.values())[0].columns)
        for name, sig in signals.items():
            if date in sig.index:
                scores += (weights[name] / total_w) * sig.loc[date].fillna(0)

        composite_scores.append(scores.rename(date))

    return pd.DataFrame(composite_scores)
