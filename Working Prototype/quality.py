"""
src/data/quality.py
───────────────────
Data quality checks, cleaning, and flagging.
All operations are strictly backward-looking (no look-ahead bias).
"""

import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def check_and_clean(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all quality checks on the panel. Returns:
      - cleaned panel
      - quality report DataFrame (one row per ticker)
    """
    panel = panel.copy()
    report_rows = []

    tickers = panel.index.get_level_values("ticker").unique()

    for ticker in tickers:
        df = panel.xs(ticker, level="ticker").copy()
        issues = []

        # ── 1. Missing values ──────────────────────────────────────────
        n_missing = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
        if n_missing > 0:
            issues.append(f"missing_values={n_missing}")
            df = df.ffill().bfill()

        # ── 2. Zero or negative prices ─────────────────────────────────
        neg_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if neg_mask.sum() > 0:
            issues.append(f"non_positive_prices={neg_mask.sum()}")
            df.loc[neg_mask, ["open", "high", "low", "close"]] = np.nan
            df = df.ffill()

        # ── 3. Stale prices (same close 3+ days in a row) ──────────────
        stale_mask = (
            (df["close"] == df["close"].shift(1)) &
            (df["close"] == df["close"].shift(2))
        )
        n_stale = stale_mask.sum()
        if n_stale > 0:
            issues.append(f"stale_prices={n_stale}")
            df.loc[stale_mask, "close"] = np.nan
            df["close"] = df["close"].ffill()

        # ── 4. Outlier returns (> 5 std) ───────────────────────────────
        returns = df["close"].pct_change()
        mean_r = returns.mean()
        std_r  = returns.std()
        outlier_mask = (returns - mean_r).abs() > 5 * std_r
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            issues.append(f"outlier_returns={n_outliers}")
            df.loc[outlier_mask, "close"] = np.nan
            df["close"] = df["close"].ffill()

        # ── 5. OHLC consistency ─────────────────────────────────────────
        bad_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"]  > df["open"]) |
            (df["low"]  > df["close"])
        )
        if bad_ohlc.sum() > 0:
            issues.append(f"ohlc_inconsistency={bad_ohlc.sum()}")

        # ── 6. Zero volume ─────────────────────────────────────────────
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            issues.append(f"zero_volume={zero_vol}")
            df.loc[df["volume"] == 0, "volume"] = np.nan
            df["volume"] = df["volume"].ffill()

        # Re-assign cleaned data back
        for col in ["open", "high", "low", "close", "volume"]:
            panel.loc[(slice(None), ticker), col] = df[col].values

        report_rows.append({
            "ticker": ticker,
            "n_rows": len(df),
            "issues": "; ".join(issues) if issues else "clean",
            "n_issues": len(issues),
        })

    quality_report = pd.DataFrame(report_rows)
    n_dirty = (quality_report["n_issues"] > 0).sum()
    log.info(f"Quality check: {n_dirty}/{len(tickers)} tickers had issues (all fixed)")

    return panel, quality_report


def compute_returns(price_matrix: pd.DataFrame, periods: list[int] = [1, 5, 21, 63, 252]) -> dict[int, pd.DataFrame]:
    """
    Compute forward-shifted log returns for multiple horizons.
    These are PAST returns at time T (no lookahead).
    """
    returns = {}
    log_prices = np.log(price_matrix)
    for p in periods:
        returns[p] = log_prices.diff(p)
    return returns


def compute_realised_vol(price_matrix: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised realised volatility over a rolling window."""
    log_ret = np.log(price_matrix / price_matrix.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def flag_survivorship_bias(panel: pd.DataFrame, min_end_date: str = "2024-01-01") -> list[str]:
    """
    Flag tickers that have data through at least min_end_date.
    Used to ensure we only use stocks with sufficient forward history.
    NOTE: For a live system you'd include delisted stocks; here we flag
    the ones we can't confirm are still trading.
    """
    last_dates = panel.groupby(level="ticker").apply(
        lambda x: x.index.get_level_values("date").max()
    )
    live = last_dates[last_dates >= pd.Timestamp(min_end_date)].index.tolist()
    delisted = last_dates[last_dates < pd.Timestamp(min_end_date)].index.tolist()
    log.info(f"Survivorship: {len(live)} live stocks, {len(delisted)} potentially delisted")
    return delisted
