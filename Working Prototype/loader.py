"""
src/data/loader.py
──────────────────
Loads the classified_dataset zip into a unified panel DataFrame.

The zip has two organisational dimensions:
  classified_dataset/size/{large,mid,small}/<TICKER>.csv
  classified_dataset/sector/<SectorName>/<TICKER>.csv

We load from the /size/ tree (canonical OHLCV) and build a lookup
table mapping ticker → (size_category, sector) from the /sector/ tree.

Output: panel DataFrame indexed by (date, ticker) with columns:
  open, high, low, close, volume, size_cat, sector, adj_close*

*adj_close = close (data is already adjusted per the source)
"""

import zipfile
import io
import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_ticker_metadata(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Build a DataFrame: ticker -> size_cat, sector."""
    records = []
    for name in zf.namelist():
        parts = Path(name).parts
        if len(parts) < 4:
            continue
        # classified_dataset/sector/<SectorName>/<TICKER>.csv
        if parts[1] == "sector" and name.endswith(".csv"):
            sector = parts[2]
            ticker = Path(parts[3]).stem
            records.append({"ticker": ticker, "sector": sector})
        # classified_dataset/size/<size>/<TICKER>.csv
        elif parts[1] == "size" and name.endswith(".csv"):
            size_cat = parts[2]
            ticker = Path(parts[3]).stem
            records.append({"ticker": ticker, "size_cat": size_cat})

    # Merge sector and size records
    df_sector = pd.DataFrame([r for r in records if "sector" in r])
    df_size   = pd.DataFrame([r for r in records if "size_cat" in r])
    meta = df_size.merge(df_sector, on="ticker", how="left")
    meta = meta.drop_duplicates("ticker").reset_index(drop=True)
    return meta


def load_single_ticker(zf: zipfile.ZipFile, zip_path: str, ticker: str) -> Optional[pd.DataFrame]:
    """Read one CSV from the zip into a clean DataFrame."""
    try:
        with zf.open(zip_path) as f:
            df = pd.read_csv(f, parse_dates=["date"])
        df = df.rename(columns=str.lower)
        df["ticker"] = ticker
        df = df.sort_values("date").reset_index(drop=True)
        # Basic sanity
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            log.warning(f"Missing columns for {ticker}: {df.columns.tolist()}")
            return None
        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:
        log.warning(f"Failed to load {ticker}: {e}")
        return None


def load_panel(
    zip_path: str,
    size_filter: list[str] = ["large", "mid"],
    min_history_days: int = 756,
    processed_dir: str = "data/processed",
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Main entry point. Returns a MultiIndex DataFrame (date, ticker).
    Caches to Parquet for fast subsequent loads.
    """
    cache_path = os.path.join(processed_dir, "panel.parquet")
    os.makedirs(processed_dir, exist_ok=True)

    if not force_reload and os.path.exists(cache_path):
        log.info(f"Loading cached panel from {cache_path}")
        return pd.read_parquet(cache_path)

    log.info(f"Loading panel from {zip_path} (size_filter={size_filter})")
    frames = []

    with zipfile.ZipFile(zip_path) as zf:
        meta = build_ticker_metadata(zf)
        log.info(f"Found {len(meta)} unique tickers in metadata")

        # Filter by size
        if size_filter:
            valid_tickers = set(meta.loc[meta["size_cat"].isin(size_filter), "ticker"])
        else:
            valid_tickers = set(meta["ticker"])

        log.info(f"Loading {len(valid_tickers)} tickers after size filter")

        # Build a path lookup for size/large or size/mid
        path_lookup = {}
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) >= 4 and parts[1] == "size" and name.endswith(".csv"):
                ticker = Path(parts[3]).stem
                size_cat = parts[2]
                if ticker in valid_tickers and size_cat in size_filter:
                    path_lookup[ticker] = name

        for ticker, zip_path_inner in path_lookup.items():
            df = load_single_ticker(zf, zip_path_inner, ticker)
            if df is not None and len(df) >= min_history_days:
                frames.append(df)

    if not frames:
        raise ValueError("No data loaded. Check zip path and size_filter.")

    panel = pd.concat(frames, ignore_index=True)

    # Attach metadata
    panel = panel.merge(meta[["ticker", "size_cat", "sector"]], on="ticker", how="left")

    # Set MultiIndex
    panel = panel.set_index(["date", "ticker"]).sort_index()

    # Persist
    panel.to_parquet(cache_path)
    log.info(f"Panel shape: {panel.shape} | Cached to {cache_path}")
    return panel


def get_price_matrix(panel: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot panel to wide format: rows=dates, cols=tickers."""
    return panel[field].unstack(level="ticker")


def get_universe_at_date(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    min_history_days: int = 252,
) -> list[str]:
    """
    Return list of tickers with sufficient history BEFORE the given date.
    This is used at each rebalance to construct the investable universe.
    Point-in-time correct — no look-ahead.
    """
    prior_data = panel.loc[:date]
    counts = prior_data.groupby(level="ticker").size()
    eligible = counts[counts >= min_history_days].index.tolist()
    return eligible


if __name__ == "__main__":
    cfg = load_config()
    panel = load_panel(
        zip_path=cfg["data"]["raw_zip"],
        size_filter=cfg["universe"]["size_filter"],
        min_history_days=cfg["universe"]["min_history_days"],
        processed_dir=cfg["data"]["processed_dir"],
    )
    print(panel.head())
    print(f"\nPanel shape: {panel.shape}")
    print(f"Date range: {panel.index.get_level_values('date').min()} → "
          f"{panel.index.get_level_values('date').max()}")
    print(f"Tickers: {panel.index.get_level_values('ticker').nunique()}")
    print(f"Sectors: {panel['sector'].unique()}")
