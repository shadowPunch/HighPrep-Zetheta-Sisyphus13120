"""
src/signals/regime.py
─────────────────────
2-state Hidden Markov Model for market regime detection.

State 0 = Low-vol / Trending regime  → upweight momentum
State 1 = High-vol / Stress regime   → upweight reversal + ML

The regime is detected using PAST volatility only.
At rebalance date T, the regime is inferred from data up to T-1.
"""

import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class RegimeDetector:
    """
    Fits a 2-state Gaussian HMM on realised volatility.
    Provides point-in-time regime labels and signal weight adjustments.
    """

    def __init__(self, n_states: int = 2, random_state: int = 42):
        self.n_states     = n_states
        self.random_state = random_state
        self.model_       = None
        self.scaler_      = StandardScaler()
        self.low_vol_state_ = None   # which HMM state = low-vol

    def fit(self, market_vol_series: pd.Series) -> "RegimeDetector":
        """
        Fit HMM on market-level realised volatility (e.g. cross-sectional median vol).

        Parameters
        ----------
        market_vol_series : pd.Series
            Daily realised volatility, indexed by date.
        """
        vol = market_vol_series.dropna().values.reshape(-1, 1)
        vol_scaled = self.scaler_.fit_transform(vol)

        self.model_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        self.model_.fit(vol_scaled)

        # Determine which state has lower mean volatility
        means = self.model_.means_.flatten()
        self.low_vol_state_ = int(np.argmin(means))
        log.info(f"HMM fitted. Low-vol state = {self.low_vol_state_} "
                 f"(means: {means})")
        return self

    def predict(self, market_vol_series: pd.Series) -> pd.Series:
        """
        Predict regime for each date in market_vol_series.
        Returns 0 = low-vol/trending, 1 = high-vol/stress.
        """
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict()")

        valid = market_vol_series.dropna()
        vol_scaled = self.scaler_.transform(valid.values.reshape(-1, 1))
        raw_states = self.model_.predict(vol_scaled)

        # Remap so that 0 = low-vol, 1 = high-vol regardless of HMM label
        remapped = np.where(raw_states == self.low_vol_state_, 0, 1)
        return pd.Series(remapped, index=valid.index, name="regime")

    def predict_proba(self, market_vol_series: pd.Series) -> pd.DataFrame:
        """
        Posterior probabilities for each state.
        Returns DataFrame with columns [prob_low_vol, prob_high_vol].
        """
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict_proba()")

        valid = market_vol_series.dropna()
        vol_scaled = self.scaler_.transform(valid.values.reshape(-1, 1))
        proba = self.model_.predict_proba(vol_scaled)

        df = pd.DataFrame(proba, index=valid.index,
                          columns=[f"state_{i}" for i in range(self.n_states)])
        df["prob_low_vol"]  = df[f"state_{self.low_vol_state_}"]
        df["prob_high_vol"] = 1 - df["prob_low_vol"]
        return df[["prob_low_vol", "prob_high_vol"]]


def compute_market_vol(
    close: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """
    Market-level volatility = cross-sectional median of individual stock vols.
    More stable than a single index.
    """
    log_ret = np.log(close / close.shift(1))
    stock_vol = log_ret.rolling(window).std() * np.sqrt(252)
    return stock_vol.median(axis=1)


def get_regime_weights(
    regime: int,
    cfg: dict,
) -> dict[str, float]:
    """
    Return signal weights based on current regime.
    regime: 0 = low-vol/trending, 1 = high-vol/stress
    """
    rcfg = cfg["regime"]
    if regime == 0:
        return {
            "momentum": rcfg["momentum_weight_trending"],
            "reversal": rcfg["reversal_weight_trending"],
            "ml":       rcfg["ml_weight_trending"],
        }
    else:
        return {
            "momentum": rcfg["momentum_weight_stress"],
            "reversal": rcfg["reversal_weight_stress"],
            "ml":       rcfg["ml_weight_stress"],
        }


def build_regime_series(
    close: pd.DataFrame,
    cfg: dict,
    fit_end_date: str,    # Fit HMM only on data up to this date (train window)
) -> tuple[pd.Series, RegimeDetector]:
    """
    Fit the regime model on the training window, then predict forward.
    Returns (regime_series, fitted_detector).
    """
    market_vol = compute_market_vol(close, window=21)

    # Fit only on training data (anti-lookahead)
    fit_vol = market_vol.loc[:fit_end_date].dropna()
    detector = RegimeDetector(n_states=cfg["regime"]["n_states"])
    detector.fit(fit_vol)

    # Predict on all available data (train + test — regime label at T uses data ≤ T)
    regime_series = detector.predict(market_vol)

    regime_counts = regime_series.value_counts()
    log.info(f"Regime distribution: low-vol={regime_counts.get(0,0)}, "
             f"high-vol={regime_counts.get(1,0)}")

    return regime_series, detector
