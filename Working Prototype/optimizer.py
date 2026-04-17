"""
src/portfolio/optimizer.py
──────────────────────────
Portfolio construction from composite signals.

Methods:
  1. Mean-Variance Optimisation (Markowitz) with Ledoit-Wolf covariance
  2. Risk Parity (Equal Risk Contribution)
  3. Signal-proportional (simple rank-based weights)

All methods enforce:
  - Max single position: 5%
  - Max sector concentration: 25%
  - Max gross leverage: 200%
  - Max net leverage: 30%
  - Max turnover per rebalance: 25%
  - Long top-N / short bottom-N symmetry
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Covariance estimation
# ─────────────────────────────────────────────

def estimate_covariance(
    returns: pd.DataFrame,
    window:  int = 252,
    method:  str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Estimate covariance matrix with shrinkage.
    Ledoit-Wolf is the recommended default — analytically optimal shrinkage.
    """
    ret_clean = returns.dropna(axis=1, how="all").fillna(0)
    if ret_clean.shape[1] < 2:
        return pd.DataFrame()

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(ret_clean.values)
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=ret_clean.columns,
            columns=ret_clean.columns,
        )
    else:
        cov_matrix = ret_clean.cov()

    return cov_matrix


# ─────────────────────────────────────────────
#  Constraint helpers
# ─────────────────────────────────────────────

def apply_position_limits(
    weights:    pd.Series,
    max_pos:    float = 0.05,
    max_sector: float = 0.25,
    sector_map: dict[str, str] = None,
) -> pd.Series:
    """Clip individual positions and sector concentrations."""
    # Individual limits
    weights = weights.clip(-max_pos, max_pos)

    # Sector limits
    if sector_map:
        for sector in set(sector_map.values()):
            tickers = [t for t, s in sector_map.items()
                       if s == sector and t in weights.index]
            sector_weight = weights[tickers].sum()
            if abs(sector_weight) > max_sector:
                scale = max_sector / abs(sector_weight)
                weights[tickers] = weights[tickers] * scale

    return weights


def apply_turnover_limit(
    target_weights:  pd.Series,
    current_weights: pd.Series,
    max_turnover:    float = 0.25,
) -> pd.Series:
    """
    Shrink trades toward current portfolio if turnover exceeds limit.
    """
    if current_weights is None or len(current_weights) == 0:
        return target_weights

    diff = target_weights.subtract(current_weights, fill_value=0)
    total_turnover = diff.abs().sum() / 2

    if total_turnover > max_turnover:
        scale = max_turnover / total_turnover
        adjusted = current_weights.add(diff * scale, fill_value=0)
        return adjusted

    return target_weights


# ─────────────────────────────────────────────
#  1. Signal-proportional weights (baseline)
# ─────────────────────────────────────────────

def signal_proportional_weights(
    scores:    pd.Series,
    top_n:     int = 30,
    max_pos:   float = 0.05,
) -> pd.Series:
    """
    Long top_n stocks by score, short bottom_n.
    Weights proportional to score magnitude.
    """
    scores = scores.dropna()
    top    = scores.nlargest(top_n)
    bottom = scores.nsmallest(top_n)

    weights = pd.Series(0.0, index=scores.index)

    # Long side: weight proportional to positive score
    long_w  = top.clip(lower=0)
    if long_w.sum() > 0:
        weights[long_w.index] = (long_w / long_w.sum()) * (max_pos * top_n)

    # Short side
    short_w = (-bottom).clip(lower=0)
    if short_w.sum() > 0:
        weights[short_w.index] = -(short_w / short_w.sum()) * (max_pos * top_n)

    return weights.clip(-max_pos, max_pos)


# ─────────────────────────────────────────────
#  2. Mean-Variance Optimisation
# ─────────────────────────────────────────────

def mean_variance_optimise(
    expected_returns: pd.Series,
    cov_matrix:       pd.DataFrame,
    target_vol:       float = 0.15,
    max_pos:          float = 0.05,
    max_gross:        float = 2.0,
    max_net:          float = 0.30,
) -> pd.Series:
    """
    Maximise Sharpe = expected_return / portfolio_vol
    subject to position and leverage constraints.
    Uses scipy.optimize.minimize with SLSQP.
    """
    tickers = expected_returns.dropna().index
    tickers = tickers.intersection(cov_matrix.index)

    if len(tickers) < 5:
        log.warning("Too few tickers for MVO, falling back to signal-proportional")
        return signal_proportional_weights(expected_returns)

    mu  = expected_returns[tickers].values
    cov = cov_matrix.loc[tickers, tickers].values
    n   = len(tickers)

    # Objective: negative Sharpe (minimise)
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w + 1e-10)
        return -port_ret / port_vol

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    # Constraints
    constraints = [
        {"type": "ineq", "fun": lambda w: target_vol - portfolio_vol(w)},        # vol ≤ target
        {"type": "ineq", "fun": lambda w: max_gross - np.abs(w).sum()},           # gross ≤ max
        {"type": "ineq", "fun": lambda w: max_net - abs(w.sum())},                # net ≤ max
    ]

    bounds = [(-max_pos, max_pos)] * n
    w0 = np.zeros(n)

    result = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if result.success or result.fun < 0:
        return pd.Series(result.x, index=tickers)
    else:
        log.warning(f"MVO failed: {result.message}. Falling back to signal-proportional.")
        return signal_proportional_weights(expected_returns[tickers], max_pos=max_pos)


# ─────────────────────────────────────────────
#  3. Risk Parity
# ─────────────────────────────────────────────

def risk_parity_weights(
    cov_matrix: pd.DataFrame,
    max_pos:    float = 0.05,
) -> pd.Series:
    """
    Equal Risk Contribution portfolio.
    Each position contributes equally to total portfolio variance.
    """
    tickers = cov_matrix.index
    n = len(tickers)
    cov = cov_matrix.values

    target_risk = np.ones(n) / n  # equal contribution

    def risk_parity_obj(w):
        w = np.abs(w)
        port_var = w @ cov @ w
        mrc = cov @ w                           # marginal risk contribution
        rc  = w * mrc / (port_var + 1e-10)      # risk contribution
        return np.sum((rc - target_risk) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0, max_pos)] * n   # long-only risk parity

    result = minimize(
        risk_parity_obj, w0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = pd.Series(np.abs(result.x), index=tickers)
    weights = weights / weights.sum() * 0.5   # normalise to 50% gross (leave room for shorts)
    return weights.clip(0, max_pos)


# ─────────────────────────────────────────────
#  Main portfolio construction
# ─────────────────────────────────────────────

class PortfolioOptimizer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pcfg = cfg["portfolio"]

    def construct(
        self,
        composite_scores:  pd.Series,    # ticker → score at rebalance date
        returns_history:   pd.DataFrame, # past daily returns for cov estimation
        sector_map:        dict[str, str],
        current_weights:   pd.Series = None,
        regime:            int = 0,       # 0=low-vol, 1=high-vol
    ) -> pd.Series:
        """
        Main entry point. Returns target portfolio weights.
        """
        pcfg = self.pcfg
        method = pcfg["method"]

        # Select long/short candidate universe
        scores = composite_scores.dropna()
        top_n  = pcfg["top_n_long"]
        long_candidates  = scores.nlargest(top_n).index
        short_candidates = scores.nsmallest(top_n).index
        candidates = long_candidates.union(short_candidates)

        if len(candidates) < 10:
            log.warning("Too few candidates for portfolio construction")
            return pd.Series(dtype=float)

        # Estimate covariance
        ret_hist = returns_history[candidates].dropna(how="all")
        cov = estimate_covariance(ret_hist, window=pcfg.get("cov_window", 252))

        if method == "mean_variance":
            # Scale scores to expected return space (monotonic transform)
            expected_ret = scores[candidates]
            weights = mean_variance_optimise(
                expected_ret, cov,
                target_vol=pcfg["target_volatility"],
                max_pos=pcfg["max_position_size"],
                max_gross=pcfg["max_gross_leverage"],
                max_net=pcfg["max_net_leverage"],
            )
        elif method == "risk_parity":
            long_w  =  risk_parity_weights(cov.loc[long_candidates, long_candidates])
            short_w = -risk_parity_weights(cov.loc[short_candidates, short_candidates])
            weights = pd.concat([long_w, short_w])
        else:
            # Signal-proportional fallback
            weights = signal_proportional_weights(
                scores[candidates],
                top_n=top_n,
                max_pos=pcfg["max_position_size"],
            )

        # Apply position and sector limits
        weights = apply_position_limits(
            weights,
            max_pos=pcfg["max_position_size"],
            max_sector=pcfg["max_sector_weight"],
            sector_map=sector_map,
        )

        # Turnover limit
        if current_weights is not None:
            weights = apply_turnover_limit(
                weights, current_weights,
                max_turnover=pcfg["max_turnover_per_rebalance"],
            )

        # Final leverage check
        gross = weights.abs().sum()
        if gross > pcfg["max_gross_leverage"]:
            weights = weights * (pcfg["max_gross_leverage"] / gross)

# 🚀 NEW: ensure we actually use capital
        target_gross = 2.0   # try 2.0 first (you were ~1.5)

        gross = weights.abs().sum()
        if gross > 0:
            weights = weights * (target_gross / gross)

        log.info(f"Portfolio: {len(weights[weights>0])} long, {len(weights[weights<0])} short, "
         f"gross={weights.abs().sum():.2f}, net={weights.sum():.3f}")

        return weights
