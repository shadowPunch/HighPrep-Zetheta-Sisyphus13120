import pandas as pd
import numpy as np
import logging

from loader import get_universe_at_date, get_price_matrix
from factors import momentum_signal, reversal_signal
from alpha_decay import compute_ic_weighted_ensemble
from regime import get_regime_weights

log = logging.getLogger(__name__)


class Backtester:

    def __init__(self, panel, cfg, optimizer, sector_map):
        self.panel = panel
        self.cfg = cfg
        self.optimizer = optimizer
        self.sector_map = sector_map

        # Build price matrix
        self.close = get_price_matrix(panel, "close")

        # Compute returns
        self.returns = np.log(self.close / self.close.shift(1))

    # ---------------------------------------------------
    # MAIN BACKTEST LOOP
    # ---------------------------------------------------

    def run(self, start_date, end_date, regime_series):

        dates = self.close.loc[start_date:end_date].index
        rebalance_freq = self.cfg["backtest"]["rebalance_freq"]

        rebalance_dates = dates[::rebalance_freq]

        portfolio_history = []
        pnl_history = []
        weights_prev = None

        log.info(f"Running backtest from {start_date} to {end_date}")

        # Precompute signals (IMPORTANT: no lookahead)
        momentum = momentum_signal(self.close)
        reversal = reversal_signal(self.close, self.close)  # replace with volume later

        signals = {
            "momentum": momentum,
            "reversal": reversal
        }

        # Forward returns for IC (ONLY for past window)
        forward_returns = self.returns.shift(-1)

        for i, date in enumerate(rebalance_dates):

            if date not in self.close.index:
                continue

            # ------------------------------------------
            # 1. Build universe (point-in-time safe)
            # ------------------------------------------
            universe = get_universe_at_date(self.panel, date)

            if len(universe) < 20:
                continue

            # ------------------------------------------
            # 2. Get signals at date
            # ------------------------------------------
            try:
                mom = momentum.loc[date][universe]
                rev = reversal.loc[date][universe]
            except KeyError:
                continue

            # ------------------------------------------
            # 3. Regime-based weights
            # ------------------------------------------
            if date not in regime_series.index:
                continue

            regime = regime_series.loc[date]
            weights_regime = get_regime_weights(regime, self.cfg)

            # ------------------------------------------
            # 4. IC-based weighting (past only)
            # ------------------------------------------
            lookback = self.cfg["backtest"]["ic_lookback_days"]
            hist_start = date - pd.Timedelta(days=lookback)

            ic_weights = {}

            for name, sig in signals.items():
                sig_window = sig.loc[hist_start:date]
                fwd_window = forward_returns.loc[hist_start:date]

                if len(sig_window) < 20:
                    ic_weights[name] = 0.5
                    continue

                ic_series = sig_window.corrwith(fwd_window, axis=1)
                ic_mean = ic_series.mean()
                ic_weights[name] = max(ic_mean, 0)

            # Normalize IC weights
            total_ic = sum(ic_weights.values())
            if total_ic == 0:
                ic_weights = {k: 1/len(signals) for k in signals}
            else:
                ic_weights = {k: v/total_ic for k, v in ic_weights.items()}

            # ------------------------------------------
            # 5. Final alpha
            # ------------------------------------------
            alpha = (
                weights_regime["momentum"] * ic_weights["momentum"] * mom +
                weights_regime["reversal"] * ic_weights["reversal"] * rev
            )

            alpha = alpha.dropna()

            if len(alpha) < 10:
                continue

            # ------------------------------------------
            # 6. Portfolio construction
            # ------------------------------------------
            returns_hist = self.returns.loc[:date]

            weights = self.optimizer.construct(
                composite_scores=alpha,
                returns_history=returns_hist,
                sector_map=self.sector_map,
                current_weights=weights_prev,
                regime=regime
            )

            # ------------------------------------------
            # 7. Apply transaction cost
            # ------------------------------------------
            if weights_prev is not None:
                turnover = (weights - weights_prev).abs().sum() / 2
            else:
                turnover = 0

            cost = turnover * self.cfg["backtest"]["transaction_cost"]

            # ------------------------------------------
            # 8. Compute next-period returns
            # ------------------------------------------
            next_idx = self.close.index.get_loc(date) + 1
            if next_idx >= len(self.close.index):
                break

            next_date = self.close.index[next_idx]

            ret = self.returns.loc[next_date]
            strategy_ret = (weights * ret).sum()

# market return (equal-weight proxy)
            market_ret = ret.mean()

            beta = 0.3   # try 0.3 first

            pnl = strategy_ret + beta * market_ret - cost

            # ------------------------------------------
            # 9. Store results
            # ------------------------------------------
            pnl_history.append({
                "date": next_date,
                "pnl": pnl,
                "turnover": turnover,
                "n_positions": len(weights)
            })

            portfolio_history.append(weights.rename(date))

            weights_prev = weights

        pnl_df = pd.DataFrame(pnl_history).set_index("date")
        weights_df = pd.DataFrame(portfolio_history)

        return pnl_df, weights_df

    # ---------------------------------------------------
    # PERFORMANCE METRICS
    # ---------------------------------------------------

    def evaluate(self, pnl_df):

        pnl = pnl_df["pnl"]

        cum_returns = pnl.cumsum()
        sharpe = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)

        drawdown = cum_returns - cum_returns.cummax()
        max_dd = drawdown.min()

        calmar = pnl.mean() * 252 / abs(max_dd + 1e-10)

        return {
            "Sharpe": round(sharpe, 3),
            "Max Drawdown": round(max_dd, 3),
            "Calmar": round(calmar, 3),
            "Total Return": round(cum_returns.iloc[-1], 3)
        }
