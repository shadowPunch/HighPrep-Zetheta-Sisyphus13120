# NIFTY 500 Quantitative trading system

## DATASET
1. get_data.ipynb - the main notebook for creating the datasets.
2. classified dataset.zip -Organized subsets for easier experimentation
3. EDA.ipynb - provides a basic EDA for the data.

#### By Size
```
size/large/
size/mid/
size/small/
```
#### By Sector
```
sector/<sector_name>/
```
1. dailysnapshot parquet - daily movement of the nifty500 from 2012 to 2025 dec, some constituents werent present earlier, so thats why each day might not have exactly 500 items. labels - date, ohlcv, might not be very clean instead use panel_clean_new.parquet. this one is not ordered, but is cleaner to work with.
2. dataset500_Clean - master dataset obtained by nifty 500 notebook
3. nifty50 unvierse classified excel - list of all companies, along with sector info, ignore market cap, dont know which date thats for
Each folder contains relevant ticker CSVs.

## Quant Model
Implementing a production-grade quantitative trading system designed to generate market-neutral alpha using a combination of:

- Momentum (12–1 month)
- Short-term Reversal (volume-adjusted)
- Regime-aware signal weighting (HMM-based)
- Risk-constrained portfolio optimization

The strategy focuses on high risk-adjusted returns (Sharpe > 2.5), low drawdowns, and robustness under realistic transaction costs.


---

##  Strategy Logic

### 1. Momentum Factor
- 12-month return excluding most recent month (12–1)
- Captures trend persistence

### 2. Reversal Factor
- Short-term mean reversion (1-month)
- Weighted by abnormal volume (stronger reversals after heavy selling)

### 3. Regime Overlay
- 2-state Hidden Markov Model on market volatility
- Adjusts signal weights:
  - Trending regime → momentum dominant  
  - Stress regime → reversal dominant  

### 4. Portfolio Construction
- Long top-ranked stocks, short bottom-ranked stocks  
- Risk-constrained optimization  
- Volatility-aware sizing  

---

##  Performance Summary

| Metric | Value |
|------|------|
| Sharpe Ratio | ~2.5+ |
| Max Drawdown | ~6–8% |
| CAGR | ~8–9% |
| Turnover | ~0.25 |
| Transaction Cost | 20 bps |



---

##  Project Structure & File Descriptions

### 🔹 `loader.py`
- Loads raw dataset from compressed files  
- Constructs panel data (multi-index: date, ticker)  
- Applies universe filtering (size, availability)

---

### 🔹 `quality.py`
- Data cleaning and validation  
- Handles missing values, outliers, stale prices  
- Ensures point-in-time correctness (no look-ahead bias)

---

### 🔹 `technical.py`
- Computes technical indicators:
  - RSI, MACD, Bollinger Bands  
  - ADX, VWAP deviation, OBV  
  - Volume ratios and volatility metrics  
- Outputs feature matrix for modelling

---

### 🔹 `factors.py`
- Implements core alpha signals:
  - Momentum (12–1)
  - Reversal (volume-conditioned)
- Produces cross-sectional rankings per date

---

### 🔹 `alpha_decay.py`
- Evaluates signal quality using:
  - Information Coefficient (IC)
  - ICIR (stability)
  - Half-life and decay curves  
- Ensures signals are statistically valid

---

### 🔹 `regime.py`
- Implements 2-state Hidden Markov Model (HMM)  
- Detects market regimes based on volatility  
- Outputs regime labels and dynamic signal weights

---

### 🔹 `optimizer.py`
- Portfolio construction engine  
- Supports:
  - Mean-variance optimization  
  - Risk parity / signal-based allocation  
- Applies constraints:
  - Leverage limits  
  - Sector exposure  
  - Turnover control  

---

### 🔹 `Backtester.py`
- Core simulation engine  
- Executes strategy over time  
- Handles:
  - Rebalancing  
  - Transaction costs  
  - Position updates  
- Outputs PnL and performance metrics

---

### 🔹 `config.yaml`
- Central configuration file  
- Controls:
  - Signal parameters  
  - Portfolio constraints  
  - Transaction costs  
  - Backtest settings  

---

---

##  How to Run

```bash
streamlit app dashboard.py
```

