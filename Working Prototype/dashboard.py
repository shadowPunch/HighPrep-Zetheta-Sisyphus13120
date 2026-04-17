"""
╔══════════════════════════════════════════════════════════════════════════╗
║          QUANT STRATEGY MONITOR  –  Streamlit Dashboard                  ║
║          Run: streamlit run dashboard.py                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Install requirements:
    pip install streamlit plotly pandas numpy scipy scikit-learn hmmlearn pyyaml pyarrow

Place this file alongside your project files (loader.py, factors.py, etc.)
Dataset should be at: datasets/classified_dataset.zip
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
#  Project imports — adjust sys.path if needed
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append("vamon")
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_panel, get_price_matrix, load_config
from quality import check_and_clean
from factors import momentum_signal, reversal_signal
from alpha_decay import compute_ic_series, compute_ic_stats, compute_ic_half_life, compute_ic_decay_by_horizon
from regime import build_regime_series, compute_market_vol
from optimizer import PortfolioOptimizer
from Backtester import Backtester

# ─────────────────────────────────────────────────────────────────────────────
#  Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantOps · Strategy Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS – dark quant-terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {
    --bg:         #0a0c10;
    --bg2:        #111318;
    --bg3:        #1a1d24;
    --border:     #2a2d35;
    --accent:     #00d4aa;
    --accent2:    #ff6b35;
    --accent3:    #7c6af7;
    --text:       #e2e8f0;
    --muted:      #64748b;
    --green:      #22c55e;
    --red:        #ef4444;
    --yellow:     #eab308;
  }

  html, body, [class*="css"]  {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  [data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
  }

  h1, h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif; color: var(--text) !important; }
  .stMarkdown h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }

  [data-testid="stMetric"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'IBM Plex Mono', monospace; }
  [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem !important; font-weight: 600; }
  [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem !important; }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #00b894 100%);
    color: #0a0c10;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    transition: all 0.2s ease;
  }
  .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,212,170,0.3); }

  .stSelectbox label, .stSlider label, .stMultiSelect label, .stNumberInput label { color: var(--muted) !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'IBM Plex Mono', monospace; }

  .stTabs [data-baseweb="tab-list"] { background: var(--bg2); border-bottom: 1px solid var(--border); gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; padding: 10px 20px; border-radius: 6px 6px 0 0; }
  .stTabs [aria-selected="true"] { background: var(--bg3) !important; color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

  .stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .stAlert { border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
  hr { border-color: var(--border); }

  .section-chip {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 12px;
  }

  .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; }
  .badge-green  { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
  .badge-red    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .badge-yellow { background: rgba(234,179,8,0.15);  color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
  .badge-blue   { background: rgba(124,106,247,0.15);color: #7c6af7; border: 1px solid rgba(124,106,247,0.3); }

  .logo-block { padding: 16px 0 24px 0; }
  .logo-block .logo-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: var(--accent); letter-spacing: 0.04em; }
  .logo-block .logo-sub   { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Plotly dark template
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(17,19,24,0)",
    plot_bgcolor="rgba(26,29,36,0.6)",
    font=dict(family="IBM Plex Mono, monospace", color="#e2e8f0", size=11),
    xaxis=dict(gridcolor="rgba(42,45,53,0.8)", zerolinecolor="rgba(42,45,53,0.8)"),
    yaxis=dict(gridcolor="rgba(42,45,53,0.8)", zerolinecolor="rgba(42,45,53,0.8)"),
    margin=dict(l=50, r=20, t=40, b=40),
)

COLOR_GREEN  = "#22c55e"
COLOR_RED    = "#ef4444"
COLOR_ACCENT = "#00d4aa"
COLOR_ACCENT2= "#ff6b35"
COLOR_PURPLE = "#7c6af7"
COLOR_YELLOW = "#eab308"

ZIP_PATH = "datasets/classified_dataset.zip"

# ─────────────────────────────────────────────────────────────────────────────
#  ── REAL DATA LOADING (cached) ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading panel data…")
def load_pipeline_data(zip_path: str, start_date: str, end_date: str, fit_end_date: str):
    """
    Run the full pipeline and return all artefacts needed by the dashboard.
    Cached so re-renders don't re-run the heavy computation.
    """
    cfg = load_config("vamon/config.yaml")

    # Override backtest dates from sidebar
    cfg["backtest"]["start_date"] = start_date
    cfg["backtest"]["end_date"]   = end_date

    # 1. Load & clean panel
    panel = load_panel(
        zip_path=zip_path,
        size_filter=cfg["universe"]["size_filter"],
        min_history_days=cfg["universe"]["min_history_days"],
        processed_dir="data/processed",
    )
    panel, quality_report = check_and_clean(panel)

    # 2. Price matrices
    close  = get_price_matrix(panel, "close")
    volume = get_price_matrix(panel, "volume")

    # 3. Signals
    momentum = momentum_signal(close)
    reversal = reversal_signal(close, volume)
    signals  = {"momentum": momentum, "reversal": reversal}

    # 4. Forward returns (IC computation only — no look-ahead in backtest)
    forward_returns_1d = close.pct_change().shift(-1)

    # 5. Regime
    regime_series, regime_detector = build_regime_series(
        close, cfg, fit_end_date=fit_end_date
    )

    # 6. Sector map
    sector_map = (
        panel.reset_index()
        .drop_duplicates("ticker")
        .set_index("ticker")["sector"]
        .to_dict()
    )

    # 7. Backtest
    optimizer = PortfolioOptimizer(cfg)
    bt = Backtester(panel, cfg, optimizer, sector_map)
    pnl_df, weights_df = bt.run(
        start_date=start_date,
        end_date=end_date,
        regime_series=regime_series,
    )

    # 8. IC series (use training window only — up to fit_end_date)
    ic_mom = compute_ic_series(
        momentum.loc[:fit_end_date],
        forward_returns_1d.loc[:fit_end_date],
    )
    ic_rev = compute_ic_series(
        reversal.loc[:fit_end_date],
        forward_returns_1d.loc[:fit_end_date],
    )
    ic_df = pd.DataFrame({"momentum": ic_mom, "reversal": ic_rev})

    # 9. Market vol for regime overlay
    market_vol = compute_market_vol(close, window=21)

    return dict(
        cfg=cfg,
        panel=panel,
        close=close,
        volume=volume,
        signals=signals,
        pnl_df=pnl_df,
        weights_df=weights_df,
        regime_series=regime_series,
        sector_map=sector_map,
        ic_df=ic_df,
        market_vol=market_vol,
        quality_report=quality_report,
        forward_returns_1d=forward_returns_1d,
    )


def build_positions_df(weights_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Extract latest rebalance weights as a tidy positions DataFrame."""
    if weights_df is None or weights_df.empty:
        return pd.DataFrame()
    # Last row of weights_df = most recent rebalance
    latest = weights_df.iloc[-1].dropna()
    latest = latest[latest != 0]
    df = latest.reset_index()
    df.columns = ["ticker", "weight"]
    df["sector"]     = df["ticker"].map(sector_map).fillna("Unknown")
    df["direction"]  = df["weight"].apply(lambda x: "🟢 Long" if x > 0 else "🔴 Short")
    df["market_cap"] = "Large/Mid"
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df


def attach_regime_to_pnl(pnl_df: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    """Merge regime labels into the pnl DataFrame."""
    df = pnl_df.copy()
    df["regime"] = regime_series.reindex(df.index).ffill()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ── PERFORMANCE METRICS ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pnl: pd.Series) -> dict:
    """
    Compute performance metrics using the ACTUAL observation frequency
    inferred from the datetime index.

    Assumes pnl contains periodic simple returns:
        0.01 = +1%
       -0.02 = -2%
    """

    # --------------------------------------------------
    # Clean input
    # --------------------------------------------------
    pnl = pnl.dropna().copy()

    if pnl.empty:
        return dict(
            ann_ret=np.nan,
            ann_vol=np.nan,
            sharpe=np.nan,
            max_dd=np.nan,
            calmar=np.nan,
            sortino=np.nan,
            win_rate=np.nan,
            total_return=np.nan,
            best_day=np.nan,
            worst_day=np.nan,
            cagr=np.nan,
        )

    # --------------------------------------------------
    # Cumulative curve / drawdown
    # --------------------------------------------------
    cum = (1 + pnl).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1

    total_return = cum.iloc[-1] - 1
    max_dd = dd.min()

    # --------------------------------------------------
    # Infer frequency from actual dates
    # --------------------------------------------------
    if isinstance(pnl.index, pd.DatetimeIndex) and len(pnl) > 1:
        elapsed_days = (pnl.index[-1] - pnl.index[0]).days
        n_years = max(elapsed_days / 365.25, 1 / 365.25)
    else:
        # fallback if no datetime index
        n_years = len(pnl) / 252

    periods_per_year = len(pnl) / n_years

    # --------------------------------------------------
    # Annualised metrics
    # --------------------------------------------------
    mean_ret = pnl.mean()
    std_ret = pnl.std()

    ann_ret = mean_ret * periods_per_year
    ann_vol = std_ret * np.sqrt(periods_per_year)

    sharpe = ann_ret / (ann_vol + 1e-10)

    # Downside deviation
    downside = pnl[pnl < 0]
    downside_std = downside.std()

    sortino = ann_ret / (
        downside_std * np.sqrt(periods_per_year) + 1e-10
    )

    calmar = ann_ret / (abs(max_dd) + 1e-10)

    # --------------------------------------------------
    # CAGR (geometric)
    # --------------------------------------------------
    ending_value = cum.iloc[-1]

    if ending_value > 0:
        cagr = ending_value ** (1 / n_years) - 1
    else:
        cagr = np.nan

    # --------------------------------------------------
    # Misc
    # --------------------------------------------------
    win_rate = (pnl > 0).mean()
    best_day = pnl.max()
    worst_day = pnl.min()

    return dict(
        ann_ret=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_dd=max_dd,
        calmar=calmar,
        sortino=sortino,
        win_rate=win_rate,
        total_return=total_return,
        best_day=best_day,
        worst_day=worst_day,
        cagr=cagr,
    )

# def compute_metrics(pnl: pd.Series) -> dict:
#     cum  = (1 + pnl).cumprod()
#     peak = cum.cummax()
#     dd   = (cum / peak) - 1
#     ann_ret  = pnl.mean() * 252
#     ann_vol  = pnl.std()  * np.sqrt(252)
#     sharpe   = ann_ret / (ann_vol + 1e-10)
#     max_dd   = dd.min()
#     calmar   = ann_ret / abs(max_dd + 1e-10)
#     down     = pnl[pnl < 0]
#     sortino  = ann_ret / (down.std() * np.sqrt(252) + 1e-10)
#     win_rate = (pnl > 0).mean()
#     total    = cum.iloc[-1] - 1
#
#     # ── CAGR ──────────────────────────────────────────────────────────────────
#     n_years = len(pnl) / 252
#     cagr    = (cum.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0.0
#     # ──────────────────────────────────────────────────────────────────────────
#
#     return dict(
#         ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
#         max_dd=max_dd, calmar=calmar, sortino=sortino,
#         win_rate=win_rate, total_return=total,
#         best_day=pnl.max(), worst_day=pnl.min(),
#         cagr=cagr,   # ← new
#     )

def compute_var(pnl: pd.Series, confidence: float = 0.99) -> float:
    return float(np.percentile(pnl.dropna(), (1 - confidence) * 100))


def compute_cvar(pnl: pd.Series, confidence: float = 0.99) -> float:
    var = compute_var(pnl, confidence)
    return float(pnl[pnl <= var].mean())


def drawdown_series(pnl: pd.Series) -> pd.Series:
    cum  = (1 + pnl).cumprod()
    peak = cum.cummax()
    return (cum / peak) - 1


def rolling_sharpe(pnl: pd.Series, window: int = 63) -> pd.Series:
    return (
        pnl.rolling(window).mean() * 252 /
        (pnl.rolling(window).std() * np.sqrt(252) + 1e-10)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ── STATISTICAL SIGNIFICANCE ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def compare_strategies_stat(pnl_a: pd.Series, pnl_b: pd.Series) -> dict:
    common = pnl_a.index.intersection(pnl_b.index)
    a, b   = pnl_a.loc[common], pnl_b.loc[common]
    diff   = a - b
    t, p   = stats.ttest_1samp(diff.dropna(), 0)
    rng    = np.random.default_rng(0)
    boots  = []
    for _ in range(2000):
        idx = rng.choice(len(diff), len(diff), replace=True)
        s   = diff.iloc[idx]
        boots.append(s.mean() * 252 / (s.std() * np.sqrt(252) + 1e-10))
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    sharpe_diff = (compute_metrics(a)["sharpe"] - compute_metrics(b)["sharpe"])
    return dict(t_stat=t, p_value=p, sharpe_diff=sharpe_diff,
                ci_lo=ci_lo, ci_hi=ci_hi, n_obs=len(common))


# ─────────────────────────────────────────────────────────────────────────────
#  ── CHART BUILDERS ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def chart_equity_curve(pnl_series: pd.Series, regime_series: pd.Series, show_regime: bool, name: str = "Strategy") -> go.Figure:
    fig = go.Figure()
    cum = (1 + pnl_series).cumprod() - 1

    if show_regime and regime_series is not None:
        regime = regime_series.reindex(pnl_series.index).ffill()
        in_stress, stress_start = False, None
        for dt, r in regime.items():
            if r == 1 and not in_stress:
                in_stress, stress_start = True, dt
            elif r == 0 and in_stress:
                fig.add_vrect(x0=stress_start, x1=dt, fillcolor="rgba(239,68,68,0.06)",
                              line_width=0, annotation_text="stress", annotation_font_size=9,
                              annotation_font_color="#ef4444")
                in_stress = False

    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values * 100,
        name=name,
        line=dict(color=COLOR_ACCENT, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.06)",
        hovertemplate="%{x|%Y-%m-%d}<br>Cum. Return: %{y:.2f}%<extra>" + name + "</extra>",
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Cumulative Return (%)", font=dict(size=13, color="#e2e8f0")),
        yaxis_ticksuffix="%",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(42,45,53,0.5)", borderwidth=1),
        height=360,
    )
    return fig


def chart_drawdown(pnl: pd.Series, name: str = "Strategy") -> go.Figure:
    fig = go.Figure()
    dd = drawdown_series(pnl) * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name=name, fill="tozeroy",
        line=dict(color=COLOR_RED, width=1.5),
        fillcolor="rgba(239,68,68,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra>" + name + "</extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Drawdown (%)", font=dict(size=13)),
        yaxis_ticksuffix="%",
        height=260,
    )
    return fig


def chart_rolling_sharpe(pnl: pd.Series, window: int, name: str = "Strategy") -> go.Figure:
    fig = go.Figure()
    rs = rolling_sharpe(pnl, window)
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values,
        name=name, line=dict(color=COLOR_ACCENT, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Rolling Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(100,116,139,0.5)")
    fig.add_hline(y=1, line_dash="dash", line_color="rgba(0,212,170,0.3)",
                  annotation_text="Sharpe=1", annotation_position="right",
                  annotation_font_color="#00d4aa", annotation_font_size=10)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f"Rolling {window}d Sharpe Ratio", font=dict(size=13)),
        height=260,
    )
    return fig


def chart_return_distribution(pnl: pd.Series, name: str) -> go.Figure:
    vals = pnl.dropna().values * 100
    var99 = np.percentile(vals, 1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=80, name=name,
        marker_color=COLOR_ACCENT, opacity=0.7,
        hovertemplate="Return: %{x:.3f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=var99, line_dash="dash", line_color=COLOR_RED,
                  annotation_text=f"VaR 99%<br>{var99:.3f}%",
                  annotation_position="top left",
                  annotation_font_color=COLOR_RED, annotation_font_size=10)
    x_grid = np.linspace(vals.min(), vals.max(), 300)
    mu, sigma = vals.mean(), vals.std()
    normal_y = stats.norm.pdf(x_grid, mu, sigma)
    count, _ = np.histogram(vals, bins=80)
    scale = count.max() / normal_y.max() * 0.9
    fig.add_trace(go.Scatter(
        x=x_grid, y=normal_y * scale, name="Normal fit",
        line=dict(color=COLOR_YELLOW, width=2, dash="dot"),
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Daily Return Distribution", font=dict(size=13)),
        xaxis_ticksuffix="%",
        height=300, showlegend=True,
    )
    return fig


def chart_position_heatmap(positions: pd.DataFrame) -> go.Figure:
    if positions.empty:
        return go.Figure()
    sector_exp = positions.groupby("sector")["weight"].sum().reset_index()
    sector_exp = sector_exp.sort_values("weight", ascending=False)
    colors = [COLOR_GREEN if w > 0 else COLOR_RED for w in sector_exp["weight"]]
    fig = go.Figure(go.Bar(
        x=sector_exp["sector"],
        y=sector_exp["weight"] * 100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in sector_exp["weight"] * 100],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Net Exposure: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Sector Net Exposure (%)", font=dict(size=13)),
        yaxis_ticksuffix="%",
        height=300,
        bargap=0.25,
    )
    return fig


def chart_position_treemap(positions: pd.DataFrame) -> go.Figure:
    if positions.empty:
        return go.Figure()
    df = positions.copy()
    df["abs_weight"] = df["weight"].abs() * 100
    fig = px.treemap(
        df, path=["sector", "ticker"],
        values="abs_weight", color="weight",
        color_continuous_scale=["#ef4444", "#374151", "#22c55e"],
        color_continuous_midpoint=0,
        hover_data={"weight": ":.4f"},
    )
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Position Treemap (size = |weight|, color = direction)", font=dict(size=13)),
        height=420,
        coloraxis_colorbar=dict(title="Weight", tickformat=".2f"),
    )
    fig.update_traces(marker=dict(cornerradius=4))
    return fig


def chart_ic_series(ic_df: pd.DataFrame, window: int = 63) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.06)
    colors = {"momentum": COLOR_ACCENT, "reversal": COLOR_ACCENT2}

    for col in ic_df.columns:
        rolling = ic_df[col].rolling(window, min_periods=window // 2).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            name=f"{col} ({window}d MA)",
            line=dict(color=colors.get(col, COLOR_PURPLE), width=2),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col} IC: %{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

    for col in ic_df.columns:
        cumIC = ic_df[col].fillna(0).cumsum()
        fig.add_trace(go.Scatter(
            x=cumIC.index, y=cumIC.values,
            name=f"{col} Cum IC",
            line=dict(color=colors.get(col, COLOR_PURPLE), width=1.5, dash="dot"),
        ), row=2, col=1)

    fig.add_hline(y=0,    line_color="rgba(100,116,139,0.4)", row=1, col=1)
    fig.add_hline(y=0.05, line_dash="dash", line_color="rgba(0,212,170,0.3)", row=1, col=1,
                  annotation_text="IC=0.05", annotation_position="right",
                  annotation_font_color="#00d4aa", annotation_font_size=10)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Information Coefficient (IC) — Rolling & Cumulative", font=dict(size=13)),
        height=420,
    )
    fig.update_yaxes(title_text="Rolling IC", row=1, col=1)
    fig.update_yaxes(title_text="Cum. IC",    row=2, col=1)
    return fig


def chart_ic_decay_real(signal_name: str, close: pd.DataFrame, signal_df: pd.DataFrame) -> go.Figure:
    """Real IC decay by horizon from actual data."""
    horizons = [1, 5, 10, 21, 42, 63]
    try:
        decay_df = compute_ic_decay_by_horizon(signal_df, close, horizons)
        mean_ics = decay_df["mean_ic"].tolist()
        icirs    = decay_df["icir"].tolist()
    except Exception:
        # Fallback: zeros
        mean_ics = [0.0] * len(horizons)
        icirs    = [0.0] * len(horizons)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=horizons, y=mean_ics, name="Mean IC",
        marker_color=[COLOR_ACCENT if v > 0.03 else COLOR_YELLOW for v in mean_ics],
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=horizons, y=icirs, name="ICIR",
        line=dict(color=COLOR_ACCENT2, width=2.5),
        mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f"Alpha Decay — {signal_name.title()} Signal", font=dict(size=13)),
        xaxis_title="Horizon (days)", height=320,
    )
    fig.update_yaxes(title_text="Mean IC",  secondary_y=False)
    fig.update_yaxes(title_text="ICIR",     secondary_y=True)
    return fig


def chart_regime_overlay(pnl: pd.Series, regime_series: pd.Series, market_vol: pd.Series) -> go.Figure:
    vol_21 = pnl.rolling(21).std() * np.sqrt(252) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06)

    cum = (1 + pnl).cumprod() - 1
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values * 100, name="Strategy",
        line=dict(color=COLOR_ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=vol_21.index, y=vol_21.values,
        name="21d Realised Vol",
        line=dict(color=COLOR_ACCENT2, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.08)",
    ), row=2, col=1)

    regime = regime_series.reindex(pnl.index).ffill()
    in_stress, stress_start = False, None
    for dt, r in regime.items():
        if r == 1 and not in_stress:
            in_stress, stress_start = True, dt
        elif r == 0 and in_stress:
            for row in [1, 2]:
                fig.add_vrect(x0=stress_start, x1=dt,
                              fillcolor="rgba(239,68,68,0.08)", line_width=0, row=row, col=1)
            in_stress = False

    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Regime Overlay (red = stress state)", font=dict(size=13)),
        height=400,
    )
    return fig


def chart_var_breakdown(pnl: pd.Series) -> go.Figure:
    conf   = [0.99, 0.95, 0.90]
    colors = [COLOR_RED, COLOR_YELLOW, COLOR_GREEN]
    fig = go.Figure()
    for i, c in enumerate(conf):
        var_rolling = pnl.rolling(63).quantile(1 - c) * 100
        fig.add_trace(go.Scatter(
            x=var_rolling.index, y=var_rolling.values,
            name=f"VaR {int(c*100)}%",
            line=dict(color=colors[i], width=1.8, dash="solid" if i == 0 else "dot"),
            hovertemplate=f"VaR {int(c*100)}%%: %{{y:.3f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="rgba(100,116,139,0.3)")
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Rolling Value at Risk (63d window)", font=dict(size=13)),
        yaxis_ticksuffix="%", height=300,
    )
    return fig


def chart_stress_test(pnl: pd.Series) -> go.Figure:
    scenarios = {
        "COVID Crash (Feb–Mar 2020)": ("2020-02-01", "2020-03-31"),
        "Rate Hike Shock (2022)":     ("2022-01-01", "2022-12-31"),
        "Post-COVID Rally (2021)":    ("2021-01-01", "2021-12-31"),
        "Sideways Grind (2019)":      ("2019-01-01", "2019-12-31"),
    }
    names, drawdowns, sharpes = [], [], []
    for scenario, (s, e) in scenarios.items():
        try:
            sub = pnl.loc[s:e]
            if len(sub) < 5:
                continue
            dd = drawdown_series(sub).min() * 100
            sh = sub.mean() / (sub.std() + 1e-10) * np.sqrt(252)
            names.append(scenario)
            drawdowns.append(dd)
            sharpes.append(sh)
        except Exception:
            pass

    if not names:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Max Drawdown (%)", "Annualised Sharpe"],
                        horizontal_spacing=0.14)
    dd_colors = [COLOR_RED if v < -10 else COLOR_YELLOW for v in drawdowns]
    sh_colors = [COLOR_GREEN if v > 0.5 else COLOR_RED for v in sharpes]

    fig.add_trace(go.Bar(x=names, y=drawdowns, marker_color=dd_colors,
                         text=[f"{v:.1f}%" for v in drawdowns], textposition="outside",
                         hovertemplate="%{x}<br>Max DD: %{y:.2f}%<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=sharpes,   marker_color=sh_colors,
                         text=[f"{v:.2f}" for v in sharpes], textposition="outside",
                         hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>"),
                  row=1, col=2)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Historical Stress Test Scenarios", font=dict(size=13)),
        showlegend=False, height=360,
    )
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    return fig


def chart_turnover_exposure(pnl_df: pd.DataFrame) -> go.Figure:
    if "turnover" not in pnl_df.columns:
        return go.Figure()
    turnover = pnl_df["turnover"].rolling(21).mean() * 100
    n_pos    = pnl_df.get("n_positions", pd.Series(25, index=pnl_df.index))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=turnover.index, y=turnover.values,
        name="21d MA Turnover", fill="tozeroy",
        line=dict(color=COLOR_PURPLE, width=2),
        fillcolor="rgba(124,106,247,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>Turnover: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pnl_df.index, y=n_pos,
        name="# Positions", line=dict(color=COLOR_YELLOW, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Positions: %{y}<extra></extra>",
    ), row=2, col=1)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Portfolio Turnover & Position Count", font=dict(size=13)),
        height=320,
    )
    return fig


def chart_stat_comparison(result: dict, name_a: str, name_b: str) -> go.Figure:
    ci_lo = result["ci_lo"]
    ci_hi = result["ci_hi"]
    diff  = result["sharpe_diff"]
    x = np.linspace(min(ci_lo - 0.5, -1.5), max(ci_hi + 0.5, 1.5), 400)
    mu_b = diff
    sig_b = (ci_hi - ci_lo) / (2 * 1.96)
    y = stats.norm.pdf(x, mu_b, sig_b)
    fig = go.Figure()
    mask = (x >= ci_lo) & (x <= ci_hi)
    fig.add_trace(go.Scatter(
        x=x[mask], y=y[mask], fill="tozeroy",
        fillcolor="rgba(0,212,170,0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, name="Bootstrap dist.",
        line=dict(color=COLOR_ACCENT, width=2),
    ))
    fig.add_vline(x=0,    line_dash="dash", line_color=COLOR_RED,    annotation_text="No diff",   annotation_font_color=COLOR_RED)
    fig.add_vline(x=diff, line_dash="dash", line_color=COLOR_ACCENT, annotation_text=f"Δ={diff:.2f}", annotation_font_color=COLOR_ACCENT)
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f"Sharpe Difference Bootstrap: {name_a} − {name_b}", font=dict(size=13)),
        xaxis_title="Sharpe Difference",
        height=300,
    )
    return fig


def chart_market_vol(market_vol: pd.Series, regime_series: pd.Series) -> go.Figure:
    """Market volatility with regime colouring."""
    fig = go.Figure()
    regime = regime_series.reindex(market_vol.index).ffill()
    in_stress, stress_start = False, None
    for dt, r in regime.items():
        if r == 1 and not in_stress:
            in_stress, stress_start = True, dt
        elif r == 0 and in_stress:
            fig.add_vrect(x0=stress_start, x1=dt,
                          fillcolor="rgba(239,68,68,0.07)", line_width=0)
            in_stress = False

    fig.add_trace(go.Scatter(
        x=market_vol.index, y=market_vol.values * 100,
        name="Market Vol (annualised)",
        line=dict(color=COLOR_ACCENT2, width=1.8),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.07)",
        hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Market Volatility (red = HMM stress state)", font=dict(size=13)),
        yaxis_ticksuffix="%", height=280,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ── SIDEBAR ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="logo-block">
      <div class="logo-title">⬡ QUANTOPS</div>
      <div class="logo-sub">Strategy Monitor v2.0 · Live Data</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-chip">BACKTEST WINDOW</div>', unsafe_allow_html=True)
    start_date   = st.selectbox("Start date",    ["2018-01-01", "2019-01-01", "2020-01-01"], index=0)
    end_date     = st.selectbox("End date",      ["2021-12-31", "2022-12-31", "2023-12-31"], index=0)
    fit_end_date = st.selectbox("HMM fit end",   ["2020-12-31", "2021-12-31", "2022-12-31"], index=1)

    st.markdown("---")
    st.markdown('<div class="section-chip">DISPLAY</div>', unsafe_allow_html=True)
    show_regime    = st.toggle("Show regime overlay", value=True)
    rolling_window = st.slider("Rolling window (days)", 21, 126, 63)

    st.markdown("---")
    run_btn = st.button("▶  RUN / RELOAD PIPELINE", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ── HEADER ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='margin-bottom:0'>
  <span style='color:#00d4aa'>●</span> Strategy Monitor
  <span style='font-size:0.55em; color:#64748b; font-family:"IBM Plex Mono",monospace; margin-left:12px; font-weight:400'>QUANTOPS · LIVE PIPELINE DATA</span>
</h1>
""", unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ── LOAD REAL DATA ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Clear cache if user clicks Run
if run_btn:
    st.cache_resource.clear()

if not Path(ZIP_PATH).exists():
    st.error(f"Dataset not found at `{ZIP_PATH}`. Please place `classified_dataset.zip` in the `datasets/` folder next to this file.")
    st.stop()

with st.spinner("Running pipeline (loading data, signals, backtest)… this may take ~30s on first run."):
    try:
        pipe = load_pipeline_data(ZIP_PATH, start_date, end_date, fit_end_date)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
        st.stop()

pnl_df         = pipe["pnl_df"]
weights_df     = pipe["weights_df"]
regime_series  = pipe["regime_series"]
ic_df          = pipe["ic_df"]
close          = pipe["close"]
sector_map     = pipe["sector_map"]
signals        = pipe["signals"]
market_vol     = pipe["market_vol"]
quality_report = pipe["quality_report"]

pnl_series = pnl_df["pnl"]

# Build positions from latest rebalance weights
pos_df = build_positions_df(weights_df, sector_map)

# Current regime (last known)
last_regime = regime_series.reindex(pnl_df.index).ffill().iloc[-1]
regime_label = "LOW-VOL · TRENDING" if last_regime == 0 else "HIGH-VOL · STRESS"
regime_badge_class = "badge-green" if last_regime == 0 else "badge-red"

# ─────────────────────────────────────────────────────────────────────────────
#  ── STATUS BAR ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

col_h1, col_h2, col_h3, col_h4 = st.columns([2, 2, 2, 2])
with col_h1:
    st.markdown(f'**Current Regime** <span class="badge {regime_badge_class}">{regime_label}</span>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"**Tickers** `{close.shape[1]}` in universe")
with col_h3:
    st.markdown(f"**Period** `{start_date}` → `{end_date}`")
with col_h4:
    st.markdown(f"**Rebalances** `{len(pnl_df)}` trading days")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  ── KPI STRIP ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

m = compute_metrics(pnl_series)

kpi_cols = st.columns(8)
kpis = [
    ("Total Return",  f"{m['total_return']*100:.1f}%",  f"{m['total_return']*100:+.1f}%"),
    ("Ann. Return",   f"{m['ann_ret']*100:.1f}%",       f"{m['ann_ret']*100:+.1f}%"),
    ("Ann. Vol",      f"{m['ann_vol']*100:.1f}%",       None),
    ("Sharpe",        f"{m['sharpe']:.2f}",             None),
    ("Sortino",       f"{m['sortino']:.2f}",            None),
    ("Max Drawdown",  f"{m['max_dd']*100:.1f}%",        f"{m['max_dd']*100:.2f}%"),
    ("Calmar",        f"{m['calmar']:.2f}",             None),
    ("Win Rate",      f"{m['win_rate']*100:.1f}%",      None),
    ("CAGR",          f"{m['cagr']*100:.1f}%",          f"{m['cagr']*100:+.1f}%"),
]
for col, (label, value, delta) in zip(kpi_cols, kpis):
    with col:
        if delta:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ── TABS ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Equity & Drawdown",
    "🗂  Positions",
    "🔬  Signal Analysis",
    "📊  Regime Analysis",
    "🛡  Risk Dashboard",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EQUITY & DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.plotly_chart(
        chart_equity_curve(pnl_series, regime_series, show_regime, name="Strategy"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_drawdown(pnl_series), use_container_width=True)
    with c2:
        st.plotly_chart(chart_rolling_sharpe(pnl_series, rolling_window), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_return_distribution(pnl_series, "Strategy"), use_container_width=True)
    with c4:
        st.plotly_chart(chart_turnover_exposure(pnl_df), use_container_width=True)

    # Monthly returns heatmap
    st.markdown('<div class="section-chip">MONTHLY RETURNS HEATMAP</div>', unsafe_allow_html=True)
    monthly = (
        pnl_series
        .resample("ME").apply(lambda x: (1 + x).prod() - 1)
        .to_frame("return")
    )
    monthly["year"]  = monthly.index.year
    monthly["month"] = monthly.index.month
    pivot = monthly.pivot(index="year", columns="month", values="return") * 100
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#ef4444"],[0.5,"#1a1d24"],[1,"#22c55e"]],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Ret %", ticksuffix="%"),
    ))
    fig_hm.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Monthly Returns (%)", font=dict(size=13)),
        height=280,
    )
    fig_hm.update_xaxes(side="top")
    st.plotly_chart(fig_hm, use_container_width=True)

    # PnL stats table
    st.markdown('<div class="section-chip">DAILY PNL TAIL</div>', unsafe_allow_html=True)
    st.dataframe(
        pnl_df.tail(20).style.format({
            "pnl": "{:.4f}",
            "turnover": "{:.4f}",
        }),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if pos_df.empty:
        st.warning("No position data available for the selected backtest window.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(chart_position_heatmap(pos_df), use_container_width=True)
        with c2:
            gross_exp = pos_df["weight"].abs().sum()
            net_exp   = pos_df["weight"].sum()
            long_exp  = pos_df[pos_df["weight"] > 0]["weight"].sum()
            short_exp = pos_df[pos_df["weight"] < 0]["weight"].sum()

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Gross Leverage", f"{gross_exp:.2f}x")
            g2.metric("Net Exposure",   f"{net_exp*100:.1f}%")
            g3.metric("Long Side",      f"{long_exp*100:.1f}%")
            g4.metric("Short Side",     f"{short_exp*100:.1f}%")

            st.plotly_chart(
                go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=abs(net_exp) * 100,
                    title={"text": "Net Exposure %", "font": {"color": "#e2e8f0", "size": 13}},
                    gauge=dict(
                        axis=dict(range=[0, 40], tickcolor="#64748b"),
                        bar=dict(color=COLOR_ACCENT if abs(net_exp) < 0.25 else COLOR_RED),
                        bgcolor="rgba(26,29,36,0.6)",
                        steps=[
                            dict(range=[0, 15],  color="rgba(34,197,94,0.1)"),
                            dict(range=[15, 30], color="rgba(234,179,8,0.1)"),
                            dict(range=[30, 40], color="rgba(239,68,68,0.1)"),
                        ],
                        threshold=dict(line=dict(color=COLOR_ACCENT2, width=3), thickness=0.75, value=30),
                    ),
                    number=dict(suffix="%", font=dict(color="#e2e8f0")),
                )).update_layout(**PLOTLY_THEME, height=260),
                use_container_width=True,
            )

        st.plotly_chart(chart_position_treemap(pos_df), use_container_width=True)

        st.markdown('<div class="section-chip">POSITION TABLE</div>', unsafe_allow_html=True)
        display_pos = pos_df.copy()
        display_pos["weight (%)"] = (display_pos["weight"] * 100).round(3)
        st.dataframe(
            display_pos[["ticker", "direction", "weight (%)", "sector"]]
            .sort_values("weight (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    # Data quality report
    st.markdown('<div class="section-chip">DATA QUALITY REPORT</div>', unsafe_allow_html=True)
    st.dataframe(quality_report, use_container_width=True, hide_index=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SIGNAL ANALYSIS (Alpha Decay)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-chip">INFORMATION COEFFICIENT</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_ic_series(ic_df, rolling_window), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.spinner("Computing momentum IC decay by horizon…"):
            st.plotly_chart(
                chart_ic_decay_real("momentum", close, signals["momentum"]),
                use_container_width=True,
            )
    with c2:
        with st.spinner("Computing reversal IC decay by horizon…"):
            st.plotly_chart(
                chart_ic_decay_real("reversal", close, signals["reversal"]),
                use_container_width=True,
            )

    # IC summary stats table — computed from real IC series
    st.markdown('<div class="section-chip">IC SUMMARY STATISTICS</div>', unsafe_allow_html=True)
    ic_stats_rows = []
    for sig_name in ["momentum", "reversal"]:
        ic_s = ic_df[sig_name].dropna()
        stats_d  = compute_ic_stats(ic_s, sig_name)
        half_life = compute_ic_half_life(ic_s)
        ic_stats_rows.append({
            "Signal":        sig_name.title(),
            "Mean IC":       f"{stats_d['mean_ic']:.4f}",
            "Std IC":        f"{stats_d['std_ic']:.4f}",
            "ICIR":          f"{stats_d['icir']:.4f}",
            "% Positive":    f"{stats_d['pct_positive']*100:.1f}%",
            "t-stat":        f"{stats_d['t_stat']:.2f}",
            "p-value":       f"{stats_d['p_value']:.4f}",
            "Half-life (d)": f"{half_life:.1f}" if not np.isnan(half_life) else "—",
            "Significant":   "✅ Yes" if stats_d["is_significant"] else "❌ No",
        })
    st.dataframe(pd.DataFrame(ic_stats_rows), use_container_width=True, hide_index=True)

    # Rolling IC heatmap by year-month (momentum)
    st.markdown('<div class="section-chip">MONTHLY IC HEATMAP — MOMENTUM</div>', unsafe_allow_html=True)
    ic_mom_monthly = ic_df["momentum"].resample("ME").mean().to_frame("ic")
    ic_mom_monthly["year"]  = ic_mom_monthly.index.year
    ic_mom_monthly["month"] = ic_mom_monthly.index.month
    ic_pivot = ic_mom_monthly.pivot(index="year", columns="month", values="ic")
    ic_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_ic_hm = go.Figure(go.Heatmap(
        z=ic_pivot.values, x=ic_pivot.columns.tolist(), y=ic_pivot.index.tolist(),
        colorscale=[[0,"#ef4444"],[0.5,"#1a1d24"],[1,"#22c55e"]],
        zmid=0,
        text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in ic_pivot.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} %{x}</b><br>IC: %{z:.4f}<extra></extra>",
    ))
    fig_ic_hm.update_layout(**PLOTLY_THEME, title=dict(text="Monthly Mean IC — Momentum", font=dict(size=13)), height=260)
    fig_ic_hm.update_xaxes(side="top")
    st.plotly_chart(fig_ic_hm, use_container_width=True)

    # Reversal monthly IC heatmap
    st.markdown('<div class="section-chip">MONTHLY IC HEATMAP — REVERSAL</div>', unsafe_allow_html=True)
    ic_rev_monthly = ic_df["reversal"].resample("ME").mean().to_frame("ic")
    ic_rev_monthly["year"]  = ic_rev_monthly.index.year
    ic_rev_monthly["month"] = ic_rev_monthly.index.month
    ic_rev_pivot = ic_rev_monthly.pivot(index="year", columns="month", values="ic")
    ic_rev_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_ic_hm2 = go.Figure(go.Heatmap(
        z=ic_rev_pivot.values, x=ic_rev_pivot.columns.tolist(), y=ic_rev_pivot.index.tolist(),
        colorscale=[[0,"#ef4444"],[0.5,"#1a1d24"],[1,"#22c55e"]],
        zmid=0,
        text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in ic_rev_pivot.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} %{x}</b><br>IC: %{z:.4f}<extra></extra>",
    ))
    fig_ic_hm2.update_layout(**PLOTLY_THEME, title=dict(text="Monthly Mean IC — Reversal", font=dict(size=13)), height=260)
    fig_ic_hm2.update_xaxes(side="top")
    st.plotly_chart(fig_ic_hm2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-chip">MARKET VOLATILITY & REGIME STATES</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_market_vol(market_vol, regime_series), use_container_width=True)

    if show_regime:
        st.plotly_chart(
            chart_regime_overlay(pnl_series, regime_series, market_vol),
            use_container_width=True,
        )

    # Regime distribution
    regime_in_backtest = regime_series.reindex(pnl_df.index).ffill()
    n_low  = (regime_in_backtest == 0).sum()
    n_high = (regime_in_backtest == 1).sum()

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Low-Vol Days",  str(n_low),  f"{n_low/(n_low+n_high)*100:.1f}% of backtest")
    rc2.metric("High-Vol Days", str(n_high), f"{n_high/(n_low+n_high)*100:.1f}% of backtest")
    rc3.metric("Regime Switches", str(int(regime_in_backtest.diff().abs().sum())))

    # Returns by regime
    st.markdown('<div class="section-chip">PERFORMANCE BY REGIME STATE</div>', unsafe_allow_html=True)
    pnl_with_regime = pnl_series.to_frame("pnl")
    pnl_with_regime["regime"] = regime_in_backtest

    regime_rows = []
    for r_val, r_name in [(0, "Low-Vol / Trending"), (1, "High-Vol / Stress")]:
        sub = pnl_with_regime[pnl_with_regime["regime"] == r_val]["pnl"]
        if len(sub) > 5:
            sub_m = compute_metrics(sub)
            regime_rows.append({
                "Regime":       r_name,
                "Days":         len(sub),
                "Ann. Return":  f"{sub_m['ann_ret']*100:.2f}%",
                "Ann. Vol":     f"{sub_m['ann_vol']*100:.2f}%",
                "Sharpe":       f"{sub_m['sharpe']:.3f}",
                "Max DD":       f"{sub_m['max_dd']*100:.2f}%",
                "Win Rate":     f"{sub_m['win_rate']*100:.1f}%",
            })
    if regime_rows:
        st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

    # Regime timeline
    st.markdown('<div class="section-chip">REGIME TIMELINE</div>', unsafe_allow_html=True)
    fig_regime_bar = go.Figure()
    fig_regime_bar.add_trace(go.Scatter(
        x=regime_series.index,
        y=regime_series.values,
        mode="lines",
        line=dict(color=COLOR_ACCENT, width=1),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        name="Regime (0=low-vol, 1=stress)",
        hovertemplate="%{x|%Y-%m-%d}<br>Regime: %{y}<extra></extra>",
    ))
    fig_regime_bar.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Regime State Over Time", font=dict(size=13)),
        # yaxis=dict(tickvals=[0, 1], ticktext=["Low-Vol", "Stress"]),
        height=200,
    )
    fig_regime_bar.update_yaxes(
        tickvals=[0, 1],
        ticktext=["Low-Vol", "Stress"],
    )
    st.plotly_chart(fig_regime_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    try:
        
        var99  = compute_var(pnl_series, 0.99)
        var95  = compute_var(pnl_series, 0.95)
        cvar99 = compute_cvar(pnl_series, 0.99)

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("VaR 99% (daily)",  f"{var99*100:.3f}%",  f"${abs(var99)*1e6:,.0f} / $1M")
        r2.metric("VaR 95% (daily)",  f"{var95*100:.3f}%",  f"${abs(var95)*1e6:,.0f} / $1M")
        r3.metric("CVaR 99%",         f"{cvar99*100:.3f}%", "Expected shortfall")
        r4.metric("Ann. Vol",         f"{m['ann_vol']*100:.2f}%")
        r5.metric("Max Drawdown",     f"{m['max_dd']*100:.2f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_var_breakdown(pnl_series), use_container_width=True)
        with c2:
            st.plotly_chart(chart_stress_test(pnl_series), use_container_width=True)

        # Sector exposure
        if not pos_df.empty:
            st.markdown('<div class="section-chip">SECTOR EXPOSURE BREAKDOWN</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_position_heatmap(pos_df), use_container_width=True, key="risk_sector")

        # Realised vol
        c3, c4 = st.columns(2)
        with c3:
            vol21 = pnl_series.rolling(21).std() * np.sqrt(252) * 100
            vol63 = pnl_series.rolling(63).std() * np.sqrt(252) * 100
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=vol21.index, y=vol21.values, name="21d Vol",
                                         line=dict(color=COLOR_ACCENT,  width=1.8)))
            fig_vol.add_trace(go.Scatter(x=vol63.index, y=vol63.values, name="63d Vol",
                                         line=dict(color=COLOR_ACCENT2, width=1.8, dash="dot")))
            fig_vol.update_layout(**PLOTLY_THEME,
                                   title=dict(text="Realised Volatility (%/yr)", font=dict(size=13)),
                                   yaxis_ticksuffix="%", height=280)
            st.plotly_chart(fig_vol, use_container_width=True)

        with c4:
            roll_skew = pnl_series.rolling(63).apply(lambda x: stats.skew(x),     raw=True)
            roll_kurt = pnl_series.rolling(63).apply(lambda x: stats.kurtosis(x),  raw=True)
            fig_sk = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   row_heights=[0.5, 0.5])
            fig_sk.add_trace(go.Scatter(x=roll_skew.index, y=roll_skew.values, name="Skewness",
                                        line=dict(color=COLOR_PURPLE, width=1.8)), row=1, col=1)
            fig_sk.add_trace(go.Scatter(x=roll_kurt.index, y=roll_kurt.values, name="Excess Kurtosis",
                                        line=dict(color=COLOR_YELLOW, width=1.8)), row=2, col=1)
            fig_sk.add_hline(y=0, line_color="rgba(100,116,139,0.4)", row=1, col=1)
            fig_sk.add_hline(y=0, line_color="rgba(100,116,139,0.4)", row=2, col=1)
            fig_sk.update_layout(**PLOTLY_THEME,
                                  title=dict(text="Rolling Skew & Excess Kurtosis (63d)", font=dict(size=13)),
                                  height=280)
            st.plotly_chart(fig_sk, use_container_width=True)

        # Risk summary
        st.markdown('<div class="section-chip">FULL RISK SUMMARY</div>', unsafe_allow_html=True)
        risk_summary = {
            "Metric": ["VaR 99% (daily)", "VaR 95% (daily)", "CVaR 99% (daily)", "CAGR",
                       "Ann. Volatility", "Max Drawdown", "Calmar Ratio",
                       "Sharpe Ratio", "Sortino Ratio",
                       "Skewness (full)", "Excess Kurtosis (full)",
                       "Best Day", "Worst Day", "Win Rate"],
            "Value": [
                f"{var99*100:.3f}%", f"{var95*100:.3f}%", f"{cvar99*100:.3f}%", f"{m['cagr']*100:.2f}%",
                f"{m['ann_vol']*100:.2f}%", f"{m['max_dd']*100:.2f}%", f"{m['calmar']:.3f}",
                f"{m['sharpe']:.3f}", f"{m['sortino']:.3f}",
                f"{stats.skew(pnl_series.dropna()):.3f}",
                f"{stats.kurtosis(pnl_series.dropna()):.3f}",
                f"{m['best_day']*100:.3f}%", f"{m['worst_day']*100:.3f}%",
                f"{m['win_rate']*100:.1f}%",
            ],
        }
        st.dataframe(pd.DataFrame(risk_summary), use_container_width=True, hide_index=True)

    except:
        st.error(f"Tab 5 crashed: {e}")
        st.exception(e)  # this prints the full traceback


# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#374151; font-family:"IBM Plex Mono",monospace; font-size:0.68rem; letter-spacing:0.12em'>
  QUANTOPS STRATEGY MONITOR · LIVE DATA FROM BACKTESTER.PY / FACTORS.PY / ALPHA_DECAY.PY / REGIME.PY / OPTIMIZER.PY
  <br>BACKTEST WINDOW: {start_date} → {end_date} · HMM FIT END: {fit_end_date}
  <br>ALL RETURNS ARE BACKTESTED · NOT INVESTMENT ADVICE
</div>
""", unsafe_allow_html=True)
