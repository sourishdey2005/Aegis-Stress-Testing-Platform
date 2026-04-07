"""
Aegis: Adversarial Stress Testing & Tail Risk Intelligence Platform
Full Production-Ready Streamlit Application
Made by Sourish Dey
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm, kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aegis Risk Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #FAFAFA;
    --bg-secondary: #F4F6F9;
    --bg-card: #FFFFFF;
    --bg-panel: #F0F3F8;
    --border-color: #E2E8F0;
    --border-accent: #C8D8F0;
    --text-primary: #0D1B2A;
    --text-secondary: #4A5568;
    --text-muted: #718096;
    --accent-blue: #1A56DB;
    --accent-navy: #0F3460;
    --accent-teal: #0EA5E9;
    --accent-red: #DC2626;
    --accent-amber: #D97706;
    --accent-green: #059669;
    --accent-purple: #7C3AED;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.10), 0 4px 12px rgba(0,0,0,0.06);
    --radius: 12px;
    --radius-sm: 8px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(145deg, #F0F4FF 0%, #FAFAFA 40%, #F5F0FF 100%) !important;
}

/* ── Navbar ── */
.aegis-navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-color);
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-sm);
    margin-bottom: 28px;
}
.aegis-navbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.aegis-logo {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #0F3460 0%, #1A56DB 50%, #7C3AED 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 12px rgba(26,86,219,0.30);
}
.aegis-brand-text {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #0F3460, #1A56DB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.aegis-nav-links {
    display: flex;
    gap: 20px;
    font-size: 12px;
    font-weight: 600;
}
.aegis-nav-link {
    padding: 8px 16px;
    border-radius: 6px;
    background: rgba(26, 86, 219, 0.1);
    color: var(--accent-blue);
    cursor: pointer;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.aegis-nav-link:hover {
    background: var(--accent-blue);
    color: white;
}
.aegis-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
}
.aegis-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #059669;
    box-shadow: 0 0 0 3px rgba(5,150,105,0.2);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 3px rgba(5,150,105,0.2); }
    50% { box-shadow: 0 0 0 6px rgba(5,150,105,0.1); }
}

/* ── Section Headers ── */
.section-header {
    margin: 36px 0 20px 0;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-color);
}
.section-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 6px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--text-primary);
}
.section-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 20px 0;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -1px;
    font-family: 'DM Mono', monospace;
    color: var(--text-primary);
}
.kpi-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 6px;
}
.kpi-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 8px;
}
.badge-red { background: #FEF2F2; color: var(--accent-red); }
.badge-green { background: #F0FDF4; color: var(--accent-green); }
.badge-amber { background: #FFFBEB; color: var(--accent-amber); }
.badge-blue { background: #EFF6FF; color: var(--accent-blue); }

/* ── Glass Panel ── */
.glass-panel {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow-md);
    margin-bottom: 20px;
}

/* ── Info Box ── */
.info-box {
    background: linear-gradient(135deg, #EFF6FF, #F5F3FF);
    border: 1px solid #C7D2FE;
    border-left: 4px solid var(--accent-blue);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
    margin: 16px 0;
}
.info-box-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent-navy);
    margin-bottom: 4px;
}
.info-box-text {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ── Warning Box ── */
.warn-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid var(--accent-amber);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 13px;
    color: #92400E;
}

/* ── Metric Row ── */
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.metric-chip {
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 120px;
}
.metric-chip-label { font-size: 11px; color: var(--text-muted); font-weight: 500; }
.metric-chip-val { font-size: 16px; font-weight: 700; font-family: 'DM Mono', monospace; }

/* ── Table ── */
.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
}
.styled-table th {
    background: var(--bg-panel);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}
.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    font-family: 'DM Mono', monospace;
}
.styled-table tr:last-child td { border-bottom: none; }

/* ── Regime Badge ── */
.regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.regime-crisis { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
.regime-stress { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
.regime-normal { background: #F0FDF4; color: #065F46; border: 1px solid #A7F3D0; }
.regime-bull   { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }

/* ── Footer ── */
.aegis-footer {
    text-align: center;
    padding: 40px 20px 20px;
    margin-top: 60px;
    border-top: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 13px;
}
.aegis-footer-brand {
    font-size: 15px;
    font-weight: 700;
    background: linear-gradient(135deg, #0F3460, #1A56DB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.aegis-footer-made {
    font-size: 12px;
    opacity: 0.7;
    letter-spacing: 0.5px;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #0F3460, #1A56DB) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(26,86,219,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,86,219,0.35) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0F3460 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: rgba(255,255,255,0.65) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #1A56DB !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 16px;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 24px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px !important;
    border: 1px solid var(--border-color) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
}

.stExpander {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── NAVBAR ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aegis-navbar">
    <div class="aegis-navbar-brand">
        <div class="aegis-logo">🛡️</div>
        <span class="aegis-brand-text">Aegis Risk Lab</span>
    </div>
    <div class="aegis-nav-links">
        <div class="aegis-nav-link">Data Engine</div>
        <div class="aegis-nav-link">Scenario Generator</div>
        <div class="aegis-nav-link">Simulation</div>
        <div class="aegis-nav-link">Analytics</div>
        <div class="aegis-nav-link">Export</div>
    </div>
    <div class="aegis-status">
        <div class="aegis-status-dot"></div>
        <span>System Online</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── HELPERS & ENGINES ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance."""
    frames = {}
    for tkr in tickers:
        try:
            df = yf.download(tkr, period=period, auto_adjust=True, progress=False)
            if not df.empty:
                frames[tkr] = df["Close"].squeeze()
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames).dropna(how="all")
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    return prices


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(ticker: str) -> dict:
    """Fetch live quote data."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period="2d")
        price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        chg   = (price - prev) / prev * 100 if prev else 0.0
        return {"price": price, "change_pct": chg, "ticker": ticker}
    except Exception:
        return {"price": 0.0, "change_pct": 0.0, "ticker": ticker}


def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def detect_regime(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling vol-based regime: bull/normal/stress/crisis."""
    ann_vol = returns.rolling(window).std() * np.sqrt(252)
    q25, q50, q75 = ann_vol.quantile(0.25), ann_vol.quantile(0.50), ann_vol.quantile(0.75)
    def label(v):
        if pd.isna(v): return "Unknown"
        if v < q25: return "Bull"
        if v < q50: return "Normal"
        if v < q75: return "Stress"
        return "Crisis"
    return ann_vol.map(label)


def block_bootstrap(returns: np.ndarray, n_scenarios: int, horizon: int, block_size: int = 10) -> np.ndarray:
    """Block bootstrap to preserve autocorrelation / vol clustering."""
    T = len(returns)
    scenarios = np.zeros((n_scenarios, horizon))
    for i in range(n_scenarios):
        path = []
        while len(path) < horizon:
            start = np.random.randint(0, max(1, T - block_size))
            block = returns[start: start + block_size]
            path.extend(block.tolist())
        scenarios[i] = np.array(path[:horizon])
    return scenarios


def vae_generate(returns: np.ndarray, n_scenarios: int, horizon: int,
                  tail_amplify: float = 1.0) -> np.ndarray:
    """
    Lightweight VAE-style generative model.
    Encodes returns via moment-matching, samples from learned distribution,
    and applies fat-tail augmentation.
    """
    mu     = np.mean(returns)
    sigma  = np.std(returns)
    kurt   = float(kurtosis(returns, fisher=False))
    sk     = float(skew(returns))
    df_t   = max(3.0, 6.0 / max(kurt - 3, 0.01))  # match excess kurtosis via Student-t

    # Sample from Student-t (fat tails) and skew via skew-normal mixture
    t_samples = stats.t.rvs(df=df_t, size=(n_scenarios, horizon)) * sigma + mu
    # Skew adjustment
    skew_adj  = sk * sigma * 0.1
    noise     = np.random.randn(n_scenarios, horizon) * sigma * 0.15
    scenarios = t_samples + skew_adj + noise

    # Tail amplification
    if tail_amplify > 1.0:
        bottom_mask = scenarios < np.percentile(scenarios, 5)
        scenarios[bottom_mask] *= tail_amplify

    return scenarios


def apply_adversarial_perturbations(scenarios: np.ndarray,
                                     vol_shock: float,
                                     drift_shift: float,
                                     correlation_crush: float) -> np.ndarray:
    """Apply adversarial modifications to scenario paths."""
    out = scenarios.copy()
    sigma = np.std(scenarios)
    # Vol shock: multiply extreme tails
    shock_mask = np.abs(out) > 1.5 * sigma
    out[shock_mask] *= (1 + vol_shock)
    # Drift shift (negative = bearish bias)
    out += drift_shift / out.shape[1]
    # Correlation crush: add common factor shock
    if correlation_crush > 0:
        common = np.random.randn(out.shape[1]) * sigma * correlation_crush
        out += common[np.newaxis, :]
    return out


def simulate_portfolio_paths(scenarios: np.ndarray, initial_value: float = 100.0) -> np.ndarray:
    """Convert log-return scenarios to portfolio value paths."""
    cum_returns = np.cumsum(scenarios, axis=1)
    paths = initial_value * np.exp(cum_returns)
    return paths


def stress_correlation_matrix(corr_matrix: np.ndarray, regime: str) -> np.ndarray:
    """Apply regime-dependent correlation stress."""
    n = corr_matrix.shape[0]
    stressed = corr_matrix.copy()
    if regime == "crisis":
        stressed = corr_matrix * 0.3 + 0.7 * np.ones((n, n))
        np.fill_diagonal(stressed, 1.0)
    elif regime == "anti_correlation":
        stressed = -corr_matrix * 0.6
        np.fill_diagonal(stressed, 1.0)
        stressed = np.clip(stressed, -0.99, 0.99)
        np.fill_diagonal(stressed, 1.0)
    elif regime == "regime_switch":
        stressed = corr_matrix * 0.5
        stressed[0, 1:] = -0.3
        stressed[1:, 0] = -0.3
        np.fill_diagonal(stressed, 1.0)
    return stressed


def compute_risk_metrics(paths: np.ndarray, confidence: float = 0.95) -> dict:
    """Compute comprehensive risk metrics from simulation paths."""
    final_vals = paths[:, -1]
    initial = paths[:, 0].mean()

    returns_final = (final_vals - initial) / initial

    # VaR & CVaR
    var_level  = 1 - confidence
    var        = np.percentile(returns_final, var_level * 100)
    cvar_mask  = returns_final <= var
    cvar       = returns_final[cvar_mask].mean() if cvar_mask.sum() > 0 else var

    # Max drawdown per path
    peak       = np.maximum.accumulate(paths, axis=1)
    dd         = (paths - peak) / peak
    max_dd_per = dd.min(axis=1)
    max_dd     = max_dd_per.mean()
    worst_dd   = max_dd_per.min()

    # Tail ratio (upside/downside 10%)
    top10  = np.percentile(returns_final, 90)
    bot10  = np.percentile(returns_final, 10)
    tail_ratio = abs(top10 / bot10) if bot10 != 0 else np.nan

    return {
        "var":        float(var),
        "cvar":       float(cvar),
        "max_dd_avg": float(max_dd),
        "worst_dd":   float(worst_dd),
        "tail_ratio": float(tail_ratio),
        "skewness":   float(skew(returns_final)),
        "kurtosis":   float(kurtosis(returns_final)),
        "mean_return": float(returns_final.mean()),
        "std_return":  float(returns_final.std()),
        "p5":         float(np.percentile(final_vals, 5)),
        "p50":        float(np.percentile(final_vals, 50)),
        "p95":        float(np.percentile(final_vals, 95)),
    }


def find_worst_paths(paths: np.ndarray, n: int = 5) -> tuple:
    """Find worst, tail, and adversarial paths."""
    final = paths[:, -1]
    worst_idx    = np.argsort(final)[:n]
    best_idx     = np.argsort(final)[-n:]
    median_idx   = np.argsort(np.abs(final - np.median(final)))[:1]
    return paths[worst_idx], paths[best_idx], paths[median_idx]


# ─── PLOTLY THEME ───────────────────────────────────────────────────────────

PLOT_TEMPLATE = dict(
    layout=dict(
        font=dict(family="DM Sans, sans-serif", size=12, color="#0D1B2A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,255,0.6)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
                   zeroline=False, linecolor="rgba(0,0,0,0.1)"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.08)",
                    borderwidth=1, font=dict(size=11)),
    )
)

COLORS = {
    "blue":   "#1A56DB",
    "navy":   "#0F3460",
    "teal":   "#0EA5E9",
    "red":    "#DC2626",
    "amber":  "#D97706",
    "green":  "#059669",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
}


def apply_theme(fig):
    fig.update_layout(**PLOT_TEMPLATE["layout"])
    return fig


# ─── SIDEBAR ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 16px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px;">
        <div style="font-size: 11px; letter-spacing: 2px; text-transform: uppercase; opacity: 0.5; margin-bottom: 4px;">CONFIGURATION</div>
        <div style="font-size: 16px; font-weight: 700; opacity: 0.95;">Simulation Parameters</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:0.5;margin-bottom:6px;">ASSET UNIVERSE</div>', unsafe_allow_html=True)

    asset_class = st.selectbox("Asset Class", ["Equities", "Crypto", "Mixed"])

    EQUITY_DEFAULTS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    CRYPTO_DEFAULTS = ["BTC-USD", "ETH-USD", "SOL-USD"]
    MIXED_DEFAULTS  = ["AAPL", "MSFT", "BTC-USD", "GLD", "SPY"]

    default_map = {"Equities": EQUITY_DEFAULTS, "Crypto": CRYPTO_DEFAULTS, "Mixed": MIXED_DEFAULTS}

    TRENDING_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
        "V", "XOM", "JPM", "WMT", "MA", "PG", "CVX", "HD", "LLY", "ABBV",
        "MRK", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT",
        "DHR", "NKE", "DIS", "TXN", "ADBE", "CRM", "NEE", "PM", "UPS", "RTX",
        "BMY", "UNP", "ORCL", "HON", "QCOM", "INTC", "AMD", "IBM", "SBUX", "GE",
        "CAT", "LOW", "SPY", "QQQ", "IWM", "DIA", "VTI", "BTC-USD", "ETH-USD", "SOL-USD",
        "GLD", "SLV", "TLT", "VNQ", "XLF", "XLE", "XLV", "XLK", "XLC"
    ]

    col_ticker_dropdown, col_ticker_text = st.columns([2, 1])
    with col_ticker_dropdown:
        selected_from_dropdown = st.multiselect(
            "Select Tickers (or type below)",
            options=TRENDING_TICKERS,
            default=default_map[asset_class] if asset_class in default_map else [],
            key="ticker_multiselect"
        )
    with col_ticker_text:
        tickers_input = st.text_input("Custom tickers", value="", placeholder="AAPL, MSFT, ...")
    
    dropdown_tickers = [t.strip().upper() for t in selected_from_dropdown if t.strip()]
    custom_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(dropdown_tickers + custom_tickers))

    data_period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=2)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:0.5;margin-bottom:6px;">MONTE CARLO</div>', unsafe_allow_html=True)

    n_simulations = st.select_slider("Simulations", options=[500, 1000, 5000, 10000, 25000, 50000], value=5000)
    horizon_days  = st.slider("Forecast Horizon (days)", 30, 252, 90)
    confidence    = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:0.5;margin-bottom:6px;">GENERATIVE ENGINE</div>', unsafe_allow_html=True)

    gen_model     = st.selectbox("Scenario Model", ["VAE (Fat-Tail)", "Block Bootstrap", "Both (Ensemble)"])
    tail_amplify  = st.slider("Tail Amplification ×", 1.0, 3.0, 1.5, 0.1)
    block_size    = st.slider("Bootstrap Block Size", 5, 30, 10)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:0.5;margin-bottom:6px;">ADVERSARIAL STRESS</div>', unsafe_allow_html=True)

    vol_shock      = st.slider("Volatility Shock", 0.0, 2.0, 0.5, 0.1)
    drift_shift    = st.slider("Drift Shift (bear bias)", -0.50, 0.0, -0.15, 0.01)
    corr_crush     = st.slider("Correlation Crush", 0.0, 1.0, 0.3, 0.05)
    corr_regime    = st.selectbox("Correlation Regime", ["normal", "crisis", "anti_correlation", "regime_switch"])

    st.markdown("---")
    st.markdown('<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:0.5;margin-bottom:6px;">WEIGHTS</div>', unsafe_allow_html=True)

    weight_mode = st.radio("Portfolio Weights", ["Equal Weight", "Custom"])
    custom_weights = {}
    if weight_mode == "Custom" and tickers:
        for t in tickers:
            custom_weights[t] = st.slider(f"{t} weight", 0.0, 1.0, 1.0/len(tickers), 0.05)

    st.markdown("---")
    run_btn = st.button("🚀  Run Stress Test", use_container_width=True)


# ─── MAIN CONTENT ───────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div style="text-align:center; padding: 10px 0 30px;">
    <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#1A56DB;font-weight:600;margin-bottom:10px;">INSTITUTIONAL RISK INTELLIGENCE</div>
    <h1 style="font-size:42px;font-weight:800;letter-spacing:-1.5px;color:#0D1B2A;margin:0 0 12px;">
        Aegis <span style="background:linear-gradient(135deg,#1A56DB,#7C3AED);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Stress Testing</span> Platform
    </h1>
    <p style="font-size:15px;color:#718096;max-width:600px;margin:0 auto;line-height:1.6;">
        Adversarial scenario generation · Tail risk quantification · Correlation breakdown simulation
    </p>
</div>
""", unsafe_allow_html=True)

# ─── DATA INGESTION ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-tag">MODULE 01</div>
    <div class="section-title">Data Ingestion & Market Intelligence</div>
    <div class="section-subtitle">Real-time quotes + historical training data via Yahoo Finance</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Fetching market data…"):
    prices = fetch_data(tickers, period=data_period)

if prices.empty:
    st.markdown('<div class="warn-box">⚠️ Could not fetch data for the specified tickers. Please check the symbols and try again.</div>', unsafe_allow_html=True)
    st.stop()

valid_tickers = list(prices.columns)
returns_df    = compute_returns(prices)

# Live quotes row
quote_cols = st.columns(min(len(valid_tickers), 6))
for i, tkr in enumerate(valid_tickers[:6]):
    q = fetch_live_quote(tkr)
    sign = "+" if q["change_pct"] >= 0 else ""
    badge_class = "badge-green" if q["change_pct"] >= 0 else "badge-red"
    arrow = "▲" if q["change_pct"] >= 0 else "▼"
    with quote_cols[i]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{tkr}</div>
            <div class="kpi-value" style="font-size:22px;">${q['price']:,.2f}</div>
            <span class="kpi-badge {badge_class}">{arrow} {sign}{q['change_pct']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

# Price chart with multiple visualizations
price_tab1, price_tab2, price_tab3, price_tab4, price_tab5, price_tab6, price_tab7 = st.tabs([
    "📈 Price History", "🕯️ Candlestick", "📊 Technicals", "🔗 Correlation", "📉 Risk Metrics", "🕯️ Candle Variants", "📊 Advanced Charts"
])

with price_tab1:
    fig_price = go.Figure()
    palette = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["purple"], COLORS["amber"], COLORS["teal"]]
    norm_prices = prices / prices.iloc[0] * 100
    for i, tkr in enumerate(valid_tickers):
        fig_price.add_trace(go.Scatter(
            x=norm_prices.index, y=norm_prices[tkr],
            name=tkr, line=dict(width=1.8, color=palette[i % len(palette)]),
            mode="lines"
        ))
    fig_price.update_layout(title="Normalized Price Performance (Base=100)",
                             height=420, **PLOT_TEMPLATE["layout"])
    st.plotly_chart(fig_price, use_container_width=True)

with price_tab2:
    ohlc_ticker = st.selectbox("Select Ticker for Candlestick", valid_tickers, key="ohlc_sel")
    ohlc_data = yf.download(ohlc_ticker, period=data_period, progress=False)
    
    if ohlc_data is not None and len(ohlc_data) > 0:
        ohlc_df = ohlc_data.copy()
        
        if isinstance(ohlc_df.columns, pd.MultiIndex):
            ohlc_df.columns = [c[1] if len(c) > 1 else c[0] for c in ohlc_df.columns]
        
        if 'Close' in ohlc_df.columns:
            ohlc_df = ohlc_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=ohlc_df.index,
                open=ohlc_df['Open'],
                high=ohlc_df['High'],
                low=ohlc_df['Low'],
                close=ohlc_df['Close'],
                name=ohlc_ticker,
                increasing_line_color=COLORS["green"],
                decreasing_line_color=COLORS["red"]
            ))
            
            sma_20 = ohlc_df['Close'].rolling(20).mean()
            ema_12 = ohlc_df['Close'].ewm(span=12).mean()
            fig_candle.add_trace(go.Scatter(x=sma_20.index, y=sma_20, name="SMA 20", line=dict(color=COLORS["blue"], width=1.5)))
            fig_candle.add_trace(go.Scatter(x=ema_12.index, y=ema_12, name="EMA 12", line=dict(color=COLORS["purple"], width=1.5)))
            
            fig_candle.update_layout(title="Candlestick Chart - " + ohlc_ticker, height=480)
            fig_candle.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_candle, use_container_width=True, key="candle_main")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latest Close", f"${ohlc_df['Close'].iloc[-1]:,.2f}", 
                         f"{((ohlc_df['Close'].iloc[-1] / ohlc_df['Close'].iloc[-2]) - 1) * 100:.2f}%")
            with col2:
                st.metric("Volume", f"{ohlc_df['Volume'].iloc[-1]:,.0f}")
        else:
            st.warning("No price data available for " + ohlc_ticker)
    else:
        st.warning("No data available for " + ohlc_ticker)

with price_tab3:
    tech_ticker = st.selectbox("Select Ticker for Technicals", valid_tickers, key="tech_sel")
    tech_data = prices[tech_ticker].dropna()
    
    ma_window = st.slider("MA Window", 10, 100, 20)
    tech_df = pd.DataFrame({"Price": tech_data})
    tech_df["SMA"] = tech_data.rolling(ma_window).mean()
    tech_df["EMA"] = tech_data.ewm(span=ma_window).mean()
    tech_df["Upper BB"] = tech_data.rolling(20).mean() + 2 * tech_data.rolling(20).std()
    tech_df["Lower BB"] = tech_data.rolling(20).mean() - 2 * tech_data.rolling(20).std()
    
    delta = tech_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    tech_df["RSI"] = 100 - (100 / (1 + rs))
    
    ema12 = tech_data.ewm(span=12).mean()
    ema26 = tech_data.ewm(span=26).mean()
    tech_df["MACD"] = ema12 - ema26
    tech_df["Signal"] = tech_df["MACD"].ewm(span=9).mean()
    
    fig_tech = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             row_heights=[0.4, 0.2, 0.2, 0.2],
                             subplot_titles=[f"Price & Bollinger Bands ({tech_ticker})", "Volume", "RSI", "MACD"])
    
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["Price"], name="Price",
                                  line=dict(color=COLORS["navy"], width=1.5)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["Upper BB"], name="BB Upper",
                                  line=dict(width=0), showlegend=False), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["Lower BB"], name="BB Lower",
                                  fill="tonexty", fillcolor="rgba(26,86,219,0.1)",
                                  line=dict(width=0), showlegend=False), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA"], name="SMA",
                                  line=dict(color=COLORS["blue"], width=1.2)), row=1, col=1)
    
    if len(valid_tickers) > 0:
        vol_data = yf.download(tech_ticker, period=data_period, progress=False)
        if vol_data is not None and not vol_data.empty:
            if isinstance(vol_data.columns, pd.MultiIndex):
                try:
                    vol_data = vol_data.xs(tech_ticker, axis=1)
                except:
                    vol_data = vol_data.droplevel(0, axis=1) if hasattr(vol_data.columns, 'droplevel') else vol_data
            if 'Volume' in vol_data.columns and 'Close' in vol_data.columns:
                vol_colors = ["green" if vol_data['Close'].iloc[i] >= vol_data['Open'].iloc[i] else "red" 
                              for i in range(len(vol_data))]
                fig_tech.add_trace(go.Bar(x=vol_data.index, y=vol_data['Volume'], name="Volume",
                                          marker_color=vol_colors, opacity=0.6), row=2, col=1)
    
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["RSI"], name="RSI",
                                  line=dict(color=COLORS["purple"], width=1.5)), row=3, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], row=3, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], row=3, col=1)
    
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["MACD"], name="MACD",
                                  line=dict(color=COLORS["blue"], width=1.5)), row=4, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df["Signal"], name="Signal",
                                  line=dict(color=COLORS["red"], width=1.5)), row=4, col=1)
    fig_tech.add_bar(x=tech_df.index, y=tech_df["MACD"] - tech_df["Signal"], name="Histogram",
                    marker_color=COLORS["teal"], opacity=0.5, row=4, col=1)
    
    fig_tech.update_layout(height=700, **PLOT_TEMPLATE["layout"])
    st.plotly_chart(fig_tech, use_container_width=True)

with price_tab4:
    corr_matrix = returns_df[valid_tickers].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Asset Correlation Matrix")
    fig_corr.update_layout(height=450, **PLOT_TEMPLATE["layout"])
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("**Strongest Correlations:**")
    corr_pairs = []
    for i, t1 in enumerate(valid_tickers):
        for j, t2 in enumerate(valid_tickers):
            if i < j:
                corr_pairs.append((t1, t2, corr_matrix.loc[t1, t2]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for t1, t2, c in corr_pairs[:5]:
        color = "green" if c > 0 else "red"
        st.markdown(f"- **{t1}** ↔ **{t2}**: <span style='color:{color}'>{c:.3f}</span>", unsafe_allow_html=True)

with price_tab5:
    vol_window = st.slider("Rolling Volatility Window", 10, 100, 30)
    fig_risk = make_subplots(rows=2, cols=2,
                             subplot_titles=["Rolling Volatility (30d)", "Cumulative Returns", 
                                           "Drawdown Series", "Return Quantiles"])
    
    for i, tkr in enumerate(valid_tickers[:4]):
        roll_vol = returns_df[tkr].rolling(vol_window).std() * np.sqrt(252) * 100
        fig_risk.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, name=tkr,
                                      line=dict(width=1.5, color=palette[i])), row=1, col=1)
    
    for i, tkr in enumerate(valid_tickers[:4]):
        cum_ret = (1 + returns_df[tkr]).cumprod()
        fig_risk.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name=tkr,
                                      line=dict(width=1.5, color=palette[i])), row=1, col=2)
    
    for i, tkr in enumerate(valid_tickers[:4]):
        cum_ret = (1 + returns_df[tkr]).cumprod()
        roll_max = cum_ret.cummax()
        drawdown = (cum_ret - roll_max) / roll_max * 100
        fig_risk.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name=tkr,
                                      line=dict(width=1.5, color=palette[i])), row=2, col=1)
    
    for i, tkr in enumerate(valid_tickers[:4]):
        ret_data = returns_df[tkr].dropna()
        fig_risk.add_trace(go.Box(y=ret_data, name=tkr, 
                                  marker_color=palette[i], 
                                  boxpoints=False), row=2, col=2)
    
    fig_risk.update_layout(height=600, showlegend=False, **PLOT_TEMPLATE["layout"])
    st.plotly_chart(fig_risk, use_container_width=True)

    n_assets = len(valid_tickers)
    vis_wts = np.ones(n_assets) / n_assets if n_assets > 0 else np.array([1.0])
    
    st.markdown("**📊 Treemap — Portfolio Weight Distribution**")
    treemap_data = pd.DataFrame({
        "Asset": valid_tickers,
        "Weight": vis_wts * 100,
        "Color": palette[:len(valid_tickers)]
    })
    fig_tree = px.treemap(treemap_data, path=["Asset"], values="Weight",
                          color="Weight", color_continuous_scale="Blues")
    fig_tree.update_layout(height=350, **PLOT_TEMPLATE["layout"])
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("**🕸️ Radar Chart — Asset Performance Comparison**")
    radar_metrics = []
    for tkr in valid_tickers[:6]:
        ret = returns_df[tkr].dropna()
        radar_metrics.append({
            "Asset": tkr,
            "Return": ret.mean() * 252 * 100,
            "Volatility": ret.std() * np.sqrt(252) * 100,
            "Sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0,
            "Max DD": ((ret.cumprod() - ret.cumprod().cummax()) / ret.cumprod().cummax()).min() * 100,
            "Skewness": skew(ret),
            "Kurtosis": kurtosis(ret)
        })
    radar_df = pd.DataFrame(radar_metrics)
    
    fig_radar = go.Figure()
    for i, row in radar_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Return"], row["Volatility"], row["Sharpe"], abs(row["Max DD"]), abs(row["Skewness"]), row["Kurtosis"]/10, row["Return"]],
            theta=["Return", "Volatility", "Sharpe", "Max DD", "Skew", "Kurt", "Return"],
            fill='toself',
            name=row["Asset"],
            line_color=palette[i]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=400, **PLOT_TEMPLATE["layout"]
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")

# ── Candlestick Variants Tab ─────────────────────────────────────────────
with price_tab6:
    candle_ticker = st.selectbox("Select Ticker for Candle Variants", valid_tickers, key="candle_var_sel")
    candle_data = yf.download(candle_ticker, period=data_period, progress=False)
    
    if candle_data is not None and not candle_data.empty:
        candle_df = candle_data.copy()
        if isinstance(candle_df.columns, pd.MultiIndex):
            try:
                candle_df = candle_df.xs(candle_ticker, axis=1)
            except:
                candle_df = candle_df.droplevel(0, axis=1) if hasattr(candle_df.columns, 'droplevel') else candle_df
        
        if 'Close' in candle_df.columns:
            candle_df = candle_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            st.markdown("### 🕯️ 20 Candlestick Chart Variants")
            
            variant_charts = [
                ("1️⃣ Standard Candlestick", "standard"),
                ("2️⃣ OHLC Bars", "ohlc_bars"),
                ("3️⃣ Line Chart", "line"),
                ("4️⃣ Area Chart", "area"),
                ("5️⃣ Heikin-Ashi", "heikin_ashi"),
                ("6️⃣ Renko Chart", "renko"),
                ("7️⃣ Point & Figure", "pnf"),
                ("8️⃣ Kagi Chart", "kagi"),
                ("9️⃣ Range Bars", "range_bars"),
                ("🔟 Candle with Volume", "candle_volume"),
                ("1️⃣1️⃣ Candle with MA", "candle_ma"),
                ("1️⃣2️⃣ Candle with BB", "candle_bb"),
                ("1️⃣3️⃣ Hollow Candles", "hollow"),
                ("1️⃣4️⃣ Colored OHLC", "colored_ohlc"),
                ("1️⃣5️⃣ Trend Columns", "trend_cols"),
                ("1️⃣6️⃣ Price Channels", "price_channels"),
                ("1️⃣7️⃣ Ichimoku Cloud", "ichimoku"),
                ("1️⃣8️⃣ Zigzag Lines", "zigzag"),
                ("1️⃣9️⃣ Multi-Timeframe", "mtf"),
                ("2️⃣0️⃣ Comparison Overlay", "comparison")
            ]
            
            var_cols = st.columns(2)
            for idx, (var_name, var_type) in enumerate(variant_charts):
                with var_cols[idx % 2]:
                    try:
                        fig_var = create_candle_variant(candle_df, var_type, candle_ticker, COLORS)
                        st.plotly_chart(fig_var, use_container_width=True, key=f"candle_{idx}")
                    except Exception as e:
                        st.caption(f"{var_name}: N/A")

def create_candle_variant(df, variant_type, ticker, COLORS):
    fig = go.Figure()
    
    if variant_type == "standard":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker, increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]
        ))
        fig.update_layout(title=f"Standard Candlestick — {ticker}", height=280)
        
    elif variant_type == "ohlc_bars":
        fig.add_trace(go.Ohlc(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker, increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]
        ))
        fig.update_layout(title=f"OHLC Bars — {ticker}", height=280)
        
    elif variant_type == "line":
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name=ticker,
            line=dict(color=COLORS["navy"], width=1.5)
        ))
        fig.update_layout(title=f"Line Chart — {ticker}", height=280)
        
    elif variant_type == "area":
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name=ticker,
            fill='tozeroy', fillcolor=f"rgba(26,86,219,0.2)",
            line=dict(color=COLORS["blue"], width=1.5)
        ))
        fig.update_layout(title=f"Area Chart — {ticker}", height=280)
        
    elif variant_type == "heikin_ashi":
        ha = pd.DataFrame()
        ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha['Open'] = (ha['Close'].shift(1) + df['Open'].shift(1)) / 2
        ha['High'] = pd.concat([df['High'], ha['Open'], ha['Close']], axis=1).max(axis=1)
        ha['Low'] = pd.concat([df['Low'], ha['Open'], ha['Close']], axis=1).min(axis=1)
        fig.add_trace(go.Candlestick(
            x=ha.index, open=ha['Open'], high=ha['High'], low=ha['Low'], close=ha['Close'],
            name="Heikin-Ashi"
        ))
        fig.update_layout(title=f"Heikin-Ashi — {ticker}", height=280)
        
    elif variant_type == "renko":
        renko_data = df['Close'].diff().dropna()
        renko_bricks = []
        bricks = 0
        for r in renko_data:
            if abs(r) > df['Close'].std() * 0.5:
                bricks += 1 if r > 0 else -1
                renko_bricks.append(bricks)
        fig.add_trace(go.Scatter(
            x=list(range(len(renko_bricks))), y=renko_bricks, mode='lines',
            line=dict(color=COLORS["blue"], width=2)
        ))
        fig.update_layout(title=f"Renko-style — {ticker}", height=280)
        
    elif variant_type == "pnf":
        pnf_data = df['Close'].diff()
        pnf_col = ["green" if x > 0 else "red" for x in pnf_data]
        fig.add_trace(go.Bar(
            x=df.index, y=pnf_data, marker_color=pnf_col
        ))
        fig.update_layout(title=f"Point & Figure style — {ticker}", height=280)
        
    elif variant_type == "kagi":
        kagi_data = df['Close'].cumsum()
        fig.add_trace(go.Scatter(
            x=kagi_data.index, y=kagi_data, mode='lines',
            line=dict(color=COLORS["purple"], width=2)
        ))
        fig.update_layout(title=f"Kagi-style — {ticker}", height=280)
        
    elif variant_type == "range_bars":
        fig.add_trace(go.Bar(
            x=df.index, y=df['High'] - df['Low'],
            marker_color=COLORS["teal"]
        ))
        fig.update_layout(title=f"Range Bars — {ticker}", height=280)
        
    elif variant_type == "candle_volume":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker
        ))
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name="Volume",
            marker_color=['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                         for i in range(len(df))], opacity=0.5
        ))
        fig.update_layout(title=f"Candlestick + Volume — {ticker}", height=280)
        
    elif variant_type == "candle_ma":
        sma20 = df['Close'].rolling(20).mean()
        sma50 = df['Close'].rolling(50).mean()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker
        ))
        fig.add_trace(go.Scatter(x=sma20.index, y=sma20, name="SMA 20", line=dict(color=COLORS["blue"])))
        fig.add_trace(go.Scatter(x=sma50.index, y=sma50, name="SMA 50", line=dict(color=COLORS["red"])))
        fig.update_layout(title=f"Candlestick + Moving Averages — {ticker}", height=280)
        
    elif variant_type == "candle_bb":
        ma = df['Close'].rolling(20).std()
        upper = df['Close'].rolling(20).mean() + 2 * ma
        lower = df['Close'].rolling(20).mean() - 2 * ma
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker
        ))
        fig.add_trace(go.Scatter(x=upper.index, y=upper, name="Upper BB", line=dict(color=COLORS["red"], dash='dash')))
        fig.add_trace(go.Scatter(x=lower.index, y=lower, name="Lower BB", line=dict(color=COLORS["green"], dash='dash')))
        fig.update_layout(title=f"Candlestick + Bollinger Bands — {ticker}", height=280)
        
    elif variant_type == "hollow":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker,
            increasing_line_color='white', decreasing_line_color=COLORS["red"],
            increasing_fillcolor='white', decreasing_fillcolor=COLORS["red"]
        ))
        fig.update_layout(title=f"Hollow Candles — {ticker}", height=280)
        
    elif variant_type == "colored_ohlc":
        colors = [COLORS["green"] if df['Close'].iloc[i] >= df['Open'].iloc[i] else COLORS["red"] 
                  for i in range(len(df))]
        fig.add_trace(go.Ohlc(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=ticker, marker_color=colors
        ))
        fig.update_layout(title=f"Colored OHLC — {ticker}", height=280)
        
    elif variant_type == "trend_cols":
        trend = [1 if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else -1 for i in range(1, len(df))]
        trend.insert(0, 0)
        colors = [COLORS["green"] if t > 0 else COLORS["red"] for t in trend]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Close'].diff(), marker_color=colors
        ))
        fig.update_layout(title=f"Trend Columns — {ticker}", height=280)
        
    elif variant_type == "price_channels":
        high_ch = df['High'].rolling(10).max()
        low_ch = df['Low'].rolling(10).min()
        mid_ch = (high_ch + low_ch) / 2
        fig.add_trace(go.Scatter(
            x=high_ch.index, y=high_ch, name="Channel High",
            line=dict(color=COLORS["red"], width=1), mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=low_ch.index, y=low_ch, name="Channel Low",
            line=dict(color=COLORS["green"], width=1), mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=mid_ch.index, y=mid_ch, name="Mid",
            line=dict(color=COLORS["blue"], width=2), mode='lines'
        ))
        fig.update_layout(title=f"Price Channels — {ticker}", height=280)
        
    elif variant_type == "ichimoku":
        nine_period = df['High'].rolling(9).max()
        nine_low = df['Low'].rolling(9).min()
        tenkan = (nine_period + nine_low) / 2
        
        twenty_six_period = df['High'].rolling(26).max()
        twenty_six_low = df['Low'].rolling(26).min()
        kijun = (twenty_six_period + twenty_six_low) / 2
        
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        fig.add_trace(go.Scatter(
            x=tenkan.index, y=tenkan, name="Tenkan-sen",
            line=dict(color=COLORS["blue"], width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=kijun.index, y=kijun, name="Kijun-sen",
            line=dict(color=COLORS["red"], width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=senkou_a.index, y=senkou_a, name="Senkou Span A",
            line=dict(color=COLORS["green"], width=1), fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        fig.update_layout(title=f"Ichimoku Cloud — {ticker}", height=280)
        
    elif variant_type == "zigzag":
        zigzag = df['Close'].diff()
        fig.add_trace(go.Scatter(
            x=zigzag.index, y=zigzag, mode='lines',
            line=dict(color=COLORS["navy"], width=1.5)
        ))
        fig.update_layout(title=f"Zigzag Lines — {ticker}", height=280)
        
    elif variant_type == "mtf":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Daily"
        ))
        fig.update_layout(title=f"Multi-Timeframe — {ticker}", height=280)
        
    elif variant_type == "comparison":
        norm_close = df['Close'] / df['Close'].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm_close, mode='lines', name="Price (normalized)",
            line=dict(color=COLORS["navy"], width=2), fill='tozeroy',
            fillcolor='rgba(26,86,219,0.1)'
        ))
        fig.add_hline(y=100, line_dash="dash", line_color=COLORS["gray"])
        fig.update_layout(title=f"Comparison — {ticker} (Base=100)", height=280)
    
    fig.update_layout(
        template="plotly_white",
        font=dict(family="DM Sans", size=10),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    )
    return fig

# ── Advanced Charts Tab ──────────────────────────────────────────────────
with price_tab7:
    adv_ticker = st.selectbox("Select Ticker for Advanced Charts", valid_tickers, key="adv_chart_sel")
    adv_data = yf.download(adv_ticker, period=data_period, progress=False)
    
    if adv_data is not None and not adv_data.empty:
        adv_df = adv_data.copy()
        if isinstance(adv_df.columns, pd.MultiIndex):
            try:
                adv_df = adv_df.xs(adv_ticker, axis=1)
            except:
                adv_df = adv_df.droplevel(0, axis=1) if hasattr(adv_df.columns, 'droplevel') else adv_df
        
        if 'Close' in adv_df.columns:
            adv_df = adv_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            st.markdown("### 📊 30+ Advanced Chart Types")
            
            adv_charts = [
                ("1️⃣ Line Chart", "line"),
                ("2️⃣ Smooth Line", "smooth"),
                ("3️⃣ Step Line", "step"),
                ("4️⃣ Area Fill", "area"),
                ("5️⃣ Stacked Area", "stacked_area"),
                ("6️⃣ Bar Chart", "bar"),
                ("7️⃣ Grouped Bar", "grouped_bar"),
                ("8️⃣ Stacked Bar", "stacked_bar"),
                ("9️⃣ Horizontal Bar", "h_bar"),
                ("🔟 Waterfall", "waterfall"),
                ("1️⃣1️⃣ Funnel", "funnel"),
                ("1️⃣2️⃣ Pie Chart", "pie"),
                ("1️⃣3️⃣ Donut Chart", "donut"),
                ("1️⃣4️⃣ Treemap", "treemap"),
                ("1️⃣5️⃣ Sunburst", "sunburst"),
                ("1️⃣6️⃣ Parallel Categories", "par_cat"),
                ("1️⃣7️⃣ Scatter Plot", "scatter"),
                ("1️⃣8️⃣ Bubble Chart", "bubble"),
                ("1️⃣9️⃣ Dot Plot", "dot"),
                ("2️⃣0️⃣ Histogram", "histogram"),
                ("2️⃣1️⃣ Box Plot", "box"),
                ("2️⃣2️⃣ Violin Plot", "violin"),
                ("2️⃣3️⃣ Strip Plot", "strip"),
                ("2️⃣4️⃣ ECDF Plot", "ecdf"),
                ("2️⃣5️⃣ QQ Plot", "qq"),
                ("2️⃣6️⃣ Density Contour", "density"),
                ("2️⃣7️⃣ Heatmap", "heatmap"),
                ("2️⃣8️⃣ 3D Scatter", "scatter3d"),
                ("2️⃣9️⃣ 3D Surface", "surface3d"),
                ("3️⃣0️⃣ Polar Chart", "polar"),
                ("3️⃣1️⃣ Radar Fill", "radar_fill"),
                ("3️⃣2️⃣ Horizontal Line", "h_line"),
                ("3️⃣3️⃣ Candle + Volume Profile", "vol_profile"),
                ("3️⃣4️⃣ Return Decomposition", "decomp"),
                ("3️⃣5️⃣ Rolling Regression", "roll_reg")
            ]
            
            chart_cols = st.columns(2)
            for idx, (chart_name, chart_type) in enumerate(adv_charts):
                with chart_cols[idx % 2]:
                    try:
                        fig_adv = create_advanced_chart(adv_df, chart_type, adv_ticker, COLORS)
                        st.plotly_chart(fig_adv, use_container_width=True, key=f"adv_{idx}")
                    except Exception as e:
                        st.caption(f"{chart_name}: N/A")

def create_advanced_chart(df, chart_type, ticker, COLORS):
    fig = go.Figure()
    
    if chart_type == "line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=ticker,
                                line=dict(color=COLORS["navy"], width=2)))
        fig.update_layout(title=f"Line Chart — {ticker}", height=280)
        
    elif chart_type == "smooth":
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(df['Close'].values, sigma=3)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Original",
                                line=dict(color='rgba(0,0,0,0.2)', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=smoothed, mode='lines', name="Smoothed",
                                line=dict(color=COLORS["blue"], width=2)))
        fig.update_layout(title=f"Smooth Line — {ticker}", height=280)
        
    elif chart_type == "step":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=ticker,
                                line=dict(shape='hv', color=COLORS["purple"], width=2)))
        fig.update_layout(title=f"Step Line — {ticker}", height=280)
        
    elif chart_type == "area":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', fill='tozeroy',
                                name=ticker, fillcolor='rgba(26,86,219,0.3)',
                                line=dict(color=COLORS["blue"], width=1.5)))
        fig.update_layout(title=f"Area Chart — {ticker}", height=280)
        
    elif chart_type == "stacked_area":
        close_prices = prices.dropna(axis=1, how='all')
        for col in close_prices.columns[:5]:
            fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices[col], mode='lines', name=col,
                                    stackgroup='one', fillcolor=palette[len(fig.data) % len(palette)]))
        fig.update_layout(title="Stacked Area — Portfolio", height=280)
        
    elif chart_type == "bar":
        fig.add_trace(go.Bar(x=df.index[-30:], y=df['Close'].diff().iloc[-30:],
                            marker_color=COLORS["teal"]))
        fig.update_layout(title=f"Bar Chart — {ticker}", height=280)
        
    elif chart_type == "grouped_bar":
        daily_ret = df['Close'].pct_change().iloc[-20:]
        fig.add_trace(go.Bar(x=daily_ret.index, y=daily_ret, name="Daily Returns",
                            marker_color=[COLORS["green"] if x >= 0 else COLORS["red"] for x in daily_ret]))
        fig.update_layout(title=f"Grouped Bar — {ticker}", height=280)
        
    elif chart_type == "stacked_bar":
        fig.add_trace(go.Bar(x=df.index[-10:], y=df['Open'].iloc[-10:], name="Open"))
        fig.add_trace(go.Bar(x=df.index[-10:], y=df['Close'].iloc[-10:], name="Close"))
        fig.update_layout(title=f"Stacked Bar — {ticker}", barmode='stack', height=280)
        
    elif chart_type == "h_bar":
        fig.add_trace(go.Bar(x=df['Close'].iloc[-15:], y=df.index[-15:], orientation='h',
                            marker_color=COLORS["amber"]))
        fig.update_layout(title=f"Horizontal Bar — {ticker}", height=280)
        
    elif chart_type == "waterfall":
        changes = df['Close'].diff().iloc[-15:]
        fig.add_trace(go.Waterfall(
            x=changes.index, y=changes,
            measure=["relative"] * len(changes),
            decreasing=dict(marker_color=COLORS["red"]),
            increasing=dict(marker_color=COLORS["green"]),
            totals=dict(marker_color=COLORS["blue"])
        ))
        fig.update_layout(title=f"Waterfall — {ticker}", height=280)
        
    elif chart_type == "funnel":
        stages = ["Awareness", "Interest", "Consideration", "Intent", "Purchase"]
        values = [1000, 750, 500, 250, 100]
        fig.add_trace(go.Funnel(y=stages, x=values, textinfo="value+percent", marker=dict(color=COLORS["blue"])))
        fig.update_layout(title="Funnel Chart", height=280)
        
    elif chart_type == "pie":
        monthly_ret = df['Close'].pct_change().resample('M').mean().iloc[-6:]
        fig.add_trace(go.Pie(labels=monthly_ret.index.strftime('%b'), values=monthly_ret.values,
                           marker=dict(colors=palette[:len(monthly_ret)])))
        fig.update_layout(title=f"Pie Chart — {ticker}", height=280)
        
    elif chart_type == "donut":
        monthly_ret = df['Close'].pct_change().resample('M').mean().iloc[-6:]
        fig.add_trace(go.Pie(labels=monthly_ret.index.strftime('%b'), values=monthly_ret.values,
                           hole=0.5, marker=dict(colors=palette[:len(monthly_ret)])))
        fig.update_layout(title=f"Donut Chart — {ticker}", height=280)
        
    elif chart_type == "treemap":
        treemap_data = pd.DataFrame({
            "Asset": valid_tickers,
            "Weight": vis_wts * 100
        })
        fig = px.treemap(treemap_data, path=["Asset"], values="Weight",
                        color="Weight", color_continuous_scale="Blues")
        fig.update_layout(title="Treemap", height=350)
        
    elif chart_type == "sunburst":
        sun_data = pd.DataFrame({
            "Asset": valid_tickers,
            "Sector": ["Tech"] * len(valid_tickers),
            "Weight": vis_wts * 100
        })
        fig = px.sunburst(sun_data, path=["Sector", "Asset"], values="Weight")
        fig.update_layout(title="Sunburst", height=350)
        
    elif chart_type == "par_cat":
        par_data = pd.DataFrame({
            "Asset": valid_tickers[:5],
            "Sector": ["Tech", "Finance", "Energy", "Healthcare", "Consumer"],
            "Return": np.random.randn(5) * 10
        })
        fig = px.parallel_categories(par_data, dimensions=["Sector", "Asset"], color="Return",
                                    color_continuous_scale="RdBu")
        fig.update_layout(title="Parallel Categories", height=280)
        
    elif chart_type == "scatter":
        fig.add_trace(go.Scatter(x=df['Volume'], y=df['Close'], mode='markers',
                                marker=dict(size=6, color=df.index, colorscale='Viridis')))
        fig.update_layout(title=f"Scatter Plot — {ticker}", height=280)
        
    elif chart_type == "bubble":
        fig.add_trace(go.Scatter(x=df.index[-30:], y=df['Close'].iloc[-30:],
                                mode='markers', marker=dict(
                                size=df['Volume'].iloc[-30:] / df['Volume'].iloc[-30:].max() * 20,
                                color=COLORS["blue"], opacity=0.6)))
        fig.update_layout(title=f"Bubble Chart — {ticker}", height=280)
        
    elif chart_type == "dot":
        fig.add_trace(go.Scatter(x=df['Close'].iloc[-20:], y=[ticker] * 20, mode='markers',
                                marker=dict(size=12, color=COLORS["purple"])))
        fig.update_layout(title=f"Dot Plot — {ticker}", height=280)
        
    elif chart_type == "histogram":
        ret = df['Close'].pct_change().dropna()
        fig.add_trace(go.Histogram(x=ret, nbinsx=40, marker_color=COLORS["teal"], opacity=0.7))
        fig.update_layout(title=f"Histogram — {ticker}", height=280)
        
    elif chart_type == "box":
        ret = df['Close'].pct_change().dropna()
        fig.add_trace(go.Box(y=ret, name=ticker, marker_color=COLORS["amber"]))
        fig.update_layout(title=f"Box Plot — {ticker}", height=280)
        
    elif chart_type == "violin":
        ret = df['Close'].pct_change().dropna()
        fig.add_trace(go.Violin(y=ret, name=ticker, box_visible=True, meanline_visible=True,
                               marker_color=COLORS["purple"]))
        fig.update_layout(title=f"Violin Plot — {ticker}", height=280)
        
    elif chart_type == "strip":
        ret = df['Close'].pct_change().dropna()
        fig.add_trace(go.Box(y=ret, name=ticker, marker=dict(color=COLORS["green"])))
        fig.add_trace(go.Box(y=ret * 0.5, name=ticker + " Scaled"))
        fig.update_layout(title=f"Strip Plot — {ticker}", height=280)
        
    elif chart_type == "ecdf":
        ret = df['Close'].pct_change().dropna().sort_values()
        ecdf = np.arange(1, len(ret) + 1) / len(ret)
        fig.add_trace(go.Scatter(x=ret, y=ecdf, mode='lines', line=dict(color=COLORS["navy"], width=2)))
        fig.update_layout(title=f"ECDF — {ticker}", height=280)
        
    elif chart_type == "qq":
        ret = df['Close'].pct_change().dropna().sort_values()
        theoretical = norm.ppf(np.linspace(0.01, 0.99, len(ret)))
        fig.add_trace(go.Scatter(x=theoretical, y=ret.values[:len(theoretical)], mode='markers',
                                marker=dict(size=4, color=COLORS["blue"])))
        fig.add_trace(go.Scatter(x=[-4, 4], y=[-4, 4], mode='lines', line=dict(color=COLORS["red"], dash='dash')))
        fig.update_layout(title=f"QQ Plot — {ticker}", height=280)
        
    elif chart_type == "density":
        x_data = df['Close'].pct_change().dropna().values
        y_data = df['Volume'].pct_change().dropna().values[:len(x_data)]
        fig.add_trace(go.Histogram2d(x=x_data, y=y_data, colorscale='Blues'))
        fig.update_layout(title=f"Density Contour — {ticker}", height=280)
        
    elif chart_type == "heatmap":
        corr_data = returns_df[valid_tickers[:6]].corr()
        fig.add_trace(go.Heatmap(z=corr_data, x=corr_data.columns, y=corr_data.columns,
                                 colorscale='RdBu_r', zmid=0))
        fig.update_layout(title="Correlation Heatmap", height=350)
        
    elif chart_type == "scatter3d":
        dates = df.index[:50]
        fig.add_trace(go.Scatter3d(x=dates, y=df['Open'].iloc[:50], z=df['Close'].iloc[:50],
                                  mode='markers', marker=dict(size=4, color=COLORS["blue"])))
        fig.update_layout(title=f"3D Scatter — {ticker}", height=350)
        
    elif chart_type == "surface3d":
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        z = np.outer(x, y) * np.sin(x) * np.cos(y)
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis'))
        fig.update_layout(title="3D Surface", height=350)
        
    elif chart_type == "polar":
        categories = ['Return', 'Vol', 'Sharpe', 'MaxDD', 'Beta', 'Alpha']
        values = [5, 8, 3, -6, 2, 1]
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=ticker,
                                     marker=dict(color=COLORS["blue"])))
        fig.update_layout(title=f"Polar Chart — {ticker}", polar=dict(radialaxis=dict(visible=True)),
                         height=280)
        
    elif chart_type == "radar_fill":
        categories = ['Return', 'Vol', 'Sharpe', 'MaxDD', 'Beta', 'Alpha']
        values1 = [5, 8, 3, -6, 2, 1]
        values2 = [3, 6, 2, -4, 1, 0.5]
        fig.add_trace(go.Scatterpolar(r=values1, theta=categories, fill='toself', name="Portfolio",
                                     marker=dict(color=COLORS["blue"])))
        fig.add_trace(go.Scatterpolar(r=values2, theta=categories, fill='toself', name="Benchmark",
                                     marker=dict(color=COLORS["red"], opacity=0.5)))
        fig.update_layout(title="Radar Fill", polar=dict(radialaxis=dict(visible=True)), height=280)
        
    elif chart_type == "h_line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines',
                                line=dict(color=COLORS["navy"], width=2)))
        fig.add_hline(y=df['Close'].mean(), line_dash="dash", line_color=COLORS["red"],
                     annotation_text="Mean")
        fig.update_layout(title=f"With Horizontal Line — {ticker}", height=280)
        
    elif chart_type == "vol_profile":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'], name=ticker))
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], yaxis='y2', opacity=0.3,
                            marker_color='gray'))
        fig.update_layout(title=f"Candle + Volume — {ticker}", height=280,
                         yaxis2=dict(title="Volume", overlaying='y', side='right'))
        
    elif chart_type == "decomp":
        close = df['Close']
        returns = close.pct_change().dropna()
        cum_ret = (1 + returns).cumprod()
        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, mode='lines',
                                fill='tozeroy', fillcolor='rgba(26,86,219,0.2)',
                                line=dict(color=COLORS["navy"])))
        fig.update_layout(title=f"Return Decomposition — {ticker}", height=280)
        
    elif chart_type == "roll_reg":
        roll_corr = df['Close'].rolling(20).corr(df['Volume'])
        fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode='lines',
                                line=dict(color=COLORS["purple"], width=2)))
        fig.update_layout(title=f"Rolling Correlation — {ticker}", height=280)
    
    fig.update_layout(
        template="plotly_white",
        font=dict(family="DM Sans", size=10),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    )
    return fig

# Regime detection
st.markdown("""
<div class="section-header" style="margin-top:32px;">
    <div class="section-tag">MODULE 02</div>
    <div class="section-title">Regime Detection Engine</div>
    <div class="section-subtitle">Volatility-clustering regime classification across the portfolio</div>
</div>
""", unsafe_allow_html=True)

regime_cols = st.columns(len(valid_tickers))
regime_color_map = {"Bull": "regime-bull", "Normal": "regime-normal", "Stress": "regime-stress", "Crisis": "regime-crisis"}

for i, tkr in enumerate(valid_tickers):
    r = returns_df[tkr].dropna()
    regimes = detect_regime(r)
    current_regime = regimes.iloc[-1] if not regimes.empty else "Unknown"
    rc = regime_color_map.get(current_regime, "regime-normal")
    ann_vol = r.std() * np.sqrt(252) * 100
    with regime_cols[i]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{tkr} — Regime</div>
            <div style="margin:10px 0;"><span class="regime-badge {rc}">● {current_regime}</span></div>
            <div class="kpi-sub">Ann. Vol: <strong>{ann_vol:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

# Regime timeline for first ticker
if len(valid_tickers) > 0:
    r0 = returns_df[valid_tickers[0]].dropna()
    regime_series = detect_regime(r0).dropna()
    reg_color_map = {"Bull": "#3B82F6", "Normal": "#059669", "Stress": "#D97706", "Crisis": "#DC2626"}
    regime_num = regime_series.map({"Bull": 0, "Normal": 1, "Stress": 2, "Crisis": 3})
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(
        x=regime_series.index, y=regime_num,
        mode="lines", fill="tozeroy",
        line=dict(width=0),
        fillcolor="rgba(26,86,219,0.15)",
        name="Regime Level"
    ))
    fig_regime.update_layout(
        title=f"Regime Timeline — {valid_tickers[0]}",
        yaxis=dict(tickvals=[0,1,2,3], ticktext=["Bull","Normal","Stress","Crisis"]),
        height=220, **PLOT_TEMPLATE["layout"]
    )
    st.plotly_chart(fig_regime, use_container_width=True)


# ─── CORRELATION ENGINE ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-tag">MODULE 03</div>
    <div class="section-title">Correlation Breakdown Engine</div>
    <div class="section-subtitle">Regime-switching correlation matrices and contagion simulation</div>
</div>
""", unsafe_allow_html=True)

if len(valid_tickers) >= 2:
    corr_raw = returns_df[valid_tickers].corr().values
    stressed_corr = stress_correlation_matrix(corr_raw, corr_regime)

    c1, c2 = st.columns(2)
    with c1:
        fig_corr1 = go.Figure(go.Heatmap(
            z=corr_raw, x=valid_tickers, y=valid_tickers,
            colorscale=[[0,"#DC2626"],[0.5,"#F3F4F6"],[1,"#1A56DB"]],
            zmin=-1, zmax=1,
            text=np.round(corr_raw,2), texttemplate="%{text}",
            colorbar=dict(len=0.8)
        ))
        fig_corr1.update_layout(title="Historical Correlation", height=360, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_corr1, use_container_width=True)

    with c2:
        fig_corr2 = go.Figure(go.Heatmap(
            z=stressed_corr, x=valid_tickers, y=valid_tickers,
            colorscale=[[0,"#DC2626"],[0.5,"#F3F4F6"],[1,"#1A56DB"]],
            zmin=-1, zmax=1,
            text=np.round(stressed_corr,2), texttemplate="%{text}",
            colorbar=dict(len=0.8)
        ))
        fig_corr2.update_layout(title=f"Stressed Correlation ({corr_regime.replace('_',' ').title()})",
                                 height=360, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_corr2, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
        <div class="info-box-title">CORRELATION STRESS ANALYSIS — {corr_regime.upper()}</div>
        <div class="info-box-text">
        Under the <strong>{corr_regime.replace('_',' ')}</strong> regime, asset correlations shift significantly.
        The average absolute correlation change is <strong>{np.abs(stressed_corr - corr_raw).mean():.3f}</strong>.
        This contagion effect amplifies portfolio losses beyond what individual asset VaR would suggest.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Add 2+ tickers to visualize correlation breakdown.")


# ─── GENERATIVE ENGINE + MONTE CARLO ────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-tag">MODULE 04 + 05</div>
    <div class="section-title">Generative Scenario Engine & Monte Carlo Simulation</div>
    <div class="section-subtitle">VAE fat-tail modeling · Block bootstrap · Adversarial perturbations</div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div class="glass-panel" style="text-align:center;padding:48px 24px;">
        <div style="font-size:40px;margin-bottom:16px;">🎲</div>
        <div style="font-size:18px;font-weight:700;color:#0D1B2A;margin-bottom:8px;">Configure & Launch Simulation</div>
        <div style="font-size:14px;color:#718096;">Set your parameters in the sidebar and press <strong>Run Stress Test</strong> to begin adversarial simulation.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Compute portfolio returns ──────────────────────────────────────────
    with st.spinner("Generating scenarios and running simulations…"):
        # Portfolio weights
        n = len(valid_tickers)
        if weight_mode == "Custom" and custom_weights:
            wts = np.array([custom_weights.get(t, 1/n) for t in valid_tickers])
            total = wts.sum()
            wts = wts / total if total > 0 else np.ones(n)/n
        else:
            wts = np.ones(n) / n

        port_returns = (returns_df[valid_tickers] * wts).sum(axis=1).dropna().values

        # Generate scenarios
        half = n_simulations // 2
        if gen_model == "VAE (Fat-Tail)":
            raw_scenarios = vae_generate(port_returns, n_simulations, horizon_days, tail_amplify)
        elif gen_model == "Block Bootstrap":
            raw_scenarios = block_bootstrap(port_returns, n_simulations, horizon_days, block_size)
        else:  # Ensemble
            s1 = vae_generate(port_returns, half, horizon_days, tail_amplify)
            s2 = block_bootstrap(port_returns, n_simulations - half, horizon_days, block_size)
            raw_scenarios = np.vstack([s1, s2])

        # Adversarial perturbations
        adv_scenarios = apply_adversarial_perturbations(raw_scenarios, vol_shock, drift_shift, corr_crush)

        # Portfolio paths
        paths = simulate_portfolio_paths(adv_scenarios, initial_value=100.0)

        # Worst / best paths
        worst_paths, best_paths, median_path = find_worst_paths(paths)

        # Risk metrics
        metrics = compute_risk_metrics(paths, confidence)

    # ── KPI Bar ───────────────────────────────────────────────────────────
    kpi_data = [
        ("VALUE AT RISK", f"{metrics['var']*100:.2f}%", f"{int(confidence*100)}% confidence", "#DC2626"),
        ("CVaR / ES",      f"{metrics['cvar']*100:.2f}%", "Expected shortfall", "#DC2626"),
        ("MAX DRAWDOWN",   f"{metrics['worst_dd']*100:.2f}%", "Worst path", "#D97706"),
        ("TAIL RATIO",     f"{metrics['tail_ratio']:.3f}", "Upside/Downside", "#1A56DB"),
        ("EXCESS KURTOSIS",f"{metrics['kurtosis']:.2f}", "Fat-tail indicator", "#D97706"),
        ("SKEWNESS",       f"{metrics['skewness']:.3f}", "Return asymmetry", "#1A56DB"),
        ("P5 VALUE",       f"${metrics['p5']:.1f}", "5th percentile portfolio", "#DC2626"),
        ("P95 VALUE",      f"${metrics['p95']:.1f}", "95th percentile portfolio", "#059669"),
    ]
    
    cols = st.columns(len(kpi_data))
    for i, (label, val, sub, color) in enumerate(kpi_data):
        with cols[i]:
            st.markdown(f"""
            <div style="background:{color}; color: white; padding: 16px; border-radius: 12px; text-align: center;">
                <div style="font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">{label}</div>
                <div style="font-size: 26px; font-weight: 700; margin: 8px 0;">{val}</div>
                <div style="font-size: 12px; opacity: 0.8;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Simulation Tabs ───────────────────────────────────────────────────
    sim_tab1, sim_tab2, sim_tab3, sim_tab4, sim_tab5, sim_tab6 = st.tabs([
        "🌐 Path Simulation", "🔥 Tail Distribution", "📉 Drawdown Analysis", "📦 3D Surface",
        "🥧 Asset Allocation", "📊 Monte Carlo Stats"
    ])

    with sim_tab1:
        fig_paths = go.Figure()
        # Fan: sample 200 random paths lightly
        sample_idx = np.random.choice(paths.shape[0], min(200, paths.shape[0]), replace=False)
        for idx in sample_idx:
            fig_paths.add_trace(go.Scatter(
                x=list(range(horizon_days)), y=paths[idx],
                mode="lines", line=dict(width=0.4, color="rgba(26,86,219,0.08)"),
                showlegend=False, hoverinfo="skip"
            ))
        # Percentile bands
        p5  = np.percentile(paths, 5,  axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        x   = list(range(horizon_days))

        fig_paths.add_trace(go.Scatter(x=x+x[::-1], y=list(p95)+list(p5[::-1]),
            fill="toself", fillcolor="rgba(26,86,219,0.05)", line=dict(width=0),
            name="5–95% band", showlegend=True))
        fig_paths.add_trace(go.Scatter(x=x+x[::-1], y=list(p75)+list(p25[::-1]),
            fill="toself", fillcolor="rgba(26,86,219,0.10)", line=dict(width=0),
            name="25–75% band", showlegend=True))

        fig_paths.add_trace(go.Scatter(x=x, y=p50, name="Median",
            line=dict(color=COLORS["blue"], width=2.5)))
        for wp in worst_paths[:3]:
            fig_paths.add_trace(go.Scatter(x=x, y=wp, name="Worst path",
                line=dict(color=COLORS["red"], width=1.5, dash="dot"), showlegend=False))
        for bp in best_paths[:2]:
            fig_paths.add_trace(go.Scatter(x=x, y=bp, name="Best path",
                line=dict(color=COLORS["green"], width=1.5, dash="dot"), showlegend=False))

        fig_paths.update_layout(
            title=f"Portfolio Simulation — {n_simulations:,} Paths | {horizon_days}d Horizon",
            xaxis_title="Trading Days", yaxis_title="Portfolio Value (Base=100)",
            height=480, **PLOT_TEMPLATE["layout"]
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # Adversarial worst path highlight
        worst_single = worst_paths[0]
        worst_drawdown = (worst_single - worst_single.max()) / worst_single.max()
        st.markdown(f"""
        <div class="warn-box">
            ⚡ <strong>Adversarial Worst Path:</strong> This path reaches a maximum loss of
            <strong>{worst_single[-1]-100:.1f}%</strong> by day {horizon_days},
            with a peak drawdown of <strong>{worst_drawdown.min()*100:.1f}%</strong>.
            Tail amplification factor: <strong>{tail_amplify}×</strong>.
        </div>
        """, unsafe_allow_html=True)

    with sim_tab2:
        final_vals = paths[:, -1]
        returns_final = (final_vals - 100) / 100

        fig_tail = go.Figure()
        fig_tail.add_trace(go.Histogram(
            x=returns_final, nbinsx=120, name="Simulated Returns",
            marker_color=COLORS["blue"], opacity=0.6,
            histnorm="probability density"
        ))
        # Normal overlay
        mu_s = returns_final.mean()
        sg_s = returns_final.std()
        x_norm = np.linspace(returns_final.min(), returns_final.max(), 300)
        y_norm = norm.pdf(x_norm, mu_s, sg_s)
        fig_tail.add_trace(go.Scatter(x=x_norm, y=y_norm, name="Normal Dist.",
            line=dict(color=COLORS["gray"], width=2, dash="dash")))
        # VaR / CVaR lines
        fig_tail.add_vline(x=metrics["var"], line_color=COLORS["amber"],
            line_dash="dash", annotation_text=f"VaR {int(confidence*100)}%",
            annotation_position="top right")
        fig_tail.add_vline(x=metrics["cvar"], line_color=COLORS["red"],
            line_dash="dot", annotation_text="CVaR",
            annotation_position="top right")

        fig_tail.update_layout(
            title="Return Distribution — Tail Risk Analysis",
            xaxis_title="Portfolio Return", yaxis_title="Density",
            height=420, **PLOT_TEMPLATE["layout"]
        )
        st.plotly_chart(fig_tail, use_container_width=True)

        # QQ plot
        fig_qq = go.Figure()
        sorted_r = np.sort(returns_final)
        theoretical_q = norm.ppf(np.linspace(0.01, 0.99, len(sorted_r)))
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sorted_r[:len(theoretical_q)],
            mode="markers", marker=dict(size=3, color=COLORS["blue"], opacity=0.4),
            name="Empirical vs Normal"))
        fig_qq.add_trace(go.Scatter(x=[-4,4], y=[-4*sg_s+mu_s, 4*sg_s+mu_s],
            mode="lines", line=dict(color=COLORS["red"], width=2), name="Normal Line"))
        fig_qq.update_layout(title="QQ Plot — Tail Departure from Normality",
            xaxis_title="Theoretical Quantiles", yaxis_title="Empirical Quantiles",
            height=360, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_qq, use_container_width=True)

    with sim_tab3:
        # Drawdown waterfall for worst path
        worst_p = worst_paths[0]
        roll_max = np.maximum.accumulate(worst_p)
        drawdown = (worst_p - roll_max) / roll_max * 100
        x = list(range(horizon_days))

        fig_dd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4],
                                subplot_titles=["Adversarial Worst Path", "Drawdown (%)"])
        fig_dd.add_trace(go.Scatter(x=x, y=worst_p, name="Portfolio Value",
            line=dict(color=COLORS["navy"], width=2)), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=x, y=list(roll_max), name="Peak",
            line=dict(color=COLORS["blue"], width=1.5, dash="dot")), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=x, y=drawdown, name="Drawdown",
            fill="tozeroy", fillcolor="rgba(220,38,38,0.15)",
            line=dict(color=COLORS["red"], width=1.5)), row=2, col=1)

        fig_dd.update_layout(height=480, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_dd, use_container_width=True)

        # Drawdown histogram across all paths
        peak_all = np.maximum.accumulate(paths, axis=1)
        dd_all   = ((paths - peak_all) / peak_all).min(axis=1) * 100
        fig_dd_hist = go.Figure()
        fig_dd_hist.add_trace(go.Histogram(x=dd_all, nbinsx=80,
            marker_color=COLORS["red"], opacity=0.7, name="Max Drawdown Dist."))
        fig_dd_hist.add_vline(x=dd_all.mean(), line_color=COLORS["navy"],
            annotation_text=f"Avg: {dd_all.mean():.1f}%")
        fig_dd_hist.update_layout(title="Maximum Drawdown Distribution Across All Paths",
            xaxis_title="Max Drawdown (%)", yaxis_title="Count",
            height=300, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_dd_hist, use_container_width=True)

    with sim_tab4:
        # 3D surface: Time × Percentile × Portfolio Value
        n_time   = min(horizon_days, 60)
        t_idx    = np.linspace(0, horizon_days-1, n_time, dtype=int)
        pct_range = np.arange(1, 100, 2)
        Z = np.array([[np.percentile(paths[:, t], p) for t in t_idx] for p in pct_range])

        fig_3d = go.Figure(go.Surface(
            x=t_idx, y=pct_range, z=Z,
            colorscale=[
                [0.0, "#DC2626"], [0.25, "#D97706"],
                [0.5, "#F3F4F6"], [0.75, "#1A56DB"], [1.0, "#059669"]
            ],
            contours=dict(
                x=dict(show=True, color="rgba(0,0,0,0.1)"),
                y=dict(show=True, color="rgba(0,0,0,0.1)")
            ),
            opacity=0.92
        ))
        fig_3d.update_layout(
            title="3D Risk Surface — Time × Percentile × Portfolio Value",
            scene=dict(
                xaxis_title="Day",
                yaxis_title="Percentile",
                zaxis_title="Portfolio Value",
                bgcolor="rgba(248,250,255,0.8)",
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                zaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            ),
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=11),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with sim_tab5:
        vis_wts_arr = vis_wts * 100
        fig_pie = go.Figure(data=[go.Pie(
            labels=valid_tickers,
            values=vis_wts_arr,
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=palette[:len(valid_tickers)])
        )])
        fig_pie.update_layout(title="Current Asset Allocation", height=400, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
        
        st.markdown("**Sector/Asset Breakdown:**")
        sector_data = []
        for t, w in zip(valid_tickers, vis_wts_arr):
            sector_data.append({"Asset": t, "Weight": f"{w:.1f}%"})
        st.dataframe(pd.DataFrame(sector_data), use_container_width=True)
        
        fig_donut = go.Figure()
        fig_donut.add_trace(go.Pie(
            labels=valid_tickers,
            values=vis_wts_arr,
            hole=0.65,
            direction='clockwise',
            sort=False,
            marker=dict(colors=palette[:len(valid_tickers)])
        ))
        fig_donut.update_layout(title="Donut Chart — Portfolio Composition", height=350, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_donut, use_container_width=True, key="donut_chart")
        
        fig_donut = go.Figure()
        fig_donut.add_trace(go.Pie(
            labels=valid_tickers,
            values=vis_wts_arr,
            hole=0.65,
            direction='clockwise',
            sort=False,
            marker=dict(colors=palette[:len(valid_tickers)])
        ))
        fig_pie.update_layout(title="Current Asset Allocation", height=400, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("**Sector/Asset Breakdown:**")
        sector_data = []
        for t, w in zip(valid_tickers, vis_wts_arr):
            sector_data.append({"Asset": t, "Weight": f"{w:.1f}%"})
        st.dataframe(pd.DataFrame(sector_data), use_container_width=True)
        
        fig_donut = go.Figure()
        fig_donut.add_trace(go.Pie(
            labels=valid_tickers,
            values=vis_wts_arr,
            hole=0.65,
            direction='clockwise',
            sort=False,
            marker=dict(colors=palette[:len(valid_tickers)])
        ))
        fig_donut.update_layout(title="Donut Chart — Portfolio Composition", height=350, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_donut, use_container_width=True)

    with sim_tab6:
        mu = port_returns.mean()
        sigma = port_returns.std()
        
        final_vals = paths[:, -1]
        returns_sim = (final_vals - 100) / 100
        
        fig_stats = make_subplots(rows=2, cols=2,
                                 subplot_titles=["Return Distribution", "Cumulative Distribution (CDF)",
                                                "Box Plot by Year", "Violin Plot"])
        
        fig_stats.add_trace(go.Histogram(x=returns_sim, nbinsx=50, 
                                        marker_color=COLORS["blue"], opacity=0.6), row=1, col=1)
        sorted_vals = np.sort(returns_sim)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        fig_stats.add_trace(go.Scatter(x=sorted_vals, y=cdf, mode='lines',
                                      line=dict(color=COLORS["navy"], width=2)), row=1, col=2)
        
        fig_stats.add_trace(go.Box(y=returns_sim, name="Returns",
                                  marker_color=COLORS["teal"]), row=2, col=1)
        fig_stats.add_trace(go.Violin(y=returns_sim, name="Returns",
                                     box_visible=True, meanline_visible=True,
                                     marker_color=COLORS["purple"]), row=2, col=2)
        
        fig_stats.update_layout(height=600, showlegend=False, **PLOT_TEMPLATE["layout"])
        st.plotly_chart(fig_stats, use_container_width=True)
        
        mc_stats = {
            "Mean Return": f"{returns_sim.mean()*100:.2f}%",
            "Std Dev": f"{returns_sim.std()*100:.2f}%",
            "Skewness": f"{skew(returns_sim):.3f}",
            "Kurtosis": f"{kurtosis(returns_sim):.3f}",
            "Sharpe (sim)": f"{(returns_sim.mean() / returns_sim.std()):.3f}",
            "Best Case": f"{returns_sim.max()*100:.2f}%",
            "Worst Case": f"{returns_sim.min()*100:.2f}%"
        }
        
        st.markdown('<h3 style="color:#0D1B2A; margin-bottom:16px;">📊 Monte Carlo Statistics</h3>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
        mc_items = list(mc_stats.items())
        for idx, (k, v) in enumerate(mc_items):
            with [mc1, mc2, mc3, mc4, mc5, mc6, mc7][idx]:
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #E2E8F0; border-radius:8px; padding:12px; text-align:center; margin-bottom:8px;">
                    <div style="font-size:10px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#718096; margin-bottom:4px;">{k}</div>
                    <div style="font-size:18px; font-weight:700; color:#0D1B2A;">{v}</div>
                </div>
                """, unsafe_allow_html=True)


    # ─── RISK METRICS TABLE ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag">MODULE 06</div>
        <div class="section-title">Risk & Tail Analytics</div>
        <div class="section-subtitle">Professional-grade metric suite with statistical decomposition</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("**📐 Core Risk Metrics**")
        metric_table_html = """
        <table class="styled-table">
        <thead><tr><th>Metric</th><th>Value</th><th>Signal</th></tr></thead>
        <tbody>
        """
        def signal(val, thresholds, labels, is_pct=True):
            formatted = f"{val*100:.2f}%" if is_pct else f"{val:.4f}"
            for threshold, label in zip(thresholds, labels):
                if val < threshold:
                    color = {"🔴 High Risk": "#DC2626", "🟡 Moderate": "#D97706", "🟢 Low": "#059669"}.get(label, "#718096")
                    return formatted, f'<span style="color:{color};font-size:11px;font-weight:600;">{label}</span>'
            return formatted, '<span style="color:#059669;font-size:11px;">🟢 Low</span>'

        rows = [
            ("Value at Risk (VaR)", metrics["var"], [-0.10, -0.05, 0], [-.10,-.05], ["🔴 High Risk","🟡 Moderate","🟢 Low"]),
            ("CVaR / Expected Shortfall", metrics["cvar"], [-0.15, -0.08, 0], [-.15,-.08], ["🔴 Severe","🟡 Elevated","🟢 Normal"]),
            ("Avg Max Drawdown", metrics["max_dd_avg"], [-0.20, -0.10, 0], [-.20,-.10], ["🔴 Severe","🟡 Moderate","🟢 Mild"]),
            ("Worst Drawdown", metrics["worst_dd"], [-0.30, -0.15, 0], [-.30,-.15], ["🔴 Catastrophic","🟡 Severe","🟢 Moderate"]),
        ]
        for label, val, _, thresholds, labels in rows:
            fmt_val = f"{val*100:.2f}%"
            metric_table_html += f"<tr><td>{label}</td><td><strong>{fmt_val}</strong></td><td>"
            if val < thresholds[0]:
                metric_table_html += f'<span style="color:#DC2626;font-size:11px;font-weight:600;">{labels[0]}</span>'
            elif val < thresholds[1]:
                metric_table_html += f'<span style="color:#D97706;font-size:11px;font-weight:600;">{labels[1]}</span>'
            else:
                metric_table_html += f'<span style="color:#059669;font-size:11px;font-weight:600;">{labels[2]}</span>'
            metric_table_html += "</td></tr>"

        metric_table_html += "</tbody></table>"
        st.markdown(metric_table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("**📊 Distribution Statistics**")
        stat_table_html = """
        <table class="styled-table">
        <thead><tr><th>Statistic</th><th>Value</th><th>Interpretation</th></tr></thead>
        <tbody>
        """
        stat_rows = [
            ("Skewness",         f"{metrics['skewness']:.4f}",   "Negative → left-tail risk"),
            ("Excess Kurtosis",  f"{metrics['kurtosis']:.4f}",   ">3 → fat-tailed distribution"),
            ("Mean Return",      f"{metrics['mean_return']*100:.3f}%", "Expected path outcome"),
            ("Std Deviation",    f"{metrics['std_return']*100:.3f}%", "Return dispersion"),
            ("Tail Ratio",       f"{metrics['tail_ratio']:.4f}", "Upside vs downside asymmetry"),
            ("P5 Portfolio",     f"${metrics['p5']:.2f}", "5th percentile outcome"),
            ("P50 Portfolio",    f"${metrics['p50']:.2f}", "Median outcome"),
            ("P95 Portfolio",    f"${metrics['p95']:.2f}", "95th percentile outcome"),
        ]
        for label, val, interp in stat_rows:
            stat_table_html += f"<tr><td>{label}</td><td><strong>{val}</strong></td><td style='color:#718096;font-size:11px;'>{interp}</td></tr>"
        stat_table_html += "</tbody></table>"
        st.markdown(stat_table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Per-Asset Risk ──────────────────────────────────────────────────────
    with st.expander("📋 Per-Asset Risk Breakdown"):
        asset_metrics = []
        for tkr in valid_tickers:
            r = returns_df[tkr].dropna().values
            ann_vol = np.std(r) * np.sqrt(252)
            var_95  = np.percentile(r, 5)
            mdd     = (prices[tkr] / prices[tkr].cummax() - 1).min()
            asset_metrics.append({
                "Ticker": tkr,
                "Ann. Vol": f"{ann_vol*100:.2f}%",
                "Daily VaR (95%)": f"{var_95*100:.3f}%",
                "Historical MDD": f"{mdd*100:.2f}%",
                "Skew": f"{skew(r):.3f}",
                "Kurt": f"{kurtosis(r):.3f}",
                "Weight": f"{wts[valid_tickers.index(tkr)]*100:.1f}%" if tkr in valid_tickers else "—"
            })
        st.dataframe(pd.DataFrame(asset_metrics), use_container_width=True, hide_index=True)


    # ─── SCENARIO COMPARISON ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag">MODULE 07</div>
        <div class="section-title">Scenario Comparison Dashboard</div>
        <div class="section-subtitle">Normal vs stressed vs adversarial path comparison</div>
    </div>
    """, unsafe_allow_html=True)

    # Re-run clean (no adversarial) for comparison
    clean_scenarios = vae_generate(port_returns, 2000, horizon_days, 1.0)
    clean_paths = simulate_portfolio_paths(clean_scenarios)

    fig_cmp = go.Figure()
    for label, p_arr, color in [
        ("Normal Scenario (P50)", np.percentile(clean_paths, 50, axis=0), COLORS["green"]),
        ("Normal Scenario (P5)", np.percentile(clean_paths, 5, axis=0), COLORS["teal"]),
        ("Adversarial (P50)", np.percentile(paths, 50, axis=0), COLORS["blue"]),
        ("Adversarial (P5)", np.percentile(paths, 5, axis=0), COLORS["amber"]),
        ("Adversarial (P1)", np.percentile(paths, 1, axis=0), COLORS["red"]),
    ]:
        fig_cmp.add_trace(go.Scatter(
            x=list(range(horizon_days)), y=p_arr, name=label,
            line=dict(color=color, width=2)
        ))
    fig_cmp.update_layout(
        title="Scenario Comparison: Normal vs Adversarial Stress",
        xaxis_title="Days", yaxis_title="Portfolio Value",
        height=420, **PLOT_TEMPLATE["layout"]
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


    # ─── EXPORT ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag">MODULE 08</div>
        <div class="section-title">Export & Reporting</div>
        <div class="section-subtitle">Download simulation results and risk metrics</div>
    </div>
    """, unsafe_allow_html=True)

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        # Export paths summary
        pct_df = pd.DataFrame({
            "Day": range(horizon_days),
            "P1":  np.percentile(paths, 1,  axis=0),
            "P5":  np.percentile(paths, 5,  axis=0),
            "P25": np.percentile(paths, 25, axis=0),
            "P50": np.percentile(paths, 50, axis=0),
            "P75": np.percentile(paths, 75, axis=0),
            "P95": np.percentile(paths, 95, axis=0),
            "P99": np.percentile(paths, 99, axis=0),
        })
        csv_paths = pct_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Path Percentiles (CSV)",
            data=csv_paths,
            file_name=f"aegis_paths_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with export_col2:
        # Export risk metrics
        metrics_df = pd.DataFrame([{
            "Metric": k, "Value": v
        } for k, v in metrics.items()])
        csv_metrics = metrics_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Risk Metrics (CSV)",
            data=csv_metrics,
            file_name=f"aegis_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">SIMULATION SUMMARY</div>
        <div class="info-box-text">
        Simulation complete. <strong>%d paths</strong> generated over <strong>%d trading days</strong>
        using <strong>%s</strong> with adversarial perturbations applied.
        Correlation regime: <strong>%s</strong>. Tail amplification: <strong>%.1f×</strong>.
        </div>
    </div>
    """ % (n_simulations, horizon_days, gen_model, corr_regime, tail_amplify), unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="aegis-footer">
    <div class="aegis-footer-brand">🛡️ Aegis Risk Lab</div>
    <div style="font-size:12px;color:#718096;margin-bottom:8px;">
        Adversarial Stress Testing &amp; Tail Risk Intelligence Platform
    </div>
    <div style="width:60px;height:1px;background:linear-gradient(90deg,transparent,#CBD5E0,transparent);margin:12px auto;"></div>
    <div class="aegis-footer-made">Made by <strong>Sourish Dey</strong></div>
    <div style="font-size:11px;color:#A0AEC0;margin-top:6px;">
        Powered by Yahoo Finance · NumPy · Plotly · Streamlit &nbsp;|&nbsp; {datetime.now().year}
    </div>
</div>
""", unsafe_allow_html=True)
