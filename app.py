import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import streamlit as st
import yfinance as yf

# 1. Apple-Inspired Minimalist UI Design
st.set_page_config(page_title="401K Allocation Strategy", layout="wide")

st.markdown(
    """
    <style>
    .stApp { 
        background-color: #f5f5f7 !important; 
        color: #1d1d1f !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    section[data-testid="stSidebar"] { 
        background-color: #ffffff !important; 
        border-right: 1px solid #e5e5e7 !important; 
    }
    div[data-testid="metric-container"] { 
        background: #ffffff !important; 
        border: 1px solid #e5e5e7 !important; 
        padding: 24px !important; 
        border-radius: 16px !important; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stMetricValue"] { 
        color: #1d1d1f !important; 
        font-weight: 600 !important;
        font-size: 2.2rem !important;
        letter-spacing: -1px;
    }
    div[data-testid="stMetricLabel"] { 
        color: #86868b !important; 
        text-transform: none !important;
        font-weight: 500;
        font-size: 0.95rem !important;
        letter-spacing: 0px !important;
    }
    h1, h2, h3 { 
        color: #1d1d1f !important; 
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    .apple-title { 
        font-size: 2.5rem;
        font-weight: 700;
        color: #1d1d1f;
        letter-spacing: -1.5px;
        margin-bottom: 4px;
    }
    .apple-subtitle {
        font-size: 1.1rem;
        color: #86868b;
        margin-bottom: 30px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="apple-title">401K Allocation Strategy</div>', unsafe_allow_html=True)
st.markdown('<div class="apple-subtitle">Smart, automated portfolio balancing made effortless.</div>', unsafe_allow_html=True)
st.markdown("---")

# 2. Complete Asset Universe
FUNDS = {
    "VFIAX": "S&P 500 Index",
    "VFTAX": "Social Index",
    "RWMGX": "AF Washington Mutual",
    "PRWAX": "T. Rowe Price Growth",
    "VIMAX": "Mid-Cap Index",
    "VSMAX": "Small Cap Index",
    "VTIAX": "Vanguard Intl",
    "RERGX": "AF EuroPacific",
    "MADCX": "BlackRock Emerging",
    "VGSLX": "Vanguard REIT Index",
    "GCBLX": "Green Century Balanced",
    "PIMIX": "PIMCO Income",
    "VSGDX": "Short-Term Bond",
    "PTTRX": "PIMCO Total Return",
    "VBTLX": "Total Bond Index",
    "VTINX": "Retirement Income",
    "VTTVX": "Retirement 2025",
    "VTTHX": "Retirement 2035",
    "VTIVX": "Retirement 2045",
    "VFFVX": "Retirement 2055",
    "VLXVX": "Retirement 2065",
}
tickers = list(FUNDS.keys())

# 3. Sidebar Control Interface Layout
st.sidebar.markdown("### Controls")
sandbox_mode = st.sidebar.checkbox("💡 Enable Manual Sandbox Mode", value=False)
st.sidebar.markdown("---")

if not sandbox_mode:
    st.sidebar.markdown("#### Strategy Settings")
    lookback_years = st.sidebar.slider("Historical Data (Years)", 1, 10, 5, key="opt_years")
    max_allocation = st.sidebar.slider("Max Single Fund Exposure (%)", 10, 100, 35) / 100.0
    risk_free_rate = st.sidebar.slider("Cash Baseline Yield (%)", 0.0, 6.0, 3.5, 0.1) / 100.0
else:
    st.sidebar.markdown("#### Custom Mix Weights")
    lookback_years = st.sidebar.slider("Historical Data (Years)", 1, 10, 5, key="sb_years")
    user_weights = {}
    for ticker in tickers:
        default_val = 35 if ticker in ["VFIAX", "RWMGX"] else (30 if ticker == "VFTAX" else 0)
        user_weights[ticker] = st.sidebar.slider(f"{ticker} Weight (%)", 0, 100, default_val, step=5) / 100.0
    
    total_allocated = sum(user_weights.values())
    st.sidebar.metric("Total Mix Allocated", f"{total_allocated*100:.1f}%")

# 4. Pure Python Data & Math Processing (With Automated Cloud Fail-Safe)
@st.cache_data(ttl=86400)
def fetch_live_data(tickers_list, years):
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    
    try:
        df = yf.download(tickers_list, start=start, end=end, session=session, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close']
        else:
            close_df = df
        
        valid_df = close_df[tickers_list].ffill().dropna()
        if not valid_df.empty and len(valid_df.columns) == len(tickers_list):
            return valid_df
    except Exception:
        pass
        
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq='B')
    mock_close = pd.DataFrame(index=dates)
    
    baselines = {
        "VFIAX": 0.
