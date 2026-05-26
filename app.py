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

# 2. Asset Universe
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

# 3. Sidebar Controls
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

# 4. Pure Python Data & Math Processing (UPDATED)
@st.cache_data(ttl=86400)
def fetch_live_data(tickers_list, years):
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    try:
        # group=True helps handle multi-ticker cleaner on cloud servers
        data = yf.download(tickers_list, start=start, end=end, group_by='ticker')
        
        # Extract just the Close prices safely
        close_df = pd.DataFrame()
        for t in tickers_list:
            if t in data:
                close_df[t] = data[t]['Close']
            else:
                # Fallback if ticker structure is flat
                close_df = data['Close']
                break
                
        return close_df[tickers_list].ffill().dropna()
    except Exception as e:
        st.sidebar.error(f"Data Fetch Diagnostic: {str(e)}")
        return pd.DataFrame()

prices = fetch_live_data(tickers, lookback_years)

if not prices.empty:
    daily_returns = prices.pct_change().dropna()
    annualized_returns = daily_returns.mean() * 252
    covariance_matrix = daily_returns.cov() * 252

    def calc_stats(w_vec):
        p_ret = np.sum(annualized_returns * w_vec)
        p_vol = np.sqrt(np.dot(w_vec.T, np.dot(covariance_matrix, w_vec)))
        return p_ret, p_vol

    if not sandbox_mode:
        def objective_minimize_sharpe(w):
            r, v = calc_stats(w)
            return -(r - risk_free_rate) / v

        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
        bnds = tuple((0.0, max_allocation) for _ in range(len(tickers)))
        res = minimize(
            objective_minimize_sharpe,
            [1.0 / len(tickers)] * len(tickers),
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
        )
        final_weights = res.x
    else:
        final_weights = np.array([user_weights[t] for t in tickers])
        if not np.isclose(sum(final_weights), 1.0):
            st.sidebar.error("⚠️ Mix adjustments must total exactly 100%.")

    opt_series = pd.Series(final_weights, index=tickers)
    
    summary_df = pd.DataFrame({
        "Fund Name": pd.Series(FUNDS),
        "Annual Return": annualized_returns,
        "Volatility (Risk)": daily_returns.std() * np.sqrt(252),
        "Target Allocation": opt_series
    })
    
    summary_df["Annual Return"] = summary_df["Annual Return"] * 100
    summary_df["Volatility (Risk)"] = summary_df["Volatility (Risk)"] * 100
    summary_df["Target Allocation"] = summary_df["Target Allocation"] * 100

    summary_df = summary_df.sort_values(by="Target Allocation", ascending=False)
    top_funds = summary_df[summary_df["Target Allocation"] > 0.001]

    # 5. Target Mix Cards
    if sandbox_mode:
        p_ret_print, p_vol_print = calc_stats(final_weights)
        st.markdown("### Portfolio Metrics")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Expected Annual Return", f"{p_ret_print*100:.2f}%")
        m_col2.metric("Portfolio Volatility (Risk)", f"{p_vol_print*100:.2f}%")
    else:
        st.markdown("### Recommended Target Mix")
        num_recommended_funds = len(top_funds)
        
        if num_recommended_funds == 0:
            st.info("No funds selected. Adjust constraints to generate allocation profiles.")
        elif num_recommended_funds <= 5:
            cols = st.columns(num_recommended_funds)
            for idx, (ticker, row) in enumerate(top_funds.iterrows()):
                with cols[idx]:
                    st.metric(label=f"Allocate to {ticker}", value=f"{row['Target Allocation']:.1f}%")
        else:
            grid_cols = st.columns(4)
            for idx, (ticker, row) in enumerate(top_funds.iterrows()):
                col_target = grid_cols[idx % 4]
                with col_target:
                    st.metric(label=f"Allocate to {ticker}", value=f"{row['Target Allocation']:.1f}%")
                    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # 6. Visualizations
    l_col1, l_col2 = st.columns([4, 6])
    with l_col1:
        st.markdown("### Distribution")
        if len(top_funds) > 0:
            fig_donut = px.pie(
                top_funds.reset_index(),
                values="Target Allocation",
                names="index",
                hole=0.6,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1d1d1f",
                showlegend=True,
                legend=dict(orientation="h", y=-0.15)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Assign allocations inside the sidebar panel.")

    with l_col2:
        st.markdown("### Risk vs. Return Space")
        fig_scatter = px.scatter(
            summary_df.reset_index(),
            x="Volatility (Risk)",
            y="Annual Return",
            text="index",
            color="Target Allocation",
            color_continuous_scale=px.colors.sequential.Blues,
            labels={
                "Volatility (Risk)": "Annual Volatility (Risk %)",
                "Annual Return": "Expected Return (%)",
            },
        )
        fig_scatter.update_traces(
            marker=dict(size=12, line=dict(width=1, color="#ffffff")),
            textposition="top center",
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1d1d1f",
            xaxis=dict(gridcolor="#e5e5e7", zeroline=False),
            yaxis=dict(gridcolor="#e5e5e7", zeroline=False),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 7. Fund Table
    st.markdown("### Performance Overview")
    st.dataframe(
        summary_df.style.format({
            "Annual Return": "{:.2f}%",
            "Volatility (Risk)": "{:.2f}%",
            "Target Allocation": "{:.2f}%",
        }),
        use_container_width=True,
    )
else:
    st.error("Data Feed Offline: Unable to download current market tickers. Try shifting your historical timeline window in the sidebar to reset the stream connection.")
