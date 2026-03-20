import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

COMPANIES = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
}

COLORS = ["#1a4480", "#c0392b", "#1a6640", "#7a5c00", "#5a3472"]

LIGHT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafaf8",
    font=dict(family="IBM Plex Sans", color="#1c1c1e", size=11),
)


def get_series(df, col_name):
    """Extract a clean 1-D Series — handles yfinance multi-level columns."""
    col = df[col_name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return pd.Series(col.values, index=df.index, name=col_name)


def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    dfs = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Close', 'Volume', 'High', 'Low', 'Open']].copy()
                df.index = pd.to_datetime(df.index)
                dfs[name] = df
        except Exception as e:
            st.warning(f"Could not fetch {name}: {e}")
    return dfs


def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Task 1 — Data Collection & Preparation</div>
        <div class="page-sub">Fetch real financial time series · Align · Normalize</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rule">Configuration</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected = st.multiselect(
            "Select Companies (minimum 3)",
            list(COMPANIES.keys()),
            default=["Reliance Industries", "TCS", "Infosys"]
        )
    with col2:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col3:
        end_date = st.date_input("End Date", value=datetime(2024, 12, 31))

    if len(selected) < 3:
        st.warning("Select at least 3 companies.")
        return

    if st.button("Fetch Data"):
        with st.spinner("Downloading from Yahoo Finance…"):
            tickers = {c: COMPANIES[c] for c in selected}
            data = fetch_data(tickers, str(start_date), str(end_date))
        st.session_state["stock_data"] = data
        st.session_state["selected_companies"] = selected

    if "stock_data" not in st.session_state:
        st.markdown("""
        <div class="note">
        Configure settings above and click <b>Fetch Data</b>.<br>
        <b>Sample:</b> Reliance Industries, TCS, Infosys · 2020-01-01 to 2024-12-31
        </div>
        """, unsafe_allow_html=True)
        return

    data = st.session_state["stock_data"]
    if not data:
        st.error("No data returned. Check your internet connection.")
        return

    st.success(f"Data loaded for {len(data)} companies — {len(list(data.values())[0])} trading days")

    # ── Summary Stats ──────────────────────────────────────────────────────────
    st.markdown('<div class="rule">Summary Statistics</div>', unsafe_allow_html=True)
    cols = st.columns(len(data))
    for i, (name, df) in enumerate(data.items()):
        close = get_series(df, 'Close')
        latest = float(close.iloc[-1])
        first  = float(close.iloc[0])
        ret    = (latest - first) / first * 100
        sign   = "+" if ret >= 0 else ""
        with cols[i]:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{name}</div>
                <div class="card-value">{latest:,.1f}</div>
                <div class="card-sub">{sign}{ret:.1f}% total return · {len(df)} days</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Time Series Plot ───────────────────────────────────────────────────────
    st.markdown('<div class="rule">Time Series Plot — Normalized Close Prices</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for i, (name, df) in enumerate(data.items()):
        close = get_series(df, 'Close')
        fig.add_trace(go.Scatter(
            x=df.index, y=normalize(close), name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=1.6),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Norm: %{{y:.3f}}<extra></extra>"
        ))
    fig.update_layout(
        **LIGHT, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#eeece8", title="Date", showline=True, linecolor="#cdc8c0"),
        yaxis=dict(gridcolor="#eeece8", title="Normalised Price [0,1]", showline=True, linecolor="#cdc8c0"),
        margin=dict(l=55, r=20, t=40, b=50), hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Preview ───────────────────────────────────────────────────────
    st.markdown('<div class="rule">Raw Data Preview</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(data.keys()))
    for tab, (name, df) in zip(tabs, data.items()):
        with tab:
            ca, cb = st.columns([3, 1])
            with ca:
                disp = df.tail(20).copy()
                if isinstance(disp.columns, pd.MultiIndex):
                    disp.columns = disp.columns.get_level_values(0)
                st.dataframe(disp.round(2), use_container_width=True, height=260)
            with cb:
                vol = get_series(df, 'Volume')
                vfig = go.Figure(go.Histogram(
                    x=vol.values, nbinsx=25,
                    marker_color="#1a4480", opacity=0.75
                ))
                vfig.update_layout(
                    **LIGHT, height=260,
                    margin=dict(l=30, r=10, t=20, b=30), showlegend=False,
                    xaxis=dict(gridcolor="#eeece8", title="Volume"),
                    yaxis=dict(gridcolor="#eeece8")
                )
                st.plotly_chart(vfig, use_container_width=True)

    # ── Correlation Matrix ─────────────────────────────────────────────────────
    st.markdown('<div class="rule">Correlation Matrix — Close Prices</div>', unsafe_allow_html=True)
    try:
        series_list = []
        for name, df in data.items():
            s = get_series(df, 'Close')
            s.name = name
            series_list.append(s)
        closes = pd.concat(series_list, axis=1).dropna()
        corr = closes.corr()
        hfig = go.Figure(go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=[[0, "#ffffff"], [0.5, "#aec2e0"], [1, "#1a4480"]],
            text=corr.values.round(2),
            texttemplate="%{text}",
            showscale=True, zmin=-1, zmax=1,
            colorbar=dict(thickness=12, tickfont=dict(size=10))
        ))
        hfig.update_layout(
            **LIGHT, height=340,
            margin=dict(l=90, r=20, t=30, b=90),
            xaxis=dict(tickangle=-30),
            yaxis=dict(tickangle=0)
        )
        st.plotly_chart(hfig, use_container_width=True)
    except Exception as e:
        st.warning(f"Correlation matrix skipped: {e}")
