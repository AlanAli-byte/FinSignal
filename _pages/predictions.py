import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LIGHT = dict(
    template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#fafaf8",
    font=dict(family="IBM Plex Sans", color="#1c1c1e", size=11),
)

def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predictions — Forecast vs Actual</div>
        <div class="page-sub">Test set evaluation · Residual analysis · Future forecast</div>
    </div>
    """, unsafe_allow_html=True)

    if "train_results" not in st.session_state:
        st.markdown('<div class="note">Train the model on the <b>CNN Model</b> page first.</div>', unsafe_allow_html=True)
        return

    r = st.session_state["train_results"]
    y_test, y_pred = r["y_test"], r["y_pred"]
    residuals = y_pred - y_test
    n = len(y_test)

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.markdown('<div class="rule">Performance Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{r['mse']:.4f}")
    c2.metric("MAE", f"{r['mae']:.4f}")
    c3.metric("R²", f"{r['r2']:.4f}")
    c4.metric("Lookahead", f"{r['lookahead']} days")

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown('<div class="rule">Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Actual vs Predicted Price", "Residuals (Prediction Error)"],
        vertical_spacing=0.13, row_heights=[0.65, 0.35])

    fig.add_trace(go.Scatter(x=np.arange(n), y=y_test,
        name="Actual", line=dict(color="#1a4480", width=1.8),
        hovertemplate="Sample %{x}<br>Actual: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(n), y=y_pred,
        name="Predicted", line=dict(color="#c0392b", width=1.8, dash="dot"),
        hovertemplate="Sample %{x}<br>Predicted: %{y:.2f}<extra></extra>"), row=1, col=1)

    sigma = np.std(residuals)
    fig.add_trace(go.Scatter(
        x=np.concatenate([np.arange(n), np.arange(n)[::-1]]),
        y=np.concatenate([y_pred+sigma, (y_pred-sigma)[::-1]]),
        fill='toself', fillcolor='rgba(192,57,43,0.06)',
        line=dict(color='rgba(0,0,0,0)'), name='±1σ', hoverinfo='skip'), row=1, col=1)

    colors = ["#c0392b" if v < 0 else "#1a6640" for v in residuals]
    fig.add_trace(go.Bar(x=np.arange(n), y=residuals, marker_color=colors,
        opacity=0.7, name="Residual",
        hovertemplate="Sample %{x}<br>Error: %{y:.4f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#e2ddd6", row=2, col=1)

    fig.update_layout(**LIGHT, height=520, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60,r=20,t=50,b=50))
    for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor="#eeece8", showline=True, linecolor="#cdc8c0")})
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter + Histogram ────────────────────────────────────────────────────
    st.markdown('<div class="rule">Scatter & Error Distribution</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)

    with ca:
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        sfig = go.Figure()
        sfig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
            marker=dict(color="#1a4480", size=4.5, opacity=0.55,
                        line=dict(color="#ffffff", width=0.5)), name="Predictions",
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"))
        sfig.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
            line=dict(color="#1a6640", dash="dash", width=1.5), name="Ideal"))
        sfig.update_layout(**LIGHT, height=320,
            xaxis=dict(gridcolor="#eeece8", title="Actual", showline=True, linecolor="#cdc8c0"),
            yaxis=dict(gridcolor="#eeece8", title="Predicted", showline=True, linecolor="#cdc8c0"),
            margin=dict(l=60,r=20,t=20,b=50),
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(sfig, use_container_width=True)

    with cb:
        hfig = go.Figure()
        hfig.add_trace(go.Histogram(x=residuals, nbinsx=28,
            marker_color="#1a4480", opacity=0.75))
        hfig.add_vline(x=0, line_dash="dash", line_color="#cdc8c0")
        hfig.add_vline(x=np.mean(residuals), line_dash="dot", line_color="#c0392b",
                       annotation_text=f"μ={np.mean(residuals):.3f}",
                       annotation_font=dict(color="#c0392b", size=10))
        hfig.update_layout(**LIGHT, height=320,
            xaxis=dict(gridcolor="#eeece8", title="Prediction Error", showline=True, linecolor="#cdc8c0"),
            yaxis=dict(gridcolor="#eeece8", title="Count", showline=True, linecolor="#cdc8c0"),
            margin=dict(l=55,r=20,t=20,b=50), showlegend=False)
        st.plotly_chart(hfig, use_container_width=True)

        mc1, mc2 = st.columns(2)
        mc1.metric("Bias (Mean Error)", f"{np.mean(residuals):.4f}")
        mc2.metric("Error Std Dev", f"{np.std(residuals):.4f}")

    # ── Future Forecast ────────────────────────────────────────────────────────
    st.markdown('<div class="rule">Simulated Future Forecast</div>', unsafe_allow_html=True)
    data = st.session_state.get("stock_data", {})
    if data:
        company = st.selectbox("Company", list(data.keys()))
        future_days = st.slider("Forecast horizon (days)", 5, 60, 20)
        df = data[company]
        close = df['Close'].dropna()
        last_val = float(close.iloc[-1])
        noise_scale = float(close.pct_change().std()) * last_val
        np.random.seed(42)
        trend = np.linspace(0, last_val * 0.03, future_days)
        forecast = last_val + trend + np.cumsum(np.random.normal(0, noise_scale*0.3, future_days))
        future_dates = pd.date_range(close.index[-1], periods=future_days+1, freq='B')[1:]

        ffig = go.Figure()
        hist = close.iloc[-90:]
        ffig.add_trace(go.Scatter(x=hist.index, y=hist.values,
            name="Historical", line=dict(color="#1a4480", width=1.8)))
        ffig.add_trace(go.Scatter(
            x=list(future_dates)+list(future_dates[::-1]),
            y=list(forecast + noise_scale*1.5)+list((forecast - noise_scale*1.5)[::-1]),
            fill='toself', fillcolor='rgba(192,57,43,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
        ffig.add_trace(go.Scatter(x=future_dates, y=forecast,
            name="CNN Forecast", line=dict(color="#c0392b", width=2),
            marker=dict(size=4)))
        ffig.update_layout(**LIGHT, height=380,
            xaxis=dict(gridcolor="#eeece8", title="Date", showline=True, linecolor="#cdc8c0"),
            yaxis=dict(gridcolor="#eeece8", title=f"{company} Price", showline=True, linecolor="#cdc8c0"),
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=65,r=20,t=30,b=50))
        st.plotly_chart(ffig, use_container_width=True)
        st.markdown('<div style="font-size:0.78rem;color:#6b6760;font-family:IBM Plex Mono,monospace">Note: forecast is illustrative — real deployment would iteratively feed spectrogram patches to the trained CNN head.</div>', unsafe_allow_html=True)
