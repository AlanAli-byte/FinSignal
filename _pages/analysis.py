import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as scipy_signal
from scipy.fft import fft

LIGHT = dict(
    template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#fafaf8",
    font=dict(family="IBM Plex Sans", color="#1c1c1e", size=11),
)

def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Task 4 — Analysis & Evaluation</div>
        <div class="page-sub">MSE evaluation · Feature comparison · Signal insights · Report</div>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.get("stock_data", {})
    results = st.session_state.get("train_results", {})

    if not data:
        st.markdown('<div class="note">Complete <b>Data Collection</b> first.</div>', unsafe_allow_html=True)
        return

    tab1, tab2, tab3, tab4 = st.tabs(["MODEL EVALUATION", "FEATURE ANALYSIS", "SIGNAL INSIGHTS", "REPORT"])

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tab1:
        if not results:
            st.markdown('<div class="note">Train the model on the <b>CNN Model</b> page first.</div>', unsafe_allow_html=True)
        else:
            y_test = results["y_test"]; y_pred = results["y_pred"]
            mse = results["mse"]; mae = results["mae"]; r2 = results["r2"]
            rmse = np.sqrt(mse)
            mape = float(np.mean(np.abs((y_test-y_pred)/(np.abs(y_test)+1e-8)))*100)
            maxe = float(np.max(np.abs(y_test-y_pred)))

            st.markdown('<div class="rule">Evaluation Metrics</div>', unsafe_allow_html=True)
            metrics = [
                ("MSE", f"{mse:.4f}", "Mean Squared Error"),
                ("RMSE", f"{rmse:.4f}", "Root MSE — same units as price"),
                ("MAE", f"{mae:.4f}", "Mean Absolute Error"),
                ("MAPE", f"{mape:.2f}%", "Mean Abs % Error"),
                ("R²", f"{r2:.4f}", "Coefficient of determination"),
                ("Max Error", f"{maxe:.4f}", "Worst case error"),
            ]
            cols = st.columns(3)
            for i, (label, val, desc) in enumerate(metrics):
                with cols[i % 3]:
                    color = "#1a6640" if label == "R²" and float(val) > 0.7 else "#1a4480"
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-label">{label}</div>
                        <div class="card-value" style="color:{color}">{val}</div>
                        <div class="card-sub">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="rule">Cumulative Absolute Error</div>', unsafe_allow_html=True)
            cum = np.cumsum(np.abs(y_pred - y_test))
            cfig = go.Figure(go.Scatter(y=cum, fill='tozeroy',
                line=dict(color="#1a4480", width=1.8),
                fillcolor="rgba(26,68,128,0.07)"))
            cfig.update_layout(**LIGHT, height=260,
                xaxis=dict(gridcolor="#eeece8", title="Test Sample", showline=True, linecolor="#cdc8c0"),
                yaxis=dict(gridcolor="#eeece8", title="Cumulative |Error|", showline=True, linecolor="#cdc8c0"),
                margin=dict(l=60,r=20,t=20,b=45), showlegend=False)
            st.plotly_chart(cfig, use_container_width=True)

            st.markdown('<div class="rule">Training History</div>', unsafe_allow_html=True)
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(y=results["history_loss"], name="Train Loss",
                line=dict(color="#1a4480", width=1.8)))
            hfig.add_trace(go.Scatter(y=results["history_val_loss"], name="Val Loss",
                line=dict(color="#c0392b", width=1.8, dash="dot")))
            hfig.update_layout(**LIGHT, height=290,
                xaxis=dict(gridcolor="#eeece8", title="Epoch", showline=True, linecolor="#cdc8c0"),
                yaxis=dict(gridcolor="#eeece8", title="MSE Loss", showline=True, linecolor="#cdc8c0"),
                margin=dict(l=60,r=20,t=20,b=45),
                legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(hfig, use_container_width=True)

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="rule">Feature Analysis — Effect on Signal Properties</div>', unsafe_allow_html=True)
        company = st.selectbox("Company", list(data.keys()), key="fa_co")
        df = data[company]
        features = [c for c in ["Close","Volume","High","Low","Open"] if c in df.columns]

        stats = {}
        for feat in features:
            s = df[feat].dropna().values.astype(float)
            n = len(s); y = fft(s)
            freqs = np.fft.fftfreq(n, d=1); pos = freqs > 0
            amps = np.abs(y[pos]) * 2 / n
            stats[feat] = {
                "cv":  float(s.std() / (s.mean() + 1e-8)),
                "dom": float(freqs[pos][np.argmax(amps)]) if len(amps) else 0,
                "ent": float(-np.sum((amps/(amps.sum()+1e-8))*np.log(amps/(amps.sum()+1e-8)+1e-9))),
            }

        COLORS = ["#1a4480","#c0392b","#1a6640","#7a5c00","#5a3472"]
        feats = list(stats.keys())

        fig_feat = make_subplots(rows=1, cols=2,
            subplot_titles=["Coefficient of Variation", "Spectral Entropy"])
        for i, f in enumerate(feats):
            c = COLORS[i % len(COLORS)]
            fig_feat.add_trace(go.Bar(x=[f], y=[stats[f]["cv"]],
                marker_color=c, showlegend=False), row=1, col=1)
            fig_feat.add_trace(go.Bar(x=[f], y=[stats[f]["ent"]],
                marker_color=c, showlegend=False), row=1, col=2)
        fig_feat.update_layout(**LIGHT, height=300,
            margin=dict(l=50,r=20,t=50,b=40))
        for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
            fig_feat.update_layout(**{ax: dict(gridcolor="#eeece8", showline=True, linecolor="#cdc8c0")})
        st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown('<div class="rule">Dominant Frequency by Feature</div>', unsafe_allow_html=True)
        dfig = go.Figure(go.Bar(
            x=feats, y=[stats[f]["dom"] for f in feats],
            marker=dict(color=COLORS[:len(feats)]),
            text=[f"{stats[f]['dom']:.4f}" for f in feats], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=9, color="#6b6760")
        ))
        dfig.update_layout(**LIGHT, height=280,
            xaxis=dict(gridcolor="#eeece8", title="Feature", showline=True, linecolor="#cdc8c0"),
            yaxis=dict(gridcolor="#eeece8", title="Dominant Freq [cycles/day]", showline=True, linecolor="#cdc8c0"),
            margin=dict(l=60,r=20,t=20,b=40), showlegend=False)
        st.plotly_chart(dfig, use_container_width=True)

        st.markdown("""
        <div class="note">
        <b>Close price</b> typically has the lowest CV and clearest dominant frequency — best for trend prediction.<br>
        <b>Volume</b> has the highest spectral entropy — most random, harder to learn from.<br>
        Features with clear dominant frequencies contribute more useful patterns to the CNN.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="rule">Resolution Trade-off — Window Length L</div>', unsafe_allow_html=True)
        windows = [8, 16, 32, 64, 128]
        tfig = go.Figure()
        tfig.add_trace(go.Scatter(x=windows, y=[1/w for w in windows],
            name="Freq resolution ∝ 1/L", line=dict(color="#c0392b", width=1.8)))
        tfig.add_trace(go.Scatter(x=windows, y=[w/128 for w in windows],
            name="Time resolution ∝ L", line=dict(color="#1a4480", width=1.8, dash="dot")))
        tfig.update_layout(**LIGHT, height=300,
            xaxis=dict(gridcolor="#eeece8", title="Window Length L", showline=True, linecolor="#cdc8c0"),
            yaxis=dict(gridcolor="#eeece8", title="Relative Resolution", showline=True, linecolor="#cdc8c0"),
            margin=dict(l=60,r=20,t=20,b=50),
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(tfig, use_container_width=True)

        ca, cb = st.columns(2)
        with ca:
            st.markdown("""
            <div class="card">
                <div class="card-label">Large Window (L ↑)</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;font-weight:600;
                            color:#c0392b;margin:0.3rem 0">Better Frequency Resolution</div>
                <div class="card-sub">Reveals seasonal cycles and long-term patterns.<br>
                Loses ability to pinpoint <em>when</em> a frequency occurs.</div>
            </div>
            """, unsafe_allow_html=True)
        with cb:
            st.markdown("""
            <div class="card">
                <div class="card-label">Small Window (L ↓)</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;font-weight:600;
                            color:#1a4480;margin:0.3rem 0">Better Time Resolution</div>
                <div class="card-sub">Tracks fast events like earnings or crashes.<br>
                Coarser frequency bins, loses spectral precision.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="rule">Spectral Interpretation</div>', unsafe_allow_html=True)
        interp = {
            "Band": ["Very Low (monthly+)", "Low (weekly)", "Mid (2–5 days)", "High (<2 days)"],
            "Period": [">20 days", "5–20 days", "2–5 days", "<2 days"],
            "Financial Meaning": ["Macro trends, long cycles", "Weekly patterns", "Short momentum", "Noise / micro-volatility"],
            "CNN Usefulness": ["High", "Medium", "Medium", "Low"],
        }
        st.dataframe(pd.DataFrame(interp), use_container_width=True, hide_index=True)

    # ── Tab 4 ──────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="rule">Assignment Report Summary</div>', unsafe_allow_html=True)
        companies = list(data.keys())
        st.markdown(f"""
        <div class="note">
        <b>Companies:</b> {', '.join(companies)}<br>
        <b>Source:</b> Yahoo Finance (yfinance)<br>
        <b>Signal:</b> Multivariate non-stationary financial time series<br>
        <b>Transform:</b> Short-Time Fourier Transform → Spectrogram S(t,f)<br>
        <b>Model:</b> CNN Regression (3× Conv2D + BN + MaxPool + GAP + Dense)
        </div>
        """, unsafe_allow_html=True)

        if results:
            st.markdown('<div class="rule">Results Table</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Metrics**")
                st.dataframe(pd.DataFrame({
                    "Metric": ["MSE","MAE","R²","Parameters"],
                    "Value": [f"{results['mse']:.4f}", f"{results['mae']:.4f}",
                              f"{results['r2']:.4f}", f"{results['model_params']:,}"]
                }), hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**Configuration**")
                st.dataframe(pd.DataFrame({
                    "Setting": ["Lookahead","Input Shape","Test Samples"],
                    "Value": [f"{results['lookahead']} days",
                              str(results['X_shape'][1:3]),
                              str(len(results['y_test']))]
                }), hide_index=True, use_container_width=True)

        st.markdown('<div class="rule">Key Findings</div>', unsafe_allow_html=True)
        findings = [
            ("Non-stationarity", "Financial signals are non-stationary — STFT is necessary to capture time-varying frequency content"),
            ("Spectrograms reveal patterns", "Low-freq bands show macro trends; high-freq bursts correspond to volatility events"),
            ("CNN learns motifs", "Convolutional filters detect recurring time-frequency shapes predictive of future prices"),
            ("Window trade-off", "L = 32 offers a good balance for daily stock data"),
            ("Feature contribution", "Close price and Volume carry complementary information"),
        ]
        for title, body in findings:
            st.markdown(f"""
            <div class="step">
                <div class="step-title">{title}</div>
                <div class="step-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="rule">References</div>', unsafe_allow_html=True)
        refs = [
            "Y. Zhang and C. Aggarwal, \"Stock Market Prediction Using Deep Learning,\" IEEE Access.",
            "A. Tsantekidis et al., \"Deep Learning for Financial Time Series Forecasting.\"",
            "S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, 1997.",
            "A. Borovykh et al., \"Conditional Time Series Forecasting with CNNs.\""
        ]
        for i, ref in enumerate(refs, 1):
            st.markdown(f"<div style='font-size:0.82rem;color:#6b6760;padding:0.35rem 0;border-bottom:1px solid #f0ede8'>[{i}] {ref}</div>", unsafe_allow_html=True)
