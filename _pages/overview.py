import streamlit as st

def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">FinSignal — Financial Time Series Forecasting</div>
        <div class="page-sub">STFT · Spectrograms · CNN Regression &nbsp;|&nbsp; Assignment 2 · Pattern Recognition</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["ABOUT", "HOW TO USE"])

    # ── ABOUT ─────────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            st.markdown('<div class="rule">Objective</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="note">
            Combine <b>time–frequency signal processing</b> with <b>deep learning</b> to predict
            stock prices. Financial series are treated as non-stationary signals, transformed via
            STFT into 2D spectrograms, and fed into a CNN regression model.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="rule">Signal Model</div>', unsafe_allow_html=True)
            st.latex(r"X(t) = [p(t),\ r(t),\ g(t),\ s(t),\ d(t)]")
            st.markdown("""
            <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.6rem">
                <span class="tag">p(t) Stock Price</span>
                <span class="tag">r(t) Revenue</span>
                <span class="tag">g(t) Profit</span>
                <span class="tag">s(t) Sensex</span>
                <span class="tag">d(t) USD-INR</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="rule">System Pipeline</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="pipe">
                <div class="pipe-node">Time Series<br><span style="font-size:0.62rem;color:#9a9590">X(t)</span></div>
                <div class="pipe-arr">→</div>
                <div class="pipe-node">STFT<br><span style="font-size:0.62rem;color:#9a9590">Sliding Window</span></div>
                <div class="pipe-arr">→</div>
                <div class="pipe-node">Spectrogram<br><span style="font-size:0.62rem;color:#9a9590">S(t,f)</span></div>
                <div class="pipe-arr">→</div>
                <div class="pipe-node">CNN f<sub>θ</sub><br><span style="font-size:0.62rem;color:#9a9590">Regression</span></div>
                <div class="pipe-arr">→</div>
                <div class="pipe-node hi">p̂(t+Δt)<br><span style="font-size:0.62rem">Forecast</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="rule">Key Formulas</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**STFT**")
                st.latex(r"STFT(t,f)=\int X(\tau)\,w(\tau-t)\,e^{-j2\pi f\tau}\,d\tau")
            with c2:
                st.markdown("**Spectrogram**")
                st.latex(r"S(t,f)=|STFT(t,f)|^2")

        with col2:
            st.markdown('<div class="rule">Assignment Tasks</div>', unsafe_allow_html=True)
            tasks = [
                ("Task 1", "Data Preparation", "≥3 companies · align · normalize"),
                ("Task 2", "Signal Processing", "FFT · STFT · Spectrograms"),
                ("Task 3", "Model Development", "CNN design · train · predict"),
                ("Task 4", "Analysis", "MSE · feature comparison · report"),
            ]
            for tid, title, desc in tasks:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">{tid}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.88rem;font-weight:600;color:#1c1c1e">{title}</div>
                    <div class="card-sub">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="rule">Figures Required</div>', unsafe_allow_html=True)
            for fig in ["Time series plot", "Frequency spectrum", "Spectrogram", "CNN architecture diagram"]:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;
                            border-bottom:1px solid #f0ede8;font-size:0.82rem;color:#1c1c1e">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                                 color:#1a4480;background:#e8eef7;padding:0.1rem 0.4rem;border-radius:2px">✓</span>
                    {fig}
                </div>
                """, unsafe_allow_html=True)

    # ── HOW TO USE ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="rule">Quick Start — 5 Steps to See Results</div>', unsafe_allow_html=True)

        steps = [
            ("Step 01", "Data Collection",
             "Go to <b>Data Collection</b> in the sidebar. Select 3 companies (e.g. Reliance, TCS, Infosys). "
             "Set date range 2020-01-01 to 2024-12-31. Click <b>Fetch Data</b>. "
             "You will see time series plots and a correlation heatmap."),
            ("Step 02", "Signal Processing",
             "Go to <b>Signal Processing</b>. Select a company and feature (Close recommended). "
             "Leave Window Length = 32, Hop Size = 8. Explore the 4 tabs: "
             "Time Domain → Frequency Spectrum → Spectrogram → All Companies."),
            ("Step 03", "Train CNN",
             "Go to <b>CNN Model</b>. Keep default hyperparameters (Epochs = 20, Lookahead = 5 days). "
             "Click <b>Train CNN</b>. A live loss curve appears during training. "
             "Without TensorFlow, a simulation runs automatically."),
            ("Step 04", "View Predictions",
             "Go to <b>Predictions</b>. See Actual vs Predicted on the test set, "
             "residual plot, scatter chart, and a simulated future forecast for any company."),
            ("Step 05", "Analysis",
             "Go to <b>Analysis</b>. Check MSE / MAE / R² metrics, training history, "
             "feature entropy comparison, resolution trade-off chart, and the full report summary."),
        ]

        for sid, stitle, sbody in steps:
            st.markdown(f"""
            <div class="step">
                <span class="step-num">{sid}</span>
                <div class="step-title">{stitle}</div>
                <div class="step-body">{sbody}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="rule">Sample Input — Run This to See Output Immediately</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="note">
        Copy these exact settings when you first open the app to see a complete end-to-end result.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Data Collection Settings**")
            st.code("""Companies : Reliance Industries
              TCS
              Infosys
Start Date : 2020-01-01
End Date   : 2024-12-31
→ Click: Fetch Data""", language="text")

            st.markdown("**Signal Processing Settings**")
            st.code("""Company  : Reliance Industries
Feature  : Close
Window L : 32
Hop Size : 8
→ View all 4 tabs""", language="text")

        with col_b:
            st.markdown("**CNN Model Settings**")
            st.code("""Window Length : 32
Hop Size      : 8
Lookahead     : 5 days
Epochs        : 20
Dropout       : 0.3
→ Click: Train CNN""", language="text")

            st.markdown("**Expected Outputs**")
            st.markdown("""
            <div style="font-size:0.82rem;line-height:1.9;color:#1c1c1e">
            ✓ &nbsp; Normalized price chart (3 companies)<br>
            ✓ &nbsp; FFT frequency spectrum with dominant peak<br>
            ✓ &nbsp; Color spectrogram S(t,f) heatmap<br>
            ✓ &nbsp; Live training loss curve<br>
            ✓ &nbsp; Actual vs Predicted price overlay<br>
            ✓ &nbsp; MSE · MAE · R² evaluation table<br>
            ✓ &nbsp; Future forecast with confidence band
            </div>
            """, unsafe_allow_html=True)


