import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

COLORS = ["#1a4480", "#c0392b", "#1a6640", "#7a5c00", "#5a3472"]


def get_series(df, col_name):
    """Safely extract 1-D Series — handles yfinance multi-level columns."""
    col = df[col_name]
    if isinstance(col, __import__('pandas').DataFrame):
        col = col.iloc[:, 0]
    import pandas as pd
    return pd.Series(col.values, index=df.index, name=col_name)

LIGHT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafaf8",
    font=dict(family="IBM Plex Sans", color="#1c1c1e", size=11),
)

def compute_fft(series):
    n = len(series)
    y = fft(series.values)
    freqs = fftfreq(n, d=1)
    pos = freqs > 0
    return freqs[pos], np.abs(y[pos]) * 2 / n

def compute_stft(series, window_len, hop):
    x = series.values.astype(float)
    x = (x - x.mean()) / (x.std() + 1e-8)
    f, t, Zxx = scipy_signal.stft(x, fs=1.0, window='hann',
                                   nperseg=window_len, noverlap=window_len - hop)
    return f, t, np.abs(Zxx) ** 2

def show():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Task 2 — Signal Processing</div>
        <div class="page-sub">Fourier Transform · STFT · Spectrogram Generation</div>
    </div>
    """, unsafe_allow_html=True)

    if "stock_data" not in st.session_state:
        st.markdown('<div class="note">Complete <b>Data Collection</b> first.</div>', unsafe_allow_html=True)
        return

    data = st.session_state["stock_data"]

    st.markdown('<div class="rule">Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: company = st.selectbox("Company", list(data.keys()))
    with c2: feature = st.selectbox("Feature", ["Close","Volume","High","Low"])
    with c3: window_len = st.slider("Window Length (L)", 8, 64, 32, step=4)
    with c4: hop = st.slider("Hop Size (H)", 1, window_len, window_len // 4)

    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                color:#6b6760;margin:0.4rem 0 0.8rem;flex-wrap:wrap">
        <span>Window L = <b style="color:#1a4480">{window_len}</b></span>
        <span>Hop H = <b style="color:#1a4480">{hop}</b></span>
        <span>Overlap = <b style="color:#1a4480">{window_len-hop}</b></span>
        <span>Freq resolution ≈ <b style="color:#1a4480">{1/window_len:.3f} /day</b></span>
    </div>
    """, unsafe_allow_html=True)

    df = data[company]
    series = get_series(df, feature).dropna()

    tab1, tab2, tab3, tab4 = st.tabs(["TIME DOMAIN", "FREQUENCY SPECTRUM", "SPECTROGRAM", "ALL COMPANIES"])

    # ── Time Domain ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="rule">Time Domain Signal X(t)</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Raw Signal", "Normalised (z-score)"], vertical_spacing=0.14)
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
            line=dict(color="#1a4480", width=1.4), name="Raw"), row=1, col=1)
        norm = (series - series.mean()) / series.std()
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values,
            line=dict(color="#c0392b", width=1.4), name="Normalised"), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#e2ddd6", row=2, col=1)
        fig.update_layout(**LIGHT, height=480, showlegend=False,
            margin=dict(l=60,r=20,t=50,b=50))
        for ax in ["xaxis","xaxis2"]: fig.update_layout(**{ax: dict(gridcolor="#eeece8", showline=True, linecolor="#cdc8c0")})
        for ax in ["yaxis","yaxis2"]: fig.update_layout(**{ax: dict(gridcolor="#eeece8", showline=True, linecolor="#cdc8c0")})
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Theory — Time Domain"):
            st.markdown(r"""
            Financial series $X(t)$ are **non-stationary**: their mean and variance change over time.
            Z-score normalisation: $\hat{X}(t) = \frac{X(t) - \mu}{\sigma}$
            This is why we need STFT instead of a single global FFT.
            """)

    # ── Frequency Spectrum ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="rule">Frequency Spectrum — FFT</div>', unsafe_allow_html=True)
        freqs, amps = compute_fft(series)
        dom_idx = np.argmax(amps)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freqs, y=amps, fill='tozeroy',
            line=dict(color="#1a4480", width=1.5),
            fillcolor="rgba(26,68,128,0.07)", name="Amplitude"))
        fig2.add_trace(go.Scatter(x=[freqs[dom_idx]], y=[amps[dom_idx]],
            mode='markers+text',
            marker=dict(color="#c0392b", size=9, symbol="diamond"),
            text=[f"  {freqs[dom_idx]:.4f}/day"],
            textfont=dict(color="#c0392b", size=10, family="IBM Plex Mono"),
            name="Dominant Freq"))
        fig2.update_layout(**LIGHT, height=380,
            xaxis=dict(gridcolor="#eeece8", title="Frequency [cycles/day]", showline=True, linecolor="#cdc8c0"),
            yaxis=dict(gridcolor="#eeece8", title="Amplitude", showline=True, linecolor="#cdc8c0"),
            margin=dict(l=60,r=20,t=30,b=50), hovermode="x",
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Dominant Frequency", f"{freqs[dom_idx]:.4f} /day")
        mc2.metric("Approx. Period", f"{1/freqs[dom_idx]:.0f} days")
        mc3.metric("Spectral Entropy", f"{-np.sum((amps/amps.sum())*np.log(amps/amps.sum()+1e-9)):.2f}")

        with st.expander("Theory — Fourier Transform"):
            st.markdown(r"""
            $$X(f) = \sum_{n=0}^{N-1} x[n]\, e^{-j2\pi fn/N}$$
            - **Low freq** → long-term trends (quarterly/yearly cycles)
            - **High freq** → short-term noise, daily fluctuations
            """)

    # ── Spectrogram ────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="rule">Spectrogram S(t,f) — Time-Frequency Representation</div>', unsafe_allow_html=True)
        if len(series) < window_len * 2:
            st.error(f"Signal too short ({len(series)} pts). Use smaller window or wider date range.")
        else:
            f_ax, t_ax, spec = compute_stft(series, window_len, hop)
            spec_db = 10 * np.log10(spec + 1e-10)

            n_times = len(t_ax)
            step = max(1, len(series.index) // n_times)
            sampled = [str(series.index[min(int(i*step), len(series.index)-1)])[:10] for i in range(n_times)]

            fig3 = go.Figure(go.Heatmap(
                z=spec_db, x=sampled, y=f_ax,
                colorscale=[
                    [0.0, "#f7f6f3"],
                    [0.25, "#c9d9ee"],
                    [0.55, "#1a4480"],
                    [0.8,  "#c0392b"],
                    [1.0,  "#1c1c1e"]
                ],
                colorbar=dict(
                    title=dict(text="Power (dB)", font=dict(color="#6b6760", size=10)),
                    tickfont=dict(color="#6b6760", size=9, family="IBM Plex Mono"),
                    thickness=12
                ),
                hovertemplate="Time: %{x}<br>Freq: %{y:.4f}/day<br>Power: %{z:.1f} dB<extra></extra>"
            ))
            fig3.update_layout(**LIGHT, height=400,
                xaxis=dict(gridcolor="#eeece8", title="Time →", nticks=10, showline=True, linecolor="#cdc8c0"),
                yaxis=dict(gridcolor="#eeece8", title="Frequency [cycles/day]", showline=True, linecolor="#cdc8c0"),
                margin=dict(l=65,r=20,t=40,b=60),
                title=dict(text=f"S(t,f) · {company} [{feature}]",
                           font=dict(color="#1c1c1e", family="IBM Plex Mono", size=12)))
            st.plotly_chart(fig3, use_container_width=True)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Shape", f"{spec.shape[0]}×{spec.shape[1]}")
            s2.metric("Freq Bins", f"{spec.shape[0]}")
            s3.metric("Time Frames", f"{len(t_ax)}")
            s4.metric("Peak Power", f"{spec_db.max():.1f} dB")

            with st.expander("Theory — Spectrogram"):
                st.markdown(r"""
                $S(t,f) = |STFT(t,f)|^2$ is a 2D matrix $S \in \mathbb{R}^{T \times F}$

                - **Rows** = frequency bins · **Cols** = time frames · **Intensity** = energy
                - Dark bands at low freq = long-term trend
                - Bright bursts at high freq = volatility events (earnings, crashes)
                - This image is the **direct input to the CNN**
                """)

    # ── All Companies ──────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="rule">Spectrograms — All Companies (Close Price)</div>', unsafe_allow_html=True)
        companies = list(data.keys())
        n_cols = min(len(companies), 3)
        for row_i in range((len(companies) + n_cols - 1) // n_cols):
            cols = st.columns(n_cols)
            for col_i in range(n_cols):
                ci = row_i * n_cols + col_i
                if ci >= len(companies): break
                cname = companies[ci]
                cseries = get_series(data[cname], 'Close').dropna()
                with cols[col_i]:
                    st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;font-weight:600;color:#1c1c1e;margin-bottom:0.4rem'>{cname}</div>", unsafe_allow_html=True)
                    if len(cseries) < window_len * 2:
                        st.info("Insufficient data.")
                        continue
                    cf, ct, cspec = compute_stft(cseries, window_len, hop)
                    cdb = 10 * np.log10(cspec + 1e-10)
                    cfig = go.Figure(go.Heatmap(
                        z=cdb,
                        colorscale=[[0,"#f7f6f3"],[0.4,"#aec2e0"],[0.75,"#1a4480"],[1,"#c0392b"]],
                        showscale=False
                    ))
                    cfig.update_layout(**LIGHT, height=210,
                        margin=dict(l=35,r=8,t=8,b=25),
                        xaxis=dict(gridcolor="#eeece8", showticklabels=False),
                        yaxis=dict(gridcolor="#eeece8", title=dict(text="Freq", font=dict(size=9))))
                    st.plotly_chart(cfig, use_container_width=True)