import streamlit as st

st.set_page_config(
    page_title="FinSignal · Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:       #f7f6f3;
  --surface:  #ffffff;
  --surface2: #f0ede8;
  --border:   #e2ddd6;
  --border2:  #cdc8c0;
  --accent:   #1a4480;
  --text:     #1c1c1e;
  --muted:    #6b6760;
  --muted2:   #9a9590;
  --green:    #1a6640;
  --red:      #b91c1c;
  --tag-bg:   #eae7e1;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { background: #0f2d5a !important; }

.stSelectbox label, .stSlider label, .stMultiSelect label,
.stNumberInput label, .stRadio label, .stDateInput label {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    font-weight: 500 !important;
}
[data-baseweb="select"] > div { background: var(--surface) !important; border-color: var(--border2) !important; border-radius: 3px !important; }
[data-baseweb="select"] span { color: var(--text) !important; }
[data-baseweb="select"] div { color: var(--text) !important; }
[data-baseweb="select"] input { color: var(--text) !important; }
[data-baseweb="select"] [data-baseweb="tag"] { background: var(--tag-bg) !important; color: var(--text) !important; }
.stSelectbox div[data-baseweb="select"] > div > div { color: var(--text) !important; }
div[data-testid="stSelectbox"] div { color: var(--text) !important; }
div[data-testid="stMultiSelect"] div { color: var(--text) !important; }
/* dropdown menu items */
[data-baseweb="menu"] { background: #ffffff !important; border: 1px solid var(--border2) !important; }
[data-baseweb="menu"] li { color: #1c1c1e !important; background: #ffffff !important; font-family: 'IBM Plex Sans', sans-serif !important; }
[data-baseweb="menu"] li:hover { background: #f0ede8 !important; color: #1c1c1e !important; }
[data-baseweb="menu"] li[aria-selected="true"] { background: #e8eef7 !important; color: #1a4480 !important; }
[role="option"] { color: #1c1c1e !important; background: #ffffff !important; }
[role="option"]:hover { background: #f0ede8 !important; }
[role="listbox"] { background: #ffffff !important; }
ul[data-baseweb="menu"] { background: #ffffff !important; }
ul[data-baseweb="menu"] li * { color: #1c1c1e !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; background: transparent !important; border: none !important; padding: 0.6rem 1rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

div[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 3px !important; padding: 1rem 1.2rem !important; }
div[data-testid="stMetric"] label { color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.09em !important; }
div[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.35rem !important; }

[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 3px !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 3px !important; }

/* Custom components */
.page-header { border-bottom: 2px solid var(--text); padding-bottom: 0.7rem; margin-bottom: 1.8rem; }
.page-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 600; color: var(--text); text-transform: uppercase; letter-spacing: 0.05em; }
.page-sub { font-size: 0.82rem; color: var(--muted); margin-top: 0.2rem; }

.rule { font-family: 'IBM Plex Mono', monospace; font-size: 0.67rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; margin: 1.6rem 0 0.8rem; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 3px; padding: 1.1rem 1.3rem; margin-bottom: 0.6rem; }
.card-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.25rem; }
.card-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem; font-weight: 600; color: var(--accent); }
.card-sub { font-size: 0.77rem; color: var(--muted2); margin-top: 0.15rem; }

.tag { display: inline-block; background: var(--tag-bg); border: 1px solid var(--border2); color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; padding: 0.12rem 0.5rem; border-radius: 2px; margin-right: 0.3rem; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.06em; }

.note { background: #eef2fa; border-left: 3px solid var(--accent); padding: 0.75rem 1rem; margin: 0.8rem 0; font-size: 0.83rem; color: var(--text); border-radius: 0 3px 3px 0; }

.step { background: var(--surface); border: 1px solid var(--border); border-radius: 3px; padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; }
.step-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.67rem; font-weight: 600; color: var(--accent); background: #e8eef7; border-radius: 2px; padding: 0.12rem 0.45rem; text-transform: uppercase; letter-spacing: 0.07em; display: inline-block; margin-bottom: 0.4rem; }
.step-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600; color: var(--text); }
.step-body { font-size: 0.82rem; color: var(--muted); margin-top: 0.25rem; line-height: 1.5; }

.pipe { display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin: 1rem 0; }
.pipe-node { background: var(--surface); border: 1px solid var(--border2); padding: 0.45rem 0.9rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--text); border-radius: 2px; }
.pipe-node.hi { background: #e8eef7; border-color: var(--accent); color: var(--accent); font-weight: 600; }
.pipe-arr { color: var(--border2); padding: 0 0.4rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.4rem 0 1.2rem">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.95rem;font-weight:600;color:#1c1c1e">FinSignal</div>
        <div style="font-size:0.75rem;color:#6b6760;font-family:'IBM Plex Sans',sans-serif;margin-top:0.2rem">
            Pattern Recognition · Assign. 2
        </div>
        <div style="border-top:1px solid #e2ddd6;margin-top:1rem"></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Overview",
         "Data Collection",
         "Signal Processing",
         "CNN Model",
         "Predictions",
         "Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="border-top:1px solid #e2ddd6;margin-top:1.5rem;padding-top:1rem">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;text-transform:uppercase;
                    letter-spacing:0.1em;color:#9a9590;margin-bottom:0.5rem">Stack</div>
    """, unsafe_allow_html=True)
    for t in ["Python 3.11", "Streamlit", "SciPy", "Plotly", "yfinance"]:
        st.markdown(f"<span class='tag'>{t}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
import importlib

page_map = {
    "Overview":          "_pages.overview",
    "Data Collection":   "_pages.data",
    "Signal Processing": "_pages.signal_proc",
    "CNN Model":         "_pages.cnn_model",
    "Predictions":       "_pages.predictions",
    "Analysis":          "_pages.analysis",
}

pg = importlib.import_module(page_map[page])
pg.show()