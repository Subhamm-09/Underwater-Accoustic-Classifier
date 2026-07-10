"""
Streamlit Web App
Underwater Acoustic Classifier
"""

import tempfile

import streamlit as st

from src.predict import predict_audio


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Underwater Acoustic Classifier",
    page_icon="🌊",
    layout="centered",
)

# ==========================================================
# Class styling (muted, professional palette)
# ==========================================================

CLASS_STYLE = {
    "ambience": {"icon": "🌊", "color": "#5B8FA8", "label": "Ambience"},
    "biological": {"icon": "🐋", "color": "#7A6A9E", "label": "Biological"},
    "vessels": {"icon": "🚢", "color": "#B08D57", "label": "Vessels"},
}
DEFAULT_STYLE = {"icon": "🔊", "color": "#8891A0", "label": "Unknown"}

ACCENT = "#C6A664"        # muted gold accent
ACCENT_SOFT = "#E8DCC0"   # pale gold for subtle highlights


def style_for(label: str) -> dict:
    return CLASS_STYLE.get(label.lower(), DEFAULT_STYLE)


# ==========================================================
# Custom CSS
# ==========================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: #0E1420;
        color: #E6E8EC;
    }}

    /* ---------- Header ---------- */
    .header-wrap {{
        text-align: center;
        padding: 1.6rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.8rem;
    }}
    .header-wrap h1 {{
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2.1rem;
        color: #F4F1EA;
        margin: 0;
        letter-spacing: 0.3px;
    }}
    .header-wrap p {{
        font-family: 'Inter', sans-serif;
        color: #9AA2AF;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }}
    .header-rule {{
        width: 60px;
        height: 2px;
        background: {ACCENT};
        margin: 0.9rem auto 0 auto;
        border-radius: 2px;
    }}

    /* ---------- Section card ---------- */
    .section-card {{
        background: #141B29;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1.4rem;
    }}
    .section-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {ACCENT};
        font-weight: 600;
        margin-bottom: 0.6rem;
    }}

    /* ---------- Result banner ---------- */
    .result-banner {{
        background: #141B29;
        border: 1px solid rgba(198,166,100,0.35);
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.1rem;
        margin: 0.4rem 0 1.6rem 0;
    }}
    .result-icon {{
        font-size: 2.2rem;
        line-height: 1;
    }}
    .result-text .label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #9AA2AF;
    }}
    .result-text h2 {{
        font-family: 'Playfair Display', serif;
        margin: 0.15rem 0 0 0;
        color: #F4F1EA;
        font-size: 1.5rem;
        font-weight: 700;
    }}

    /* ---------- Confidence score rows ---------- */
    .score-card {{
        background: #141B29;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.65rem;
    }}
    .score-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.55rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.92rem;
        color: #E6E8EC;
    }}
    .score-top .pct {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }}
    .score-bar-bg {{
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }}
    .score-bar-fill {{
        height: 100%;
        border-radius: 999px;
    }}

    /* ---------- Buttons ---------- */
    .stButton>button {{
        background: {ACCENT};
        color: #14181F;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        transition: all 0.15s ease;
    }}
    .stButton>button:hover {{
        background: #D8BC7E;
        color: #14181F;
    }}

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] {{
        border-radius: 10px;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: #10151F;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    .sidebar-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        color: #F4F1EA;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .sidebar-sub {{
        color: #9AA2AF;
        font-size: 0.82rem;
        margin-bottom: 1.1rem;
    }}
    .sidebar-heading {{
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: {ACCENT};
        font-weight: 600;
        margin: 1.1rem 0 0.5rem 0;
    }}
    .sidebar-row {{
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #C7CCD4;
        padding: 0.25rem 0;
        border-bottom: 1px dashed rgba(255,255,255,0.08);
    }}
    .sidebar-row span:last-child {{
        color: #F4F1EA;
        font-weight: 600;
    }}
    .badge-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.4rem;
    }}
    .badge {{
        background: rgba(198,166,100,0.12);
        border: 1px solid rgba(198,166,100,0.3);
        color: {ACCENT_SOFT};
        font-size: 0.72rem;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Sidebar — model info
# ==========================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">🌊 About this Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Underwater acoustic scene classifier</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-row"><span>Algorithm</span><span>Logistic Regression</span></div>
        <div class="sidebar-row"><span>Selected from</span><span>5 candidates</span></div>
        <div class="sidebar-row"><span>Accuracy</span><span>87.9%</span></div>
        <div class="sidebar-row"><span>Macro F1</span><span>0.87</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Classes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-row"><span>🌊 Ambience</span><span>360 samples</span></div>
        <div class="sidebar-row"><span>🐋 Biological</span><span>500 samples</span></div>
        <div class="sidebar-row"><span>🚢 Vessels</span><span>951 samples</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Built With</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="badge-row">
            <span class="badge">Python</span>
            <span class="badge">scikit-learn</span>
            <span class="badge">Streamlit</span>
            <span class="badge">Librosa</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size:0.8rem; color:#9AA2AF; line-height:1.5;">
        Logistic Regression was chosen over KNN, SVM, Random Forest, and
        XGBoost after per-class evaluation showed it handled the Ambience
        class far more reliably.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# Header
# ==========================================================

st.markdown(
    """
    <div class="header-wrap">
        <h1>Underwater Acoustic Classifier</h1>
        <p>Upload a .wav recording to identify Ambience, Biological, or Vessel sounds.</p>
        <div class="header-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Upload Audio
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Upload Audio</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# Prediction
# ==========================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Analyze Recording"):

        with st.spinner("Analyzing audio..."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            try:
                prediction, scores = predict_audio(temp_path)

                top_style = style_for(prediction)

                # ---- Result banner ----
                st.markdown(
                    f"""
                    <div class="result-banner">
                        <div class="result-icon">{top_style['icon']}</div>
                        <div class="result-text">
                            <div class="label">Predicted Class</div>
                            <h2>{prediction}</h2>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Confidence scores ----
                st.markdown('<div class="section-label">Confidence Scores</div>', unsafe_allow_html=True)

                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                for label, probability in sorted_scores:
                    s = style_for(label)
                    pct = probability * 100
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-top">
                                <span>{s['icon']} {s['label']}</span>
                                <span class="pct" style="color:{s['color']};">{pct:.2f}%</span>
                            </div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{pct}%; background:{s['color']};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)