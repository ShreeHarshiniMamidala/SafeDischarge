from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from readability import get_readability_metrics
from completeness import detect_elements

MODEL_PATH = BASE_DIR / "models" / "quality_model.joblib"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"

st.set_page_config(
    page_title="SafeDischarge Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.hero h1 {
    font-size: 2rem;
    margin: 0;
    font-weight: 700;
}

.hero p {
    margin: 0.45rem 0 0 0;
    color: #dbeafe;
    font-size: 1rem;
}

.card {
    background: white;
    padding: 1rem 1.1rem;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(226,232,240,0.9);
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(226,232,240,0.9);
    min-height: 100px;
}

.metric-label {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 0.35rem;
}

.metric-value {
    color: #0f172a;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-sub {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 0.25rem;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 0.2rem;
}

.badge-high {
    background: #dcfce7;
    color: #166534;
}

.badge-medium {
    background: #fef3c7;
    color: #92400e;
}

.badge-low {
    background: #fee2e2;
    color: #991b1b;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
}

.small-note {
    color: #64748b;
    font-size: 0.88rem;
}

textarea {
    border-radius: 14px !important;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

div[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 16px;
    padding: 0.4rem 0.4rem 0.2rem 0.4rem;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(226,232,240,0.9);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>SafeDischarge Dashboard</h1>
    <p>Analyze discharge instructions for quality, readability, completeness, and patient actionability.</p>
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    "Paste discharge instructions here",
    height=220,
    placeholder="Example: Take your antibiotics as prescribed. Drink plenty of fluids. Follow up with your doctor in one week. Seek urgent care if breathing gets worse."
)

def label_to_score(label):
    mapping = {"low": 35, "medium": 65, "high": 90}
    return mapping.get(str(label).lower(), 0)

def label_badge(label):
    label = str(label).lower()
    if label == "high":
        return '<span class="badge badge-high">High</span>'
    if label == "medium":
        return '<span class="badge badge-medium">Medium</span>'
    return '<span class="badge badge-low">Low</span>'

def safe_metric(metrics, key, default="N/A"):
    value = metrics.get(key, default)
    if isinstance(value, float):
        return round(value, 2)
    return value

def make_gauge(score, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100"},
        title={"text": f"Predicted Quality: {str(label).title()}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 75], "color": "#fef3c7"},
                {"range": [75, 100], "color": "#dcfce7"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def make_completeness_bar(comp):
    rename_map = {
        "has_medication_review": "Medication",
        "has_followup": "Follow-up",
        "has_warning_signs": "Warnings",
        "has_home_care": "Home Care",
        "has_test_results": "Test Results"
    }
    df = pd.DataFrame({
        "Category": [rename_map.get(k, k) for k in comp.keys()],
        "Present": [int(v) for v in comp.values()]
    })

    fig = px.bar(
        df,
        x="Category",
        y="Present",
        text="Present",
        color="Present",
        color_continuous_scale=["#cbd5e1", "#2563eb"],
        title="Completeness Elements"
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1.2], tickvals=[0, 1], title="Present")
    fig.update_xaxes(title="Category")
    fig.update_layout(height=340, showlegend=False, coloraxis_showscale=False)
    return fig

def make_radar(comp):
    rename_map = {
        "has_medication_review": "Medication",
        "has_followup": "Follow-up",
        "has_warning_signs": "Warnings",
        "has_home_care": "Home Care",
        "has_test_results": "Test Results"
    }
    categories = [rename_map.get(k, k) for k in comp.keys()]
    values = [int(v) for v in comp.values()]

    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="Profile",
        line=dict(color="#2563eb", width=3),
        fillcolor="rgba(37, 99, 235, 0.25)"
    ))
    fig.update_layout(
        title="Instruction Profile",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 1])
        ),
        showlegend=False,
        height=360,
        margin=dict(l=30, r=30, t=60, b=20)
    )
    return fig

def make_readability_bar(readability):
    rows = [
        ("FK Grade", safe_metric(readability, "flesch_kincaid_grade", 0)),
        ("Reading Ease", safe_metric(readability, "flesch_reading_ease", 0)),
        ("Words", safe_metric(readability, "word_count", 0)),
        ("Sentences", safe_metric(readability, "sentence_count", 0)),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    fig = px.bar(df, x="Metric", y="Value", text="Value", title="Readability Overview")
    fig.update_traces(textposition="outside", marker_color="#0f766e")
    fig.update_layout(height=340, showlegend=False)
    return fig

if st.button("Analyze Instructions", use_container_width=True):
    if not text.strip():
        st.warning("Please paste discharge instructions before running the analysis.")
    else:
        readability = get_readability_metrics(text)
        completeness = detect_elements(text)

        predicted_label = "N/A"
        quality_score = 0

        if MODEL_PATH.exists() and ENCODER_PATH.exists():
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            X_input = pd.DataFrame({"instruction_text": [text]})
            pred = model.predict(X_input)[0]
            predicted_label = encoder.inverse_transform([pred])[0]
            quality_score = label_to_score(predicted_label)

        word_count = len(text.split())
        reading_grade = safe_metric(readability, "flesch_kincaid_grade", "N/A")
        sentence_count = safe_metric(readability, "sentence_count", "N/A")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted quality</div>
                <div class="metric-value">{str(predicted_label).title()}</div>
                <div class="metric-sub">Model-based label</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Reading grade</div>
                <div class="metric-value">{reading_grade}</div>
                <div class="metric-sub">Lower is easier</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Word count</div>
                <div class="metric-value">{word_count}</div>
                <div class="metric-sub">Instruction length</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            coverage = sum(int(v) for v in completeness.values())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Elements covered</div>
                <div class="metric-value">{coverage}/5</div>
                <div class="metric-sub">Completeness score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Prediction")
        st.markdown(label_badge(predicted_label), unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Dashboard", "Readability", "Completeness"])

        with tab1:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(make_gauge(quality_score, predicted_label), use_container_width=True)
                st.plotly_chart(make_readability_bar(readability), use_container_width=True)
            with right:
                st.plotly_chart(make_completeness_bar(completeness), use_container_width=True)
                st.plotly_chart(make_radar(completeness), use_container_width=True)

        with tab2:
            st.markdown('<div class="section-title">Readability details</div>', unsafe_allow_html=True)
            rd1, rd2 = st.columns(2)
            with rd1:
                st.json(readability)
            with rd2:
                st.markdown("""
                <div class="card">
                    <div class="section-title">Interpretation guide</div>
                    <div class="small-note">
                        Flesch-Kincaid grade estimates the education level needed to understand the text.
                        Lower grade levels are usually better for patient-facing instructions.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-title">Completeness details</div>', unsafe_allow_html=True)
            cp1, cp2 = st.columns([1.1, 1])
            with cp1:
                st.json(completeness)
            with cp2:
                st.markdown(f"""
                <div class="card">
                    <div class="section-title">Coverage summary</div>
                    <div class="small-note">
                        This instruction covers <b>{sum(int(v) for v in completeness.values())}</b> out of 5 tracked discharge elements:
                        medication review, follow-up, warning signs, home care, and test results.
                    </div>
                </div>
                """, unsafe_allow_html=True)