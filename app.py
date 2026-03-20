"""
app.py — Heart Disease Classifier · Streamlit GUI
Run with:  streamlit run app.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
SRC     = os.path.join(BASE, "src")
MODELS  = os.path.join(BASE, "models")
PLOTS   = os.path.join(BASE, "plots")
sys.path.insert(0, SRC)

from preprocessor import engineer_features   # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- base background ---------- */
.stApp { background: #0d1117; color: #e6edf3; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h2 { color: #ff6b6b; font-size: 1.1rem; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #161b22;
    border-radius: 8px 8px 0 0;
    color: #8b949e;
    padding: 10px 22px;
    font-weight: 600;
    border: 1px solid #30363d;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff6b6b22, #56cfe122) !important;
    color: #56cfe1 !important;
    border-color: #56cfe1 !important;
}

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }

/* ---------- slider ---------- */
.stSlider > div > div > div { background: #56cfe1 !important; }

/* ---------- select box ---------- */
[data-baseweb="select"] > div { background: #161b22 !important; border-color: #30363d !important; }

/* ---------- result cards ---------- */
.result-danger {
    background: linear-gradient(135deg, #ff6b6b22, #ff6b6b11);
    border: 2px solid #ff6b6b;
    border-radius: 16px; padding: 24px; text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #56cfe122, #56cfe111);
    border: 2px solid #56cfe1;
    border-radius: 16px; padding: 24px; text-align: center;
}
.result-title { font-size: 2rem; font-weight: 700; margin-bottom: 6px; }
.result-sub   { font-size: 1rem; color: #8b949e; }

/* ---------- section headers ---------- */
.section-hdr {
    font-size: 1.25rem; font-weight: 700; color: #e6edf3;
    border-left: 4px solid #56cfe1; padding-left: 12px; margin: 20px 0 12px;
}

/* ---------- info box ---------- */
.info-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = os.path.join(MODELS, "heart_disease_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = os.path.join(MODELS, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def plot_path(name: str) -> str:
    return os.path.join(PLOTS, name)


def gauge_chart(probability: float, label: str, color: str) -> go.Figure:
    """Animated Plotly gauge for disease probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        delta={"reference": 50, "increasing": {"color": "#ff6b6b"},
               "decreasing": {"color": "#56cfe1"}},
        title={"text": label, "font": {"size": 18, "color": "#e6edf3"}},
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#30363d", "tickfont": {"color": "#8b949e"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#1a2d2a"},
                {"range": [30, 60], "color": "#2d2a1a"},
                {"range": [60, 100],"color": "#2d1a1a"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.8, "value": probability * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", font_color="#e6edf3",
        height=320, margin=dict(t=40, b=20, l=30, r=30),
    )
    return fig


def roc_chart(report: dict) -> go.Figure:
    """Approximate ROC bar from precision / recall."""
    classes = {"No Disease": "0", "Disease": "1"}
    prec, rec, f1 = [], [], []
    for label, key in classes.items():
        r = report.get(key, {})
        prec.append(r.get("precision", 0))
        rec.append(r.get("recall", 0))
        f1.append(r.get("f1-score", 0))

    fig = go.Figure()
    cats = list(classes.keys())
    for vals, name, color in [
        (prec, "Precision", "#56cfe1"),
        (rec,  "Recall",    "#ff6b6b"),
        (f1,   "F1-Score",  "#a0d995"),
    ]:
        fig.add_trace(go.Bar(name=name, x=cats, y=vals,
                             marker_color=color, text=[f"{v:.2f}" for v in vals],
                             textposition="outside"))
    fig.update_layout(
        barmode="group", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", title="Per-Class Metrics",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d", range=[0, 1.1]),
        height=350, margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Patient Input Form
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🫀 Heart Disease\nClassifier")
    st.markdown("---")
    st.markdown("### Patient Data")

    age      = st.slider("Age (years)",           29, 77, 54, key="age")
    sex      = st.selectbox("Sex", ["Male (1)", "Female (0)"], key="sex")
    cp       = st.selectbox("Chest Pain Type",
                            ["Typical Angina (0)", "Atypical Angina (1)",
                             "Non-Anginal Pain (2)", "Asymptomatic (3)"], key="cp")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 131, key="trestbps")
    chol     = st.slider("Serum Cholesterol (mg/dL)",     126, 564, 246, key="chol")
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                            ["No (0)", "Yes (1)"], key="fbs")
    restecg  = st.selectbox("Resting ECG Results",
                            ["Normal (0)", "ST-T Abnormality (1)",
                             "Left Ventricular Hypertrophy (2)"], key="restecg")
    thalach  = st.slider("Max Heart Rate Achieved",   71, 202, 149, key="thalach")
    exang    = st.selectbox("Exercise-Induced Angina",
                            ["No (0)", "Yes (1)"], key="exang")
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1, key="oldpeak")
    slope    = st.selectbox("Slope of Peak ST Segment",
                            ["Upsloping (0)", "Flat (1)", "Downsloping (2)"], key="slope")
    ca       = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)",
                            ["0", "1", "2", "3", "4"], key="ca")
    thal     = st.selectbox("Thalassemia (thal)",
                            ["Normal (0)", "Fixed Defect (1)",
                             "Reversible Defect (2)", "Unknown (3)"], key="thal")

    st.markdown("---")
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True, type="primary")


# ── Parse sidebar values ───────────────────────────────────────────────────────
def _last_int(s: str) -> int:
    return int(s.split("(")[-1].rstrip(")"))

patient = {
    "age":      age,
    "sex":      _last_int(sex),
    "cp":       _last_int(cp),
    "trestbps": trestbps,
    "chol":     chol,
    "fbs":      _last_int(fbs),
    "restecg":  _last_int(restecg),
    "thalach":  thalach,
    "exang":    _last_int(exang),
    "oldpeak":  oldpeak,
    "slope":    _last_int(slope),
    "ca":       int(ca),
    "thal":     _last_int(thal),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:10px 0 20px;'>
  <h1 style='font-size:2.6rem; font-weight:800;
             background:linear-gradient(90deg,#ff6b6b,#56cfe1);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             margin-bottom:4px;'>
    Heart Disease Classifier
  </h1>
  <p style='color:#8b949e; font-size:1rem;'>
    Machine Learning · Random Forest · ROC-AUC 0.898
  </p>
</div>
""", unsafe_allow_html=True)

model   = load_model()
metrics = load_metrics()

if model is None:
    st.error("Model not found. Please run `python main.py` first to train the model.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_pred, tab_eda, tab_perf, tab_feat = st.tabs([
    "🔍 Prediction",
    "📊 EDA Dashboard",
    "📈 Model Performance",
    "🎯 Feature Importance",
])


# ─────────────────────────────────────────────────────────────────
#  TAB 1 — PREDICTION
# ─────────────────────────────────────────────────────────────────
with tab_pred:
    # Run prediction on every slider change OR on button press
    df_input = pd.DataFrame([patient])
    df_feat  = engineer_features(df_input.copy())

    prob     = model.predict_proba(df_feat)[0][1]
    pred     = model.predict(df_feat)[0]
    is_sick  = bool(pred == 1)

    col_result, col_gauge = st.columns([1, 1])

    with col_result:
        if is_sick:
            st.markdown(f"""
            <div class='result-danger'>
              <div class='result-title' style='color:#ff6b6b;'>⚠️ Heart Disease Detected</div>
              <div class='result-sub'>The model predicts a high likelihood of heart disease.</div>
              <br>
              <span style='font-size:3rem; font-weight:800; color:#ff6b6b;'>{prob:.1%}</span>
              <br><span style='color:#8b949e;'>Disease Probability</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-safe'>
              <div class='result-title' style='color:#56cfe1;'>✅ No Heart Disease</div>
              <div class='result-sub'>The model predicts a low likelihood of heart disease.</div>
              <br>
              <span style='font-size:3rem; font-weight:800; color:#56cfe1;'>{(1-prob):.1%}</span>
              <br><span style='color:#8b949e;'>Healthy Probability</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Patient summary table
        st.markdown("<div class='section-hdr'>Patient Summary</div>", unsafe_allow_html=True)
        labels = {
            "age": "Age", "sex": "Sex", "cp": "Chest Pain", "trestbps": "Blood Pressure",
            "chol": "Cholesterol", "fbs": "High Fasting BS", "restecg": "ECG Result",
            "thalach": "Max Heart Rate", "exang": "Exercise Angina", "oldpeak": "ST Depression",
            "slope": "ST Slope", "ca": "Major Vessels", "thal": "Thalassemia",
        }
        rows = [(labels[k], v) for k, v in patient.items()]
        df_display = pd.DataFrame(rows, columns=["Feature", "Value"])
        st.dataframe(df_display, hide_index=True, use_container_width=True)

    with col_gauge:
        color = "#ff6b6b" if is_sick else "#56cfe1"
        label = "Disease Probability" if is_sick else "Disease Probability"
        st.plotly_chart(gauge_chart(prob, label, color),
                        use_container_width=True, config={"displayModeBar": False})

        # Risk level indicator
        st.markdown("<div class='section-hdr'>Risk Breakdown</div>", unsafe_allow_html=True)
        risk_data = {
            "Risk Level": ["Low", "Moderate", "High"],
            "Range":      ["0 – 30%", "30 – 60%", "60 – 100%"],
            "Status":     [
                "✅ Safe"     if prob <= 0.30 else "—",
                "⚠️  Moderate" if 0.30 < prob <= 0.60 else "—",
                "🔴 High"    if prob > 0.60 else "—",
            ],
        }
        st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)

        # Quick stats
        st.markdown("<div class='section-hdr'>Model Confidence</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("No Disease", f"{(1-prob):.1%}")
        c2.metric("Disease",    f"{prob:.1%}", delta=f"{prob-0.5:+.1%}")


# ─────────────────────────────────────────────────────────────────
#  TAB 2 — EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("<div class='section-hdr'>Exploratory Data Analysis</div>",
                unsafe_allow_html=True)
    st.caption("All plots generated from the heart disease dataset (n=303).")

    plot_files = [
        ("01_target_distribution.png",  "Target Distribution"),
        ("02_feature_distributions.png","Feature Distributions by Target"),
        ("03_correlation_heatmap.png",  "Correlation Heatmap"),
        ("04_boxplots.png",             "Feature Box-Plots by Target"),
        ("05_pairplot.png",             "Pair-Plot (Key Features)"),
    ]

    for fname, title in plot_files:
        fpath = plot_path(fname)
        if os.path.exists(fpath):
            st.markdown(f"<div class='section-hdr'>{title}</div>", unsafe_allow_html=True)
            st.image(fpath, use_container_width=True)
        else:
            st.warning(f"Plot not found: {fname}. Run `python main.py` to generate it.")


# ─────────────────────────────────────────────────────────────────
#  TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("<div class='section-hdr'>Model Performance Summary</div>",
                unsafe_allow_html=True)

    # Top-level KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",   metrics.get("model_name", "N/A"))
    c2.metric("Accuracy",     f"{metrics.get('accuracy', 0):.2%}")
    c3.metric("ROC-AUC",      f"{metrics.get('roc_auc', 0):.4f}")
    report = metrics.get("report", {})
    c4.metric("Macro F1",     f"{report.get('macro avg', {}).get('f1-score', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_bar, col_img = st.columns([1, 1])

    with col_bar:
        st.markdown("<div class='section-hdr'>Per-Class Metrics</div>",
                    unsafe_allow_html=True)
        if report:
            st.plotly_chart(roc_chart(report), use_container_width=True,
                            config={"displayModeBar": False})

    with col_img:
        st.markdown("<div class='section-hdr'>Confusion Matrix & ROC Curve</div>",
                    unsafe_allow_html=True)
        eval_path = plot_path("07_evaluation.png")
        if os.path.exists(eval_path):
            st.image(eval_path, use_container_width=True)
        else:
            st.warning("Evaluation plot not found. Run `python main.py` first.")

    # Detailed classification report table
    st.markdown("<div class='section-hdr'>Detailed Classification Report</div>",
                unsafe_allow_html=True)
    if report:
        rows = []
        for key in ["0", "1", "macro avg", "weighted avg"]:
            r = report.get(key, {})
            if r:
                rows.append({
                    "Class":     {"0":"No Disease","1":"Disease"}.get(key, key),
                    "Precision": f"{r.get('precision',0):.4f}",
                    "Recall":    f"{r.get('recall',0):.4f}",
                    "F1-Score":  f"{r.get('f1-score',0):.4f}",
                    "Support":   int(r.get("support", 0)),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Model comparison plot
    st.markdown("<div class='section-hdr'>Cross-Validation Model Comparison</div>",
                unsafe_allow_html=True)
    cmp_path = plot_path("06_model_comparison.png")
    if os.path.exists(cmp_path):
        st.image(cmp_path, use_container_width=True)
    else:
        st.warning("Model comparison plot not found. Run `python main.py` first.")


# ─────────────────────────────────────────────────────────────────
#  TAB 4 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────
with tab_feat:
    st.markdown("<div class='section-hdr'>Feature Importances</div>",
                unsafe_allow_html=True)

    fi_path = plot_path("08_feature_importances.png")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)
    else:
        st.warning("Feature importance plot not found. Run `python main.py` first.")

    # Interactive Plotly version using model directly
    try:
        clf  = model.named_steps["clf"]
        pre  = model.named_steps["pre"]
        feat = pre.get_feature_names_out()

        if hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imps = np.abs(clf.coef_[0])
        else:
            raise AttributeError

        idx  = np.argsort(imps)[::-1][:15]
        top_n = feat[idx]
        top_v = imps[idx]

        fig = go.Figure(go.Bar(
            x=top_v[::-1], y=top_n[::-1], orientation="h",
            marker=dict(
                color=top_v[::-1],
                colorscale=[[0, "#56cfe1"], [1, "#ff6b6b"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in top_v[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Top 15 Feature Importances (Interactive)",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis=dict(gridcolor="#30363d", title="Importance"),
            yaxis=dict(gridcolor="#30363d"),
            height=500, margin=dict(t=50, b=20, l=20, r=80),
        )
        st.markdown("<div class='section-hdr'>Interactive Feature Chart</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        # Feature description table
        st.markdown("<div class='section-hdr'>Feature Reference Guide</div>",
                    unsafe_allow_html=True)
        ref = pd.DataFrame([
            ("age",          "Age in years"),
            ("sex",          "1 = Male, 0 = Female"),
            ("cp",           "Chest pain type: 0=Typical Angina … 3=Asymptomatic"),
            ("trestbps",     "Resting blood pressure (mm Hg)"),
            ("chol",         "Serum cholesterol (mg/dL)"),
            ("fbs",          "Fasting blood sugar > 120 mg/dL (1=Yes)"),
            ("restecg",      "Resting ECG: 0=Normal, 1=ST-T wave, 2=LVH"),
            ("thalach",      "Maximum heart rate achieved"),
            ("exang",        "Exercise-induced angina (1=Yes)"),
            ("oldpeak",      "ST depression induced by exercise relative to rest"),
            ("slope",        "Slope of peak exercise ST: 0=Up, 1=Flat, 2=Down"),
            ("ca",           "Number of major vessels coloured by fluoroscopy (0–4)"),
            ("thal",         "Thalassemia: 0=Normal, 1=Fixed, 2=Reversible, 3=Unknown"),
            ("high_thalach", "[Engineered] Max heart rate > 150"),
            ("high_chol",    "[Engineered] Cholesterol > 240 mg/dL"),
            ("age_band",     "[Engineered] Age decade band (0=<40 … 4=70+)"),
            ("oldpeak_severity","[Engineered] ST depression severity (0–3)"),
        ], columns=["Feature", "Description"])
        st.dataframe(ref, hide_index=True, use_container_width=True)

    except Exception:
        st.info("Feature importance details require a tree-based or linear model.")


# ─────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#30363d; margin-top:40px;'>
<p style='text-align:center; color:#8b949e; font-size:0.8rem;'>
  Heart Disease Classifier &nbsp;|&nbsp; Random Forest &nbsp;|&nbsp;
  Dataset: <a href='https://huggingface.co/datasets/mubashir1837/heart-disease'
  style='color:#56cfe1;'>mubashir1837/heart-disease</a> &nbsp;|&nbsp;
  303 patients &nbsp;|&nbsp; ROC-AUC 0.898
</p>
""", unsafe_allow_html=True)
