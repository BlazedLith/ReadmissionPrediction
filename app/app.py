import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path
import sklearn.compose._column_transformer as _sklearn_column_transformer
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


def _patch_sklearn_remainder_cols_list_pickle():
    mod = _sklearn_column_transformer
    if hasattr(mod, "_RemainderColsList"):
        return
    class _RemainderColsList(list):
        pass

    mod._RemainderColsList = _RemainderColsList


_patch_sklearn_remainder_cols_list_pickle()


def _iter_nested_estimators(estimator, seen=None):
    if seen is None:
        seen = set()
    if estimator is None:
        return
    obj_id = id(estimator)
    if obj_id in seen:
        return
    seen.add(obj_id)
    yield estimator

    for attr in ("steps", "transformers", "transformers_"):
        if hasattr(estimator, attr):
            for item in getattr(estimator, attr):
                if isinstance(item, tuple):
                    for part in item:
                        if hasattr(part, "__dict__"):
                            yield from _iter_nested_estimators(part, seen)
                elif hasattr(item, "__dict__"):
                    yield from _iter_nested_estimators(item, seen)

    if hasattr(estimator, "named_steps"):
        for part in estimator.named_steps.values():
            yield from _iter_nested_estimators(part, seen)

    if hasattr(estimator, "named_transformers_"):
        for part in estimator.named_transformers_.values():
            if hasattr(part, "__dict__"):
                yield from _iter_nested_estimators(part, seen)


def _patch_legacy_simple_imputer_pickle(model):
    for est in _iter_nested_estimators(model):
        if not isinstance(est, SimpleImputer):
            continue
        stats = getattr(est, "statistics_", None)
        stats_dtype = getattr(stats, "dtype", None) if stats is not None else None
        fallback_dtype = stats_dtype if stats_dtype is not None else np.dtype("O")
        if not hasattr(est, "_fill_dtype"):
            est._fill_dtype = fallback_dtype
        if not hasattr(est, "_fit_dtype"):
            est._fit_dtype = fallback_dtype

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Diabetic Readmission Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# shared style
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 16px 20px;
        text-align: center;
        color: #212529 !important;
    }
    .metric-card .label { font-size: 13px; color: #495057 !important; margin-bottom: 4px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #212529 !important; }
    .metric-card .sub   { font-size: 12px; color: #495057 !important; }
    .find-box {
        background: #e9ecef !important;
        border-left: 4px solid #495057;
        padding: 10px 14px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 10px;
        font-size: 14px;
        color: #212529 !important;
    }
    h1 { font-size: 1.7rem !important; }
    h2 { font-size: 1.3rem !important; border-bottom: 1px solid #dee2e6; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(OUTPUTS_DIR / "practical_model.pkl")
    _patch_legacy_simple_imputer_pickle(model)
    return model

@st.cache_data
def load_data():
    return pd.read_csv(OUTPUTS_DIR / "diabetes_clean.csv", keep_default_na=False)

@st.cache_data
def load_thresholds():
    t = pd.read_csv(OUTPUTS_DIR / "group_thresholds.csv")
    return dict(zip(t["race"], t["threshold"]))

model = load_model()
df    = load_data()
group_thresholds = load_thresholds()

# ── sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("Diabetic Readmission Prediction")
page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "EDA",
        "Model Performance",
        "Statistical Analysis",
        "Fairness Analysis",
        "Predict",
    ],
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Predicting 30-Day Hospital Readmission in Diabetic Patients")
    st.markdown("""
    This application presents the findings of a comparative machine learning study on 30-day hospital
    readmission prediction in diabetic patients, extending the work of Strack et al. (2014). The study
    uses the UCI Diabetes 130-US Hospitals dataset covering 101,766 patient encounters across 130 US
    hospitals from 1999 to 2008.
    """)

    st.markdown("## Dataset")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in zip(
        [c1, c2, c3, c4],
        ["Original Records", "After Preprocessing", "Features", "Positive Class"],
        ["101,766", "69,973", "32", "9.0%"],
        ["raw encounters", "one per patient", "after engineering", "readmitted < 30 days"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("## What This Study Does")
    st.markdown("""
    The original Strack et al. (2014) paper used only logistic regression to test the HbA1c hypothesis.
    This study extends that work in four ways:

    1. Treats readmission as a full binary classification problem and compares four supervised models:
       Logistic Regression, Decision Tree, Random Forest, and XGBoost.
    2. Handles the severe class imbalance (91% vs 9%) using class-weight balancing.
    3. Runs K-Means clustering to identify natural patient risk groups without using the readmission label.
    4. Evaluates and mitigates racial bias in model predictions using threshold adjustment — the novel
       contribution not present in any prior comparative study on this dataset.
    """)

    st.markdown("## Key Findings")
    findings = [
        "XGBoost and Random Forest are statistically equivalent in AUC (p = 0.93, DeLong test). Decision Tree is significantly worse (p < 0.01).",
        "All models land in AUC 0.627 to 0.644, consistent with the published ceiling of 0.63 to 0.70 for this dataset.",
        "XGBoost is selected for deployment: AUC 0.6438, Recall 55.7%. Higher recall is clinically preferred since missing a high-risk patient is more costly than a false alarm.",
        "K-Means (k=2) identified a high-risk cluster (11.4% readmission rate, 36% of patients) and a low-risk cluster (7.6%, 64% of patients) without access to the readmission label.",
        "Before mitigation, the model showed a 17.4 percentage-point Equal Opportunity Difference across racial groups. After threshold adjustment, this dropped to 1.2 pp with no change in AUC.",
        "Chi-squared test confirms mitigation: TPR disparity across racial groups goes from p = 0.40 to p = 1.00 after adjustment.",
    ]
    for f in findings:
        st.markdown(f'<div class="find-box">{f}</div>', unsafe_allow_html=True)

    st.markdown("## Reference")
    st.markdown("""
    Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014).
    *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database
    Patient Records.* BioMed Research International, 2014, Article 781670.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.markdown("## Target Distribution")
    col1, col2 = st.columns([1, 2])
    with col1:
        counts = df["readmitted"].value_counts()
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:10px">
            <div class="label">Not Readmitted</div>
            <div class="value">{counts[0]:,}</div>
            <div class="sub">{counts[0]/len(df)*100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Readmitted within 30 days</div>
            <div class="value">{counts[1]:,}</div>
            <div class="sub">{counts[1]/len(df)*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Not Readmitted", "Readmitted"], counts.values,
               color=["steelblue", "tomato"], edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title("Target Class Distribution")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("## Numerical Feature Distributions by Readmission")
    num_features = ["time_in_hospital", "num_lab_procedures", "num_medications",
                    "number_inpatient", "number_diagnoses", "total_prior_visits"]

    feat = st.selectbox("Select feature", num_features)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, val, color, ax in zip(
        ["Not Readmitted", "Readmitted"], [0, 1],
        ["steelblue", "tomato"], axes
    ):
        data = df[df["readmitted"] == val][feat]
        ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(f"{feat}\n{label}  (mean={data.mean():.2f}, median={data.median():.0f})")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("## Readmission Rate by Key Categorical Features")
    cat_col = st.selectbox("Select categorical feature", ["race", "age_group", "gender", "A1Cresult"])
    rates = df.groupby(cat_col)["readmitted"].agg(["mean", "count"]).reset_index()
    rates.columns = [cat_col, "rate", "n"]
    rates["rate_pct"] = (rates["rate"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(rates[cat_col].astype(str), rates["rate_pct"],
                  color="steelblue", edgecolor="black")
    ax.axhline(df["readmitted"].mean() * 100, color="tomato",
               linestyle="--", linewidth=1.5, label=f"Overall avg ({df['readmitted'].mean()*100:.1f}%)")
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title(f"Readmission Rate by {cat_col}")
    ax.legend()
    for bar, v in zip(bars, rates["rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v}%", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("## HbA1c Testing Rate")
    a1c = df["A1Cresult"].value_counts()
    tested = df[df["A1Cresult"] != "None"].shape[0]
    total  = len(df)
    col1, col2 = st.columns(2)
    col1.markdown(f"""
    <div class="metric-card">
        <div class="label">HbA1c Tested</div>
        <div class="value">{tested/total*100:.1f}%</div>
        <div class="sub">of encounters — replicates Strack et al. (2014) finding of 18.4%</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""
    <div class="metric-card">
        <div class="label">Not Tested</div>
        <div class="value">{(1 - tested/total)*100:.1f}%</div>
        <div class="sub">despite all patients having a diabetes diagnosis</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("## Correlation with Readmission (Point-Biserial)")
    corr_data = {
        "number_inpatient":    0.1005,
        "total_prior_visits":  0.0842,
        "number_emergency":    0.0655,
        "num_medications":     0.0537,
        "time_in_hospital":    0.0519,
        "num_lab_procedures":  0.0361,
        "number_diagnoses":    0.0301,
        "num_med_changes":     0.0238,
        "number_outpatient":   0.0117,
        "num_procedures":     -0.0001,
    }
    corr_df = pd.DataFrame(list(corr_data.items()), columns=["Feature", "Correlation"])
    corr_df = corr_df.sort_values("Correlation")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if v < 0 else "steelblue" for v in corr_df["Correlation"]]
    ax.barh(corr_df["Feature"], corr_df["Correlation"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Point-Biserial Correlation with 30-Day Readmission")
    ax.set_xlabel("Correlation")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("All correlations are weak (max 0.10), reflecting the hard predictability ceiling of this dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")

    st.markdown("## Model Comparison")
    perf = pd.DataFrame({
        "Model":     ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Soft Voting"],
        "AUC":       [0.6414, 0.6277, 0.6442, 0.6438, 0.6458],
        "Recall":    [0.529,  0.583,  0.208,  0.557,  0.444],
        "Precision": [0.1356, 0.1235, 0.2127, 0.1371, 0.1371],
        "F1":        [0.2158, 0.2038, 0.2103, 0.2201, 0.2252],
    })

    st.dataframe(
        perf.set_index("Model")
        .style.set_properties(**{"color": "#212529", "background-color": "#ffffff"})
        .highlight_max(subset=["AUC", "Recall", "F1"], color="#d4edda")
        .highlight_min(subset=["AUC", "Recall", "F1"], color="#f8d7da"),
        width="stretch",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    perf_plot = perf[perf["Model"] != "Soft Voting"].copy()
    x = np.arange(len(perf_plot))
    width = 0.2
    metrics = ["AUC", "Recall", "F1", "Precision"]
    colors  = ["steelblue", "darkorange", "seagreen", "tomato"]

    for i, (m, c) in enumerate(zip(metrics, colors)):
        axes[0].bar(x + i * width, perf_plot[m], width, label=m, color=c, edgecolor="black")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(perf_plot["Model"], rotation=15, ha="right")
    axes[0].set_title("All Metrics by Model")
    axes[0].legend()

    axes[1].bar(perf_plot["Model"], perf_plot["AUC"],
                color="steelblue", edgecolor="black")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random baseline")
    axes[1].set_ylim(0.5, 0.70)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC Comparison")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend()
    for i, v in enumerate(perf_plot["AUC"]):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("## Literature Benchmark")
    lit = pd.DataFrame({
        "Study": ["This Study (XGBoost)", "This Study (Ensemble)",
                  "Cureus 2025 (XGBoost)", "Gandra 2024 (CATBoost)",
                  "Liu et al. 2024 (XGBoost)", "Strack et al. 2014 (LR)"],
        "Best AUC": ["0.6438", "0.6458", "0.667", "0.70", "0.65-0.70", "N/A"],
        "Notes": [
            "Deployment model, recall 55.7%",
            "4-model soft vote",
            "Same UCI dataset, 80/20 split",
            "More advanced boosting models",
            "10-model comparison",
            "Original paper, HbA1c hypothesis"
        ]
    })
    st.dataframe(lit.set_index("Study"), width="stretch")
    st.caption("Our results sit within the published range. Studies exceeding AUC 0.67 use LightGBM or CATBoost, which are outside this study's scope.")

    st.markdown("## Why XGBoost for Deployment")
    col1, col2 = st.columns(2)
    col1.markdown("""
    **Random Forest** has the highest single-model AUC (0.6442) but **Recall = 0.208**.
    It only catches 1 in 5 actual readmissions. In a clinical setting, missing 79% of
    high-risk patients makes the tool nearly useless.
    """)
    col2.markdown("""
    **XGBoost** is within 0.0004 AUC of Random Forest and catches **55.7% of actual
    readmissions** — more than double. In clinical prediction, the cost of a missed
    patient is greater than the cost of a false alarm.
    """)

    st.markdown("## Feature Importance (Random Forest)")
    feat_imp = {
        "num_lab_procedures": 0.1021, "num_medications": 0.0908,
        "time_in_hospital": 0.0662,  "number_diagnoses": 0.0481,
        "discharge_disposition_id_1": 0.0392, "total_prior_visits": 0.0293,
        "number_inpatient": 0.0286,  "discharge_disposition_id_22": 0.0241,
        "num_med_changes": 0.0198,   "number_outpatient": 0.0187,
    }
    fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="steelblue", edgecolor="black")
    ax.set_title("Top 10 Feature Importances (Random Forest)")
    ax.set_xlabel("Mean Gini Decrease")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Statistical Analysis":
    st.title("Statistical Analysis")

    st.markdown("## DeLong Test: Pairwise AUC Comparison")
    delong = pd.DataFrame({
        "Model A":    ["Logistic Regression", "Logistic Regression", "Logistic Regression",
                       "Decision Tree", "Decision Tree", "Random Forest"],
        "Model B":    ["Decision Tree", "Random Forest", "XGBoost",
                       "Random Forest", "XGBoost", "XGBoost"],
        "AUC A":      [0.6414, 0.6414, 0.6414, 0.6277, 0.6277, 0.6442],
        "AUC B":      [0.6277, 0.6442, 0.6438, 0.6442, 0.6438, 0.6438],
        "AUC Diff":   [0.0137, -0.0028, -0.0024, -0.0165, -0.0161, 0.0004],
        "Z-stat":     [2.6034, -0.6483, -0.7375, -2.9454, -3.2490, 0.0931],
        "P-value":    [0.0092, 0.5168, 0.4608, 0.0032, 0.0012, 0.9258],
        "Significant":[  "Yes",   "No",   "No",   "Yes",  "Yes",   "No"],
    })
    st.dataframe(delong.set_index("Model A"), width="stretch")

    fig, ax = plt.subplots(figsize=(9, 4))
    pairs  = [f"{r['Model A']} vs\n{r['Model B']}" for _, r in delong.iterrows()]
    diffs  = delong["AUC Diff"].values
    colors = ["tomato" if s == "Yes" else "steelblue" for s in delong["Significant"]]
    bars   = ax.barh(pairs, diffs, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, p in zip(bars, delong["P-value"]):
        label = f"p={p}" if p >= 0.001 else "p<0.001"
        x_pos = bar.get_width() + (0.0003 if bar.get_width() >= 0 else -0.003)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    red   = mpatches.Patch(color="tomato",    label="Significant (p<0.05)")
    blue  = mpatches.Patch(color="steelblue", label="Not significant")
    ax.legend(handles=[red, blue])
    ax.set_xlabel("AUC Difference (A minus B)")
    ax.set_title("DeLong Test: Pairwise AUC Differences")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Key result:** Random Forest, XGBoost, and Logistic Regression are statistically equivalent
    (all pairwise p > 0.46). Decision Tree is significantly worse than all three (p < 0.01).
    The numeric AUC differences among the top three models are noise, not signal.
    """)

    st.markdown("## Mann-Whitney U Test: Features vs Readmission")
    mw = pd.DataFrame({
        "Feature": ["time_in_hospital", "num_lab_procedures", "num_medications",
                    "number_emergency", "number_diagnoses", "number_inpatient",
                    "total_prior_visits", "num_med_changes", "number_outpatient"],
        "Mean (Not Readmitted)": [4.222, 42.675, 15.571, 0.099, 7.195, 0.157, 0.408, 1.343, 0.152],
        "Mean (Readmitted)":     [4.797, 44.915, 16.626, 0.150, 7.513, 0.369, 0.770, 1.421, 0.251],
        "P-value": ["<0.0001"] * 9,
        "Significant": ["Yes"] * 9,
    })
    st.dataframe(mw.set_index("Feature"), width="stretch")
    st.caption("All 9 numerical features are significantly associated with 30-day readmission (p < 0.0001).")

    st.markdown("## Kruskal-Wallis Test: Feature Distributions Across Racial Groups")
    kw = pd.DataFrame({
        "Feature": ["number_diagnoses", "num_medications", "number_inpatient",
                    "time_in_hospital", "num_lab_procedures"],
        "H-statistic": [585.704, 417.103, 61.076, 54.460, 22.727],
        "P-value": ["<0.0001", "<0.0001", "<0.0001", "<0.0001", "0.000144"],
        "Significant": ["Yes"] * 5,
    })
    st.dataframe(kw.set_index("Feature"), width="stretch")
    st.caption("All five features differ significantly across racial groups. This statistically confirms that racial groups have different clinical profiles, providing a mechanism for why the model produces biased predictions.")

    st.markdown("## Probability Distribution Fitting")
    col1, col2 = st.columns(2)
    fitting = {
        "num_lab_procedures": [
            ("Weibull", 0.0368), ("Normal", 0.0448),
            ("Lognormal", 0.0448), ("Gamma", 0.0528), ("Exponential", 0.2679)
        ],
        "time_in_hospital": [
            ("Exponential", 0.1496), ("Normal", 0.1759),
            ("Gamma", 0.1810), ("Weibull", 0.3058), ("Lognormal", 0.4814)
        ]
    }
    for col, (feat, fits) in zip([col1, col2], fitting.items()):
        fit_df = pd.DataFrame(fits, columns=["Distribution", "KS Statistic"])
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["tomato"] + ["steelblue"] * 4
        ax.bar(fit_df["Distribution"], fit_df["KS Statistic"], color=colors, edgecolor="black")
        ax.set_title(f"{feat}\n(red = best fit)")
        ax.set_ylabel("KS Statistic (lower = better)")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        col.pyplot(fig)
        plt.close()
    st.caption("No standard parametric distribution fits either feature well (all p = 0.0). Both are discrete and bounded, making continuous distribution approximations imperfect by design.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAIRNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fairness Analysis":
    st.title("Fairness Analysis")

    st.markdown("""
    This section evaluates whether the XGBoost model produces equitable predictions across racial groups.
    A model can have acceptable overall AUC while systematically missing high-risk patients from certain
    groups, which is clinically harmful. The mitigation method used is post-processing threshold adjustment:
    group-specific decision thresholds are applied so that each group's True Positive Rate (recall) is
    brought to the overall model TPR. The model itself is not retrained.
    """)

    st.markdown("## Actual Readmission Rates by Race")
    actual = pd.DataFrame({
        "Race": ["AfricanAmerican", "Caucasian", "Hispanic", "Asian", "Other"],
        "Actual Readmission %": [10.3, 8.8, 8.2, 7.5, 8.1],
        "N (test set)": [1392, 10782, 362, 149, 213],
    })
    st.dataframe(actual.set_index("Race"), width="stretch")
    st.caption("AfricanAmerican patients have a genuinely higher readmission rate. Differences in prediction rates reflecting this are not bias. Bias is when the model makes more errors on one group than another, not when raw rates differ.")

    st.markdown("## Before vs After Mitigation")
    fairness = pd.DataFrame({
        "Race": ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"],
        "TPR Before": [0.516, 0.636, 0.562, 0.690, 0.545],
        "TPR After":  [0.556, 0.545, 0.558, 0.552, 0.545],
        "FPR Before": [0.536, 0.347, 0.471, 0.440, 0.480],
        "FPR After":  [0.578, 0.312, 0.467, 0.389, 0.480],
        "PPR Before": [0.519, 0.376, 0.477, 0.469, 0.488],
        "PPR After":  [0.558, 0.336, 0.474, 0.398, 0.488],
    })
    st.dataframe(fairness.set_index("Race"), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""<div class="metric-card">
        <div class="label">EOD Before</div>
        <div class="value">17.4 pp</div>
        <div class="sub">Equal Opportunity Difference</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown("""<div class="metric-card">
        <div class="label">EOD After</div>
        <div class="value">1.2 pp</div>
        <div class="sub">After threshold adjustment</div>
    </div>""", unsafe_allow_html=True)
    col3.markdown("""<div class="metric-card">
        <div class="label">AUC Change</div>
        <div class="value">0.000</div>
        <div class="sub">No degradation in model performance</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    races = fairness["Race"].tolist()
    x     = np.arange(len(races))
    w     = 0.35

    for ax, metric, title in zip(
        axes,
        [("TPR Before", "TPR After"), ("FPR Before", "FPR After")],
        ["True Positive Rate (Recall) by Race", "False Positive Rate by Race"]
    ):
        ax.bar(x - w/2, fairness[metric[0]], w, label="Before", color="steelblue", edgecolor="black")
        ax.bar(x + w/2, fairness[metric[1]], w, label="After",  color="darkorange", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(races, rotation=15, ha="right")
        ax.set_title(title)
        ax.legend()
    plt.suptitle("Fairness Metrics Before vs After Threshold Adjustment", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("## Chi-Squared Test on Fairness Improvement")
    chi2_data = pd.DataFrame({
        "": ["Before Mitigation", "After Mitigation"],
        "Chi2": [4.0124, 0.0191],
        "P-value": [0.4043, 1.0000],
        "Interpretation": [
            "No significant group-level disparity detected, but directional differences existed",
            "TPR distribution across races is statistically uniform"
        ]
    })
    st.dataframe(chi2_data.set_index(""), width="stretch")

    st.markdown("""
    **Interpretation:** The chi-squared p-value after mitigation of 1.00 means the adjusted TPR
    distribution across racial groups is indistinguishable from perfectly uniform. This is the strongest
    result in the fairness analysis. The mitigation did not require retraining the model and produced
    zero AUC cost.

    This is the first analysis on this dataset to apply formal fairness metrics and post-processing
    mitigation to the four standard ML models used in comparative studies. Ding and Shah (2022) identified
    the bias directionally but provided no statistical proof and no mitigation.
    """)

    st.markdown("## Group-Specific Thresholds Used")
    thresh_df = pd.DataFrame(list(group_thresholds.items()), columns=["Race", "Threshold"])
    thresh_df["Threshold"] = thresh_df["Threshold"].round(4)
    st.dataframe(thresh_df.set_index("Race"), width="stretch")
    st.caption("Thresholds below 0.5 mean the model is more sensitive for that group (flags at lower probability). The prediction page uses these thresholds when a racial group is selected.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Readmission Risk Prediction")
    st.markdown("""
    Enter patient information below to get a 30-day readmission risk prediction from the deployed
    XGBoost model. When a racial group is selected, the group-specific threshold from the fairness
    analysis is applied so the prediction reflects the bias-adjusted decision boundary.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age_group = st.selectbox("Age Group", ["young (0-30)", "middle (30-60)", "senior (60-100)"])

        st.markdown("**Admission Info**")
        admission_type_id = st.selectbox("Admission Type", ["1", "2", "3", "4", "5", "6", "7", "8"])
        discharge_disposition_id = st.selectbox("Discharge Disposition",
            [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                               16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30]])
        admission_source_id = st.selectbox("Admission Source", [str(i) for i in range(1, 26)])

    with col2:
        st.markdown("**Hospital Stay**")
        time_in_hospital    = st.slider("Days in Hospital", 1, 14, 4)
        num_lab_procedures  = st.slider("Lab Procedures", 1, 132, 43)
        num_medications     = st.slider("Number of Medications", 1, 81, 15)
        number_diagnoses    = st.slider("Number of Diagnoses", 1, 16, 7)
        specialty_known     = st.selectbox("Medical Specialty Known", [1, 0],
                                           format_func=lambda x: "Yes" if x else "No")

        st.markdown("**Lab Results**")
        A1Cresult    = st.selectbox("HbA1c Result", ["None", "Norm", ">7", ">8"])
        max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])

    with col3:
        st.markdown("**Prior Visit History**")
        number_outpatient = st.slider("Prior Outpatient Visits", 0, 42, 0)
        number_emergency  = st.slider("Prior Emergency Visits",  0, 76, 0)
        number_inpatient  = st.slider("Prior Inpatient Visits",  0, 21, 0)

        st.markdown("**Medications**")
        insulin     = st.selectbox("Insulin",    ["No", "Steady", "Up", "Down"])
        metformin   = st.selectbox("Metformin",  ["No", "Steady", "Up", "Down"])
        glipizide   = st.selectbox("Glipizide",  ["No", "Steady", "Up", "Down"])
        glyburide   = st.selectbox("Glyburide",  ["No", "Steady", "Up", "Down"])
        glimepiride = st.selectbox("Glimepiride",["No", "Steady", "Up", "Down"])
        pioglitazone= st.selectbox("Pioglitazone",["No","Steady","Up","Down"])
        rosiglitazone=st.selectbox("Rosiglitazone",["No","Steady","Up","Down"])
        repaglinide = st.selectbox("Repaglinide",["No", "Steady", "Up", "Down"])

    # derived features
    total_prior_visits = number_outpatient + number_emergency + number_inpatient
    med_changes        = [insulin, metformin, glipizide, glyburide,
                          glimepiride, pioglitazone, rosiglitazone, repaglinide]
    num_med_changes    = sum(1 for m in med_changes if m in ["Up", "Down"])
    A1C_tested         = 0 if A1Cresult == "None" else 1

    # primary diagnosis placeholder (most common)
    diag_1 = "Circulatory"
    diag_2 = "Diabetes"
    diag_3 = "Other"

    input_dict = {
        "race": race, "gender": gender, "age_group": age_group,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "specialty_known": specialty_known,
        "A1Cresult": A1Cresult,
        "max_glu_serum": max_glu_serum,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "total_prior_visits": total_prior_visits,
        "num_med_changes": num_med_changes,
        "A1C_tested": A1C_tested,
        "insulin": insulin, "metformin": metformin,
        "glipizide": glipizide, "glyburide": glyburide,
        "glimepiride": glimepiride, "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone, "repaglinide": repaglinide,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
    }

    if st.button("Predict Readmission Risk", type="primary"):
        input_df = pd.DataFrame([input_dict])

        # add any missing columns with defaults
        reference_cols = [c for c in df.columns if c not in ["readmitted", "num_procedures", "cluster"]]
        for col in reference_cols:
            if col not in input_df.columns:
                sample_val = df[col].mode()[0] if df[col].dtype == "object" else 0
                input_df[col] = sample_val

        try:
            prob = model.predict_proba(input_df[reference_cols])[0][1]

            # apply group-specific threshold from fairness analysis
            threshold = group_thresholds.get(race, 0.5)
            prediction = int(prob >= threshold)

            st.markdown("---")
            st.markdown("## Prediction Result")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="metric-card">
                <div class="label">Readmission Probability</div>
                <div class="value">{prob*100:.1f}%</div>
                <div class="sub">raw model score</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-card">
                <div class="label">Decision Threshold</div>
                <div class="value">{threshold:.3f}</div>
                <div class="sub">fairness-adjusted for {race}</div>
            </div>""", unsafe_allow_html=True)
            risk_label = "High Risk" if prediction == 1 else "Low Risk"
            risk_color = "#dc3545" if prediction == 1 else "#198754"
            c3.markdown(f"""<div class="metric-card">
                <div class="label">Prediction</div>
                <div class="value" style="color:{risk_color}">{risk_label}</div>
                <div class="sub">readmission within 30 days</div>
            </div>""", unsafe_allow_html=True)

            # cleaner probability display (avoids awkward matplotlib gauge styling)
            st.markdown("### Readmission Probability vs Decision Threshold")
            st.progress(float(prob), text=f"Model probability: {prob:.1%}")
            st.caption(
                f"Decision threshold for {race}: {threshold:.3f} "
                f"({risk_label} when probability >= threshold)."
            )

            st.markdown("### What Drove This Score")
            factors = []
            if number_inpatient > 1:
                factors.append(f"Prior inpatient visits = {number_inpatient} (mean readmitted: 0.37, not readmitted: 0.16)")
            if time_in_hospital > 5:
                factors.append(f"Hospital stay = {time_in_hospital} days (mean readmitted: 4.8, not readmitted: 4.2)")
            if num_medications > 18:
                factors.append(f"Medications = {num_medications} (mean readmitted: 16.6, not readmitted: 15.6)")
            if total_prior_visits > 2:
                factors.append(f"Total prior visits = {total_prior_visits} (mean readmitted: 0.77, not readmitted: 0.41)")
            if A1C_tested == 0:
                factors.append("HbA1c not tested during this admission (Strack et al. found testing reduces readmission when medication is adjusted)")
            if discharge_disposition_id == "22":
                factors.append("Discharge to rehab facility (associated with higher readmission risk)")

            if factors:
                st.markdown("Factors in this patient's profile that are associated with higher readmission risk:")
                for f in factors:
                    st.markdown(f'<div class="find-box">{f}</div>', unsafe_allow_html=True)
            else:
                st.markdown("No strong individual risk factors detected. Risk is driven by combined feature interactions.")

            st.caption("""
            **Disclaimer:** This tool is for research demonstration only. It is not a clinical decision support
            system and should not be used to make treatment decisions. Model overall recall is 55.7%, meaning
            roughly 44% of high-risk patients will not be flagged.
            """)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Ensure practical_model.pkl and diabetes_clean.csv are present in the outputs/ folder.")
    st.markdown("---")
    st.markdown("## Batch Prediction from CSV")
    st.markdown("""
    Upload a CSV file with patient records to get predictions for each row.
    The model will output the readmission probability, the fairness-adjusted decision,
    and the threshold used per patient based on their racial group.
    """)
 
    # show expected columns
    with st.expander("Show expected CSV columns"):
        expected_cols = {
            "race": "Caucasian / AfricanAmerican / Hispanic / Asian / Other",
            "gender": "Male / Female",
            "age_group": "young (0-30) / middle (30-60) / senior (60-100)",
            "admission_type_id": "1-8 (as string)",
            "discharge_disposition_id": "1-30 (as string)",
            "admission_source_id": "1-25 (as string)",
            "time_in_hospital": "1-14 (integer)",
            "num_lab_procedures": "1-132 (integer)",
            "num_medications": "1-81 (integer)",
            "number_diagnoses": "1-16 (integer)",
            "specialty_known": "1 or 0",
            "A1Cresult": "None / Norm / >7 / >8",
            "max_glu_serum": "None / Norm / >200 / >300",
            "number_outpatient": "0+ (integer)",
            "number_emergency": "0+ (integer)",
            "number_inpatient": "0+ (integer)",
            "insulin": "No / Steady / Up / Down",
            "metformin": "No / Steady / Up / Down",
            "glipizide": "No / Steady / Up / Down",
            "glyburide": "No / Steady / Up / Down",
            "glimepiride": "No / Steady / Up / Down",
            "pioglitazone": "No / Steady / Up / Down",
            "rosiglitazone": "No / Steady / Up / Down",
            "repaglinide": "No / Steady / Up / Down",
            "diag_1": "Circulatory / Respiratory / Digestive / Diabetes / Injury / Musculoskeletal / Genitourinary / Neoplasms / Other",
            "diag_2": "same as diag_1",
            "diag_3": "same as diag_1",
        }
        col_df = pd.DataFrame([
            {"Column": k, "Expected Values": v}
            for k, v in expected_cols.items()
        ])
        st.dataframe(col_df.set_index("Column"), width="stretch")
 
        # download template
        template_row = {
            "race": "Caucasian",
            "gender": "Male",
            "age_group": "senior (60-100)",
            "admission_type_id": "1",
            "discharge_disposition_id": "1",
            "admission_source_id": "7",
            "time_in_hospital": 5,
            "num_lab_procedures": 45,
            "num_medications": 15,
            "number_diagnoses": 7,
            "specialty_known": 1,
            "A1Cresult": "None",
            "max_glu_serum": "None",
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "insulin": "Steady",
            "metformin": "No",
            "glipizide": "No",
            "glyburide": "No",
            "glimepiride": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "repaglinide": "No",
            "diag_1": "Circulatory",
            "diag_2": "Diabetes",
            "diag_3": "Other",
        }
        template_df = pd.DataFrame([template_row])
        st.download_button(
            label="Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="patient_template.csv",
            mime="text/csv"
        )
 
    uploaded_csv = st.file_uploader("Upload patient CSV", type=["csv"])
 
    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv, keep_default_na=False)
            st.write(f"Loaded {len(batch_df)} rows.")
 
            # derive features if not present
            if "total_prior_visits" not in batch_df.columns:
                batch_df["total_prior_visits"] = (
                    batch_df["number_outpatient"] +
                    batch_df["number_emergency"] +
                    batch_df["number_inpatient"]
                )
 
            if "num_med_changes" not in batch_df.columns:
                med_cols = ["insulin", "metformin", "glipizide", "glyburide",
                            "glimepiride", "pioglitazone", "rosiglitazone", "repaglinide"]
                batch_df["num_med_changes"] = batch_df[med_cols].apply(
                    lambda row: sum(1 for v in row if v in ["Up", "Down"]), axis=1
                )
 
            if "A1C_tested" not in batch_df.columns:
                batch_df["A1C_tested"] = (batch_df["A1Cresult"] != "None").astype(int)
 
            # fill any missing columns with mode from training data
            reference_cols = [c for c in df.columns if c not in ["readmitted", "num_procedures", "cluster"]]
            for col in reference_cols:
                if col not in batch_df.columns:
                    fill_val = df[col].mode()[0] if df[col].dtype == "object" else 0
                    batch_df[col] = fill_val
 
            # get probabilities
            probs = model.predict_proba(batch_df[reference_cols])[:, 1]
 
            # apply group-specific thresholds
            thresholds_applied = batch_df["race"].map(group_thresholds).fillna(0.5)
            predictions = (probs >= thresholds_applied).astype(int)
 
            # build output
            result_df = batch_df[["race"] + [
                c for c in ["gender", "age_group", "time_in_hospital",
                             "num_medications", "number_inpatient", "total_prior_visits"]
                if c in batch_df.columns
            ]].copy()
 
            result_df["readmission_probability"] = (probs * 100).round(1)
            result_df["threshold_used"] = thresholds_applied.round(3)
            result_df["prediction"] = predictions
            result_df["risk"] = result_df["prediction"].map({1: "High Risk", 0: "Low Risk"})
 
            st.markdown("### Prediction Results")
 
            # summary stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Patients", len(result_df))
            c2.metric("Flagged High Risk", int(predictions.sum()))
            c3.metric("Flag Rate", f"{predictions.mean()*100:.1f}%")
 
            # color risk column
            def highlight_risk(val):
                if val == "High Risk":
                    return "background-color: #f8d7da; color: #721c24"
                return "background-color: #d4edda; color: #155724"
 
            max_styler_cells = 262_144
            table_cells = result_df.shape[0] * result_df.shape[1]
            if table_cells <= max_styler_cells:
                st.dataframe(
                    result_df.style.map(highlight_risk, subset=["risk"]),
                    width="stretch",
                )
            else:
                st.info(
                    "Large file detected: showing an unstyled table for performance. "
                    "Download CSV for full results."
                )
                st.dataframe(result_df, width="stretch")
 
            # probability distribution chart
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(probs * 100, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
            ax.axvline(50, color="tomato", linestyle="--", linewidth=1.5, label="Default threshold (50%)")
            ax.set_xlabel("Predicted Readmission Probability (%)")
            ax.set_ylabel("Number of Patients")
            ax.set_title("Distribution of Predicted Probabilities")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            # download results
            st.download_button(
                label="Download Results CSV",
                data=result_df.to_csv(index=False),
                file_name="readmission_predictions.csv",
                mime="text/csv"
            )
 
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            st.info("Make sure your CSV has the required columns. Download the template above to see the expected format.")
