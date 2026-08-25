from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "train_data.csv"
TARGET_COL = "target"

st.set_page_config(
    page_title="Diabetes Risk XAI",
    page_icon="🩺",
    layout="wide",
)

# ============================================================
# Feature groups used by the uploaded model/data
# ============================================================
NUMERIC_FEATURES = [
    "Age", "Height_cm", "Weight_kg", "BMI", "Waist_Circumference_cm",
    "Blood_Glucose", "HbA1c", "Fasting_Blood_Sugar", "Insulin_Level",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Total_Cholesterol",
    "HDL", "LDL", "Triglycerides", "Heart_Rate", "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes", "Sleep_Hours", "Daily_Water_Intake_L",
]

GROUPS = {
    "Gender": (
        ["Gender_Male", "Gender_Other"],
        ["Male", "Other"],
        "Female",
    ),
    "Physical Activity Level": (
        ["Physical_Activity_Level_Low", "Physical_Activity_Level_Moderate"],
        ["Low", "Moderate"],
        "High",
    ),
    "Diet Quality": (
        ["Diet_Quality_Healthy", "Diet_Quality_Poor"],
        ["Healthy", "Poor"],
        "Average",
    ),
    "Sugar Intake Level": (
        ["Sugar_Intake_Level_Low", "Sugar_Intake_Level_Moderate"],
        ["Low", "Moderate"],
        "High",
    ),
    "Stress Level": (
        ["Stress_Level_Low", "Stress_Level_Moderate"],
        ["Low", "Moderate"],
        "High",
    ),
    "Smoking Status": (
        ["Smoking_Status_Former", "Smoking_Status_Never"],
        ["Former", "Never"],
        "Current",
    ),
    "Alcohol Consumption": (
        ["Alcohol_Consumption_Never", "Alcohol_Consumption_Occasionally"],
        ["Never", "Occasionally"],
        "Frequently",
    ),
    "Medication Adherence": (
        ["Medication_Adherence_Good", "Medication_Adherence_Poor"],
        ["Good", "Poor"],
        "Moderate",
    ),
    "Work Type": (
        ["Work_Type_Government", "Work_Type_Private", "Work_Type_Retired", "Work_Type_Student"],
        ["Government", "Private", "Retired", "Student"],
        "Unemployed",
    ),
}

COUNTRY_COLS = [
    "Country_Australia", "Country_Bangladesh", "Country_Brazil", "Country_Canada",
    "Country_China", "Country_Egypt", "Country_France", "Country_Germany",
    "Country_India", "Country_Indonesia", "Country_Italy", "Country_Japan",
    "Country_Malaysia", "Country_Mexico", "Country_Nigeria", "Country_Pakistan",
    "Country_Russia", "Country_Saudi Arabia", "Country_South Africa",
    "Country_South Korea", "Country_Spain", "Country_Turkey",
    "Country_United Kingdom", "Country_United States",
]

BINARY_FLAGS = {
    "Family_History_Diabetes_Yes": "Family history of diabetes",
    "Hypertension_Yes": "Hypertension",
    "Heart_Disease_Yes": "Heart disease",
    "Fatty_Liver_Yes": "Fatty liver",
    "PCOS_Yes": "PCOS",
    "Residence_Type_Urban": "Lives in an urban area",
}

# Features allowed to change in the custom counterfactual search.
ACTIONABLE_NUMERIC = [
    "Weight_kg", "BMI", "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes", "Sleep_Hours", "Daily_Water_Intake_L",
]

ACTIONABLE_GROUPS = {
    "Physical Activity Level",
    "Diet Quality",
    "Sugar Intake Level",
    "Stress Level",
    "Smoking Status",
    "Alcohol Consumption",
    "Medication Adherence",
}

RISK_ORDER = {"Low": 0, "Moderate": 1, "High": 2}

# ============================================================
# Cached loading: NO SHAP / DiCE initialization here
# ============================================================
@st.cache_resource

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data

def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column is missing from train_data.csv")
    X = df.drop(columns=[TARGET_COL])
    return df, X, list(X.columns)


@st.cache_data

def get_numeric_ranges(X):
    ranges = {}
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(s):
                ranges[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "q10": float(s.quantile(0.10)),
                    "q25": float(s.quantile(0.25)),
                    "q50": float(s.quantile(0.50)),
                    "q75": float(s.quantile(0.75)),
                    "q90": float(s.quantile(0.90)),
                }
    return ranges


def friendly_name(col):
    for prefix in [
        "Country_", "Gender_", "Physical_Activity_Level_", "Diet_Quality_",
        "Sugar_Intake_Level_", "Stress_Level_", "Smoking_Status_",
        "Alcohol_Consumption_", "Medication_Adherence_", "Work_Type_",
    ]:
        if col.startswith(prefix):
            group = prefix.rstrip("_").replace("_", " ")
            return f"{group}: {col[len(prefix):]}"
    if col in BINARY_FLAGS:
        return BINARY_FLAGS[col]
    return col.replace("_", " ")


def build_input_row(numeric_values, group_choices, country_choice, flag_values, feature_order, X):
    row = {c: 0 for c in feature_order}
    row.update(numeric_values)

    for label, (dummy_cols, options, baseline) in GROUPS.items():
        choice = group_choices[label]
        for c in dummy_cols:
            row[c] = False
        if choice in options:
            row[dummy_cols[options.index(choice)]] = True

    for c in COUNTRY_COLS:
        if c in row:
            row[c] = False
    country_col = f"Country_{country_choice}"
    if country_col in row:
        row[country_col] = True

    for col in BINARY_FLAGS:
        if col in row:
            row[col] = bool(flag_values[col])

    result = pd.DataFrame([row], columns=feature_order)

    # Match the training dtypes exactly where possible.
    for c in feature_order:
        if c in X.columns and pd.api.types.is_bool_dtype(X[c]):
            result[c] = result[c].astype(bool)
        elif c in X.columns:
            result[c] = pd.to_numeric(result[c], errors="coerce").astype(float)
    return result


def native_catboost_shap(model, input_df, feature_order, class_idx):
    """Return SHAP values for one predicted class using CatBoost native SHAP.

    CatBoost multiclass output is (rows, classes, features + 1).
    The final value is the expected value/base value, so it is removed.
    """
    pool = Pool(input_df, feature_names=feature_order)
    raw = np.asarray(model.get_feature_importance(pool, type="ShapValues"))

    if raw.ndim != 3:
        raise ValueError(f"Unexpected CatBoost SHAP shape: {raw.shape}")

    values = raw[0, class_idx, :-1]
    base_value = float(raw[0, class_idx, -1])
    return values, base_value


def make_shap_chart(top5, pred_class):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = top5.values.astype(float)
    labels = [friendly_name(x) for x in top5.index]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to the predicted class score")
    ax.set_title(f"Top features pushing toward/away from '{pred_class}'")
    fig.tight_layout()
    return fig


def group_value_from_row(row, group_name):
    dummy_cols, options, baseline = GROUPS[group_name]
    for c, option in zip(dummy_cols, options):
        if bool(row.get(c, False)):
            return option
    return baseline


def set_group_value(row, group_name, value):
    dummy_cols, options, baseline = GROUPS[group_name]
    for c in dummy_cols:
        row[c] = False
    if value in options:
        row[dummy_cols[options.index(value)]] = True


def set_group_value_in_df(df, row_index, group_name, value):
    dummy_cols, options, baseline = GROUPS[group_name]
    for c in dummy_cols:
        df.at[row_index, c] = False
    if value in options:
        df.at[row_index, dummy_cols[options.index(value)]] = True


def counterfactual_search(model, original, X, predicted_class, n_results=3, seed=42):
    """Generate model-based actionable counterfactuals.

    It searches only over lifestyle variables and keeps age, medical history,
    country and other non-actionable variables fixed.
    """
    rng = np.random.default_rng(seed)
    ranges = get_numeric_ranges(X)

    desired = "Low" if predicted_class in {"High", "Moderate"} else "Moderate"
    candidates = []

    # Candidate numeric values come from quantiles plus small perturbations.
    numeric_choices = {}
    for col in ACTIONABLE_NUMERIC:
        if col not in ranges or col not in original.columns:
            continue
        r = ranges[col]
        current = float(original.iloc[0][col])
        vals = [r["q10"], r["q25"], r["q50"], r["q75"], r["q90"]]
        vals += [current * (1 - d) for d in (0.05, 0.10, 0.15)]
        vals += [current * (1 + d) for d in (0.05, 0.10)]
        vals = [float(np.clip(v, r["min"], r["max"])) for v in vals]
        numeric_choices[col] = sorted(set(round(v, 3) for v in vals))

    # Random single/group changes.
    for _ in range(1800):
        cand = original.copy()

        # Change 1-3 actionable variables/groups.
        n_changes = int(rng.integers(1, 4))
        chosen_numeric = list(rng.choice(ACTIONABLE_NUMERIC, size=min(n_changes, len(ACTIONABLE_NUMERIC)), replace=False))

        for col in chosen_numeric:
            if col in numeric_choices and numeric_choices[col]:
                cand.at[cand.index[0], col] = rng.choice(numeric_choices[col])

        if rng.random() < 0.70:
            group = rng.choice(sorted(ACTIONABLE_GROUPS))
            _, options, baseline = GROUPS[group]
            current = group_value_from_row(cand.iloc[0], group)
            choices = [x for x in options + [baseline] if x != current]
            if choices:
                set_group_value_in_df(cand, cand.index[0], group, rng.choice(choices))

        candidates.append(cand)

    # Add deterministic quantile candidates so results are reproducible/useful.
    base = original.copy()
    for col in ACTIONABLE_NUMERIC:
        if col in numeric_choices:
            for value in numeric_choices[col][:5]:
                c = base.copy()
                c.at[c.index[0], col] = value
                candidates.append(c)

    if not candidates:
        return pd.DataFrame(), desired

    all_candidates = pd.concat(candidates, ignore_index=True)
    all_candidates = all_candidates.drop_duplicates().reset_index(drop=True)

    # Restore the exact dtypes expected by CatBoost.
    feature_order_from_x = list(X.columns)
    for c in feature_order_from_x:
        if pd.api.types.is_bool_dtype(X[c]):
            all_candidates[c] = all_candidates[c].astype(bool)
        else:
            all_candidates[c] = pd.to_numeric(all_candidates[c], errors="coerce").astype(float)

    all_candidates = all_candidates[feature_order_from_x]
    probs = model.predict_proba(all_candidates)
    classes = list(model.classes_)
    desired_idx = classes.index(desired) if desired in classes else None
    pred = model.predict(all_candidates).reshape(-1)

    valid = np.where(pred == desired)[0]
    if len(valid) == 0:
        # If Low is unreachable, try any lower-risk class than the prediction.
        valid = np.where([RISK_ORDER.get(str(p), 99) < RISK_ORDER.get(predicted_class, 99) for p in pred])[0]

    if len(valid) == 0:
        return pd.DataFrame(), desired

    # Score candidates: few changes first, then high desired probability.
    original_row = original.iloc[0]
    rows = []
    for idx in valid:
        row = all_candidates.iloc[idx]
        changed = []
        distance = 0.0

        for col in ACTIONABLE_NUMERIC:
            if col in row.index:
                old = float(original_row[col])
                new = float(row[col])
                if abs(old - new) > 1e-8:
                    changed.append((col, old, new))
                    scale = max(abs(old), 1.0)
                    distance += abs(new - old) / scale

        for group in ACTIONABLE_GROUPS:
            old = group_value_from_row(original_row, group)
            new = group_value_from_row(row, group)
            if old != new:
                changed.append((group, old, new))
                distance += 1.0

        if not changed:
            continue

        desired_prob = float(probs[idx, desired_idx]) if desired_idx is not None else 0.0
        rows.append((len(changed), distance, -desired_prob, idx, changed, desired_prob))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    selected = []
    seen_signatures = set()
    for item in rows:
        _, _, _, idx, changed, desired_prob = item
        signature = tuple((x[0], str(x[2])) for x in changed)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append((idx, changed, desired_prob))
        if len(selected) >= n_results:
            break

    result_rows = []
    for idx, changed, desired_prob in selected:
        row = all_candidates.iloc[idx].copy()
        row["_desired_probability"] = desired_prob
        row["_changes"] = changed
        result_rows.append(row)

    return pd.DataFrame(result_rows), desired


# ============================================================
# App
# ============================================================
st.title("🩺 Diabetes Risk Predictor — Explainable AI")
st.caption("CatBoost prediction + native SHAP + actionable counterfactual explanations")

# Load only model/data. NO SHAP explainer and NO DiCE startup initialization.
try:
    with st.spinner("Loading model and data..."):
        model = load_model()
        full_df, X_train, FEATURE_ORDER = load_data()
except Exception as e:
    st.error("Could not load the model or training data.")
    st.exception(e)
    st.stop()

if not isinstance(model, CatBoostClassifier):
    st.warning(f"Loaded model type: {type(model).__name__}. CatBoost native SHAP may not be available.")

CLASS_LABELS = list(model.classes_)

# Sidebar information
with st.sidebar:
    st.header("TEAM: Kuchu Puchu")
    st.subheader("Members:")
    st.write("Riya Patil(2025CSB068)")
    st.write("Pravesh(2025CSB070)")
    st.write("Niraj Kumar Singh(2025CSB084)")
    st.write("Ayan Mansoori(2025CSB088)")
    st.write("Shivam Gupta(2025CSB100)")
    st.write("Aaman Manzar(2025ETB032)")

# Defaults
numeric_defaults = X_train[NUMERIC_FEATURES].median(numeric_only=True)

with st.form("patient_form"):
    st.subheader("Patient details")
    numeric_values = {}
    cols = st.columns(4)
    for i, feat in enumerate(NUMERIC_FEATURES):
        with cols[i % 4]:
            value = float(numeric_defaults.get(feat, 0.0))
            numeric_values[feat] = st.number_input(
                friendly_name(feat),
                value=value,
                format="%.2f",
            )

    st.subheader("Lifestyle and health profile")
    group_choices = {}
    gcols = st.columns(3)
    for i, (label, (dummy_cols, options, baseline)) in enumerate(GROUPS.items()):
        with gcols[i % 3]:
            group_choices[label] = st.selectbox(label, options + [baseline], index=len(options))

    country_options = [c.replace("Country_", "") for c in COUNTRY_COLS if c in FEATURE_ORDER]
    country_choice = st.selectbox("Country", country_options)

    st.subheader("Existing conditions")
    flag_values = {}
    fcols = st.columns(3)
    for i, (col, label) in enumerate(BINARY_FLAGS.items()):
        with fcols[i % 3]:
            flag_values[col] = st.checkbox(label)

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

if submitted:
    try:
        input_df = build_input_row(
            numeric_values,
            group_choices,
            country_choice,
            flag_values,
            FEATURE_ORDER,
            X_train,
        )
        st.session_state["input_df"] = input_df
    except Exception as e:
        st.error("Could not build the model input.")
        st.exception(e)
        st.stop()

if "input_df" not in st.session_state:
    st.info("Fill in the profile above and click **Predict Risk**.")
    st.stop()

input_df = st.session_state["input_df"]

# ============================================================
# Prediction
# ============================================================
proba = np.asarray(model.predict_proba(input_df))[0]
pred_idx = int(np.argmax(proba))
pred_class = str(CLASS_LABELS[pred_idx])

st.header("1️⃣ Prediction")
pcols = st.columns(len(CLASS_LABELS))
for c, label, p in zip(pcols, CLASS_LABELS, proba):
    c.metric(str(label), f"{float(p) * 100:.1f}%")

if pred_class == "High":
    st.error(f"Predicted risk category: **{pred_class}**")
elif pred_class == "Moderate":
    st.warning(f"Predicted risk category: **{pred_class}**")
else:
    st.success(f"Predicted risk category: **{pred_class}**")

# ============================================================
# SHAP — native CatBoost, only after prediction
# ============================================================
st.header("2️⃣ Why this prediction? — SHAP")

with st.spinner("Calculating SHAP explanation..."):
    try:
        shap_values, base_value = native_catboost_shap(
            model, input_df, FEATURE_ORDER, pred_idx
        )

        shap_series = pd.Series(shap_values, index=FEATURE_ORDER)
        top5 = shap_series.loc[
            shap_series.abs().sort_values(ascending=False).index[:5]
        ]

        fig = make_shap_chart(top5, pred_class)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.caption(
            "Positive SHAP values push the model toward the predicted class; "
            "negative values push it away. These are model contributions, not clinical effects."
        )

        st.subheader("Top 5 contributing features")
        explanation_rows = []
        for feat, val in top5.items():
            direction = "toward" if val > 0 else "away from"
            explanation_rows.append({
                "Feature": friendly_name(feat),
                "Patient value": input_df.iloc[0][feat],
                "SHAP": round(float(val), 4),
                "Effect": f"Pushes {direction} {pred_class}",
            })
        st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True, hide_index=True)

        st.subheader("Plain-English explanation")
        for feat, val in top5.items():
            direction = "pushes the model toward" if val > 0 else "pushes the model away from"
            value = input_df.iloc[0][feat]
            st.markdown(
                f"- **{friendly_name(feat)}** = `{value}` — this feature **{direction} "
                f"{pred_class}** (SHAP: `{val:+.3f}`)."
            )

    except Exception as e:
        st.error("SHAP calculation failed.")
        st.exception(e)

# ============================================================
# Counterfactuals — lazy, actionable model search
# ============================================================
st.header("3️⃣ What could change the prediction? — Counterfactual")
st.caption(
    "Searches for small changes to lifestyle variables that make the model predict a lower-risk class. "
    "Age, country, family history and existing conditions are kept fixed."
)

if st.button("🔄 Generate Counterfactuals", use_container_width=True):
    with st.spinner("Searching for actionable counterfactuals..."):
        try:
            cf_df, desired = counterfactual_search(
                model, input_df, X_train, pred_class, n_results=3
            )

            if cf_df.empty:
                st.warning(
                    f"No actionable counterfactual was found that changes the prediction toward **{desired}** "
                    "within the searched ranges. Try another profile."
                )
            else:
                st.success(f"Found counterfactuals that change the prediction toward **{desired}**.")

                original = input_df.iloc[0]
                for i, (_, row) in enumerate(cf_df.iterrows(), start=1):
                    st.subheader(f"Option {i}")
                    changed = row["_changes"]
                    desired_prob = row["_desired_probability"]

                    for item in changed:
                        name, old, new = item
                        if name in GROUPS:
                            st.markdown(
                                f"- **{name}:** `{old}` → `{new}`"
                            )
                        else:
                            st.markdown(
                                f"- **{friendly_name(name)}:** `{float(old):.2f}` → `{float(new):.2f}`"
                            )

                    st.caption(f"Model probability for {desired}: {desired_prob * 100:.1f}%")

                    # Show the resulting prediction explicitly.
                    cf_only = pd.DataFrame([row.drop(labels=["_desired_probability", "_changes"])])
                    cf_only = cf_only[FEATURE_ORDER]
                    cf_pred = str(model.predict(cf_only).reshape(-1)[0])
                    st.write(f"Resulting model prediction: **{cf_pred}**")

        except Exception as e:
            st.error("Counterfactual generation failed.")
            st.exception(e)

st.divider()
st.caption(
    "⚠️ Research/demo tool only. The model output and explanations should not be used as a medical diagnosis."
)
