import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Employee Retention Predictor", page_icon="👔", layout="centered"
)


# Load Model Artifact
@st.cache_resource
def load_model():
    model_path = os.path.join(".", "artifacts", "rf_cv_model.pickle")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model from `../artifacts/rf_cv_model.pickle`: {e}")
    st.stop()

# Header
st.title("👔 Employee Churn Prediction")
st.markdown("Adjust employee attributes below to predict the likelihood of churn.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Employee Metrics")

    col1, col2 = st.columns(2)

    with col1:
        satisfaction_level = st.slider(
            "Satisfaction Level", 0.0, 1.0, 0.75, 0.01
        )
        last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.78, 0.01)
        number_project = st.number_input(
            "Number of Projects", min_value=1, max_value=20, value=4, step=1
        )
        average_montly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=50,
            max_value=400,
            value=200,
            step=1,
        )

    with col2:
        time_spend_company = st.number_input(
            "Years at Company", min_value=1, max_value=20, value=3, step=1
        )
        work_accident = st.selectbox(
            "Had Work Accident?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        promotion_last_5years = st.selectbox(
            "Promoted in Last 5 Years?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        salary = st.selectbox(
            "Salary Level", options=["low", "medium", "high"], index=1
        )
        department = st.selectbox(
            "Department",
            options=[
                "sales",
                "accounting",
                "hr",
                "technical",
                "support",
                "management",
                "IT",
                "product_mng",
                "marketing",
                "RandD",
            ],
            index=0,
        )

    submit_button = st.form_submit_button(
        "Predict Churn Risk", type="primary", use_container_width=True
    )

# Execution Logic
if submit_button:
    # 1. Map Form Values
    salary_map = {"low": 0, "medium": 1, "high": 2}

    raw_input = {
        "satisfaction_level": float(satisfaction_level),
        "last_evaluation": float(last_evaluation),
        "number_project": int(number_project),
        "average_montly_hours": int(average_montly_hours),
        "time_spend_company": int(time_spend_company),
        "work_accident": int(work_accident),
        "promotion_last_5years": int(promotion_last_5years),
        "salary": salary_map[salary],
    }

    departments = [
        "RandD",
        "accounting",
        "hr",
        "management",
        "marketing",
        "product_mng",
        "sales",
        "support",
        "technical",
    ]
    for d in departments:
        raw_input[f"department_{d}"] = 1 if department == d else 0

    final_input = pd.DataFrame([raw_input])

    # 2. Extract Model's Expected Feature Order
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = list(final_input.columns)

    # Re-align final input DataFrame
    final_input_aligned = final_input.reindex(columns=expected_cols, fill_value=0)

    # --- DEBUG COMPARISON TABLE BLOCK ---
    st.markdown("---")
    st.subheader("🔍 Feature Alignment Debugger")

    comparison_data = []
    for idx, col in enumerate(expected_cols):
        val = (
            final_input_aligned[col].values[0]
            if col in final_input_aligned.columns
            else "MISSING"
        )
        comparison_data.append(
            {
                "Index": idx,
                "Expected Feature Name": col,
                "Passed Value": val,
                "Data Type": type(val).__name__,
            }
        )

    debug_df = pd.DataFrame(comparison_data)
    st.dataframe(debug_df, use_container_width=True)

    # Highlight any mismatched or missing columns
    missing_in_input = set(expected_cols) - set(final_input.columns)
    extra_in_input = set(final_input.columns) - set(expected_cols)

    if missing_in_input:
        st.warning(f"⚠️ Missing columns filled with 0: {list(missing_in_input)}")
    if extra_in_input:
        st.warning(f"⚠️ Extra columns dropped: {list(extra_in_input)}")
    # -------------------------------------

    # 3. Model Prediction
    prediction = model.predict(final_input_aligned)[0]
    probability = model.predict_proba(final_input_aligned)[0][1]

    st.markdown("---")
    st.subheader("Prediction Output")
    st.metric(
        label="Probability of Leaving", value=f"{probability * 100:.1f}%"
    )
    # 1. Create raw DataFrame matching original notebook structure
    raw_df = pd.DataFrame(
        [
            {
                "satisfaction_level": satisfaction_level,
                "last_evaluation": last_evaluation,
                "number_project": number_project,
                "average_montly_hours": average_montly_hours,
                "time_spend_company": time_spend_company,
                "work_accident": work_accident,
                "promotion_last_5years": promotion_last_5years,
                "salary": salary,
                "department": department,
            }
        ]
    )

    # 2. Map ordinal salary
    salary_map = {"high": 2, "medium": 1, "low": 0}
    raw_df["salary"] = raw_df["salary"].map(salary_map)

    # 3. Create full dummy representation matching pd.get_dummies output
    # Define every department option that was present in training data
    all_departments = [
        "RandD",
        "accounting",
        "hr",
        "management",
        "marketing",
        "product_mng",
        "sales",
        "support",
        "technical",
    ]  # 'IT' is reference category (dropped)

    for d in all_departments:
        raw_df[f"department_{d}"] = 1 if department == d else 0

    raw_df = raw_df.drop(columns=["department"])

    # 4. Strictly reorder columns to match model's trained signature
    if hasattr(model, "feature_names_in_"):
        final_input = raw_df[model.feature_names_in_]
    else:
        final_input = raw_df

    # 5. Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    # Output Render
    st.markdown("---")
    st.subheader("Prediction Output")

    if prediction == 1:
        st.error(
            "🚨 **High Risk:** The employee is predicted to leave the company."
        )
    else:
        st.success("✅ **Low Risk:** The employee is predicted to stay.")

    if probability is not None:
        st.metric(
            label="Probability of Leaving", value=f"{probability * 100:.1f}%"
        )
        st.progress(float(probability))