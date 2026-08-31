import streamlit as st
from utils import check_api_health, get_prediction

# Page Configuration
st.set_page_config(
    page_title="Employee Retention Predictor",
    layout="centered",
)

# Sidebar Health Check
st.sidebar.title("System Status")
api_online = check_api_health()

if api_online:
    st.sidebar.success("API Service: Connected")
else:
    st.sidebar.error("API Service: Offline")
    st.sidebar.warning(
        "Backend server is starting up or unreachable. Please allow up to 1 minute and refresh the page."
    )

# Header
st.title("Ergatility Employee Churn Prediction")
st.markdown("Adjust employee attributes below to predict the likelihood of churn.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Employee Metrics")

    col1, col2 = st.columns(2)

    with col1:
        satisfaction_level = st.slider(
            "Satisfaction Level", 0.0, 1.0, 0.75, 0.01
        )
        last_evaluation = st.slider(
            "Last Evaluation Score", 0.0, 1.0, 0.78, 0.01
        )
        number_project = st.number_input(
            "Number of Projects", min_value=2, max_value=7, value=4, step=1
        )
        average_montly_hours = st.number_input(
            "Average Monthly Hours",
            min_value=90,
            max_value=310,
            value=200,
            step=1,
        )

    with col2:
        time_spend_company = st.number_input(
            "Years at Company", min_value=2, max_value=10, value=3, step=1
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

# Cold Start Notice (Placed near submit button)
st.caption(
    "ℹ️ *Note: If the backend is idle, startup can take up to 30–60 seconds. Refreshing the page after a few seconds will re-check the status.*"
)

# Execution Logic
if submit_button:
    if not api_online:
        st.error(
            "Cannot submit prediction: Backend API server is unavailable or still waking up. Please refresh the page in a few moments."
        )
    else:
        # Build JSON payload for FastAPI endpoint
        payload = {
            "satisfaction_level": float(satisfaction_level),
            "last_evaluation": float(last_evaluation),
            "number_project": int(number_project),
            "average_montly_hours": int(average_montly_hours),
            "time_spend_company": int(time_spend_company),
            "work_accident": int(work_accident),
            "promotion_last_5years": int(promotion_last_5years),
            "salary": salary,
            "department": department,
        }

        with st.spinner("Calculating churn risk via API (this may take up to 60s if starting up)..."):
            response = get_prediction(payload)

        st.markdown("---")
        st.subheader("Prediction Output")

        if response and "error" not in response:
            prediction = response.get("prediction")
            probability = response.get("probability", 0.0)
            risk_level = response.get("risk_level", "Unknown")

            if prediction == 1:
                st.error(
                    f"{risk_level}: The employee is predicted to leave the company."
                )
            else:
                st.success(
                    f"{risk_level}: The employee is predicted to stay."
                )

            st.metric(
                label="Probability of Leaving",
                value=f"{probability * 100:.1f}%",
            )
            st.progress(float(probability))
        else:
            error_msg = (
                response.get("error") if response else "Unknown API Error"
            )
            st.error(f"Prediction Request Failed: {error_msg}")