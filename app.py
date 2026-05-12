import streamlit as st
import pandas as pd
import joblib

def add_features(X):
    X = X.copy()

    X["screen_sleep_ratio"] = X["screen_time_before_sleep"] / X["sleep_hours"]
    X["social_physical_ratio"] = X["daily_social_media_hours"] / (X["physical_activity"] + 0.1)

    X["high_social_usage"] = (X["daily_social_media_hours"] > 6).astype(int)
    X["low_sleep"] = (X["sleep_hours"] < 6).astype(int)

    X["mental_health_score"] = (
        X["stress_level"] +
        X["anxiety_level"] +
        X["addiction_level"]
    ) / 3

    return X

# Load Pipeline
pipeline = joblib.load("models/pipeline.pkl")

# App Title
st.title("Teen Mental Health Prediction")
st.write("Predict likelihood of depression based on social media and lifestyle factors")


# User Inputs
age = st.slider("Age", 13, 19, 16)

gender = st.selectbox("Gender", ["male", "female"])

daily_social_media_hours = st.slider("Daily Social Media Usage (hours)", 0.0, 10.0, 4.0)

platform_usage = st.selectbox("Platform Usage", ["Instagram", "TikTok", "Both"])

sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0)

screen_time_before_sleep = st.slider("Screen Time Before Sleep (hours)", 0.0, 5.0, 1.5)

academic_performance = st.slider("Academic Performance (GPA approx)", 0.0, 4.0, 3.0)

physical_activity = st.slider("Physical Activity (hours/day)", 0.0, 3.0, 1.0)

social_interaction_level = st.selectbox("Social Interaction Level", ["low", "medium", "high"])

stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

anxiety_level = st.slider("Anxiety Level (1-10)", 1, 10, 5)

addiction_level = st.slider("Addiction Level (1-10)", 1, 10, 5)


# Predict Button
if st.button("Predict"):

    # Create DataFrame (RAW input)
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "daily_social_media_hours": daily_social_media_hours,
        "platform_usage": platform_usage,
        "sleep_hours": sleep_hours,
        "screen_time_before_sleep": screen_time_before_sleep,
        "academic_performance": academic_performance,
        "physical_activity": physical_activity,
        "social_interaction_level": social_interaction_level,
        "stress_level": stress_level,
        "anxiety_level": anxiety_level,
        "addiction_level": addiction_level
    }])

    # Prediction
    prob = pipeline.predict_proba(input_data)[0][1]
    prediction = 1 if prob > 0.4 else 0

    # Output
    st.subheader("Result")

    if prediction == 1:
        st.error(f"High Risk of Depression (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk of Depression (Probability: {prob:.2f})")