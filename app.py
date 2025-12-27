import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

with open('exam_score_model.pkl', 'rb') as f:
    linear = pickle.load(f)  
    
with open('exam_score_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model_loaded = True

st.set_page_config(
    page_title="Exam Score Predictor",
    layout="centered"
)

st.title("Exam Score Predictor")
st.markdown("Enter your details to predict your exam score")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5
    )
    
    class_attendance = st.number_input(
        "Class Attendance (%)",
        min_value=40.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

with col2:
    sleep_hours = st.number_input(
        "Sleep Hours per Night",
        min_value=4.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )
    
    facility_rating = st.selectbox(
        "Facility Rating",
        options=[1, 2, 3],
        format_func=lambda x: ["Low", "Medium", "High"][x-1],
        index=1
    )

# Predict button
if st.button("PREDICT MY SCORE", type="primary"):
    
    if model_loaded:
        try:
            # Prepare input
            user_input = np.array([[study_hours, class_attendance, sleep_hours, facility_rating]])
            
            # Scale the input
            scaled_input = scaler.transform(user_input)
            
            # Make prediction
            predicted_score = linear.predict(scaled_input)[0]
            
            # Display result
            st.markdown("---")
            st.markdown(f"Predicted Score: **{predicted_score:.1f}/100**")
            
            # Interpretation
            if predicted_score >= 85:
                st.success("Excellent! You're on track for top grades!")
                st.balloons()
            elif predicted_score >= 70:
                st.info("Good job! Keep up the consistent effort!")
            elif predicted_score >= 50:
                st.warning("Average. Focus on study habits.")
            else:
                st.error("Needs improvement. Focus on study habits.")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check that input order matches training data")
    else:
        st.error("Model not loaded! Please check .pkl files.")


