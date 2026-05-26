import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚖")
st.title("🚖 Real-Time Taxi Fare Predictor")
st.write("Adjust any variable below. Your model will calculate a brand new price instantly.")

st.divider()

# 2. Dynamic Input Fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trip Metrics")
    distance = st.slider("Trip Distance (km)", min_value=0.5, max_value=100.0, value=12.5, step=0.5)
    duration = st.slider("Trip Duration (minutes)", min_value=1.0, max_value=120.0, value=25.0, step=1.0)
    passengers = st.slider("Passenger Count", min_value=1.0, max_value=6.0, value=1.0, step=1.0)
    
    st.subheader("Environment")
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    day_of_week = st.selectbox("Day of the Week", ["Weekday", "Weekend"])
    traffic = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather Conditions", ["Clear", "Rain", "Snow"])

with col2:
    st.subheader("Fare Rates & Multipliers")
    st.info("These values interact mathematically inside your model's pipeline.")
    base_fare = st.number_input("Base Fare ($)", min_value=0.0, max_value=50.0, value=3.50, step=0.50)
    per_km = st.number_input("Per Km Rate ($)", min_value=0.0, max_value=10.0, value=1.25, step=0.05)
    per_minute = st.number_input("Per Minute Rate ($)", min_value=0.0, max_value=5.0, value=0.30, step=0.05)

st.divider()

# 3. Calculation and Payload Request
if st.button("Calculate Dynamic Fare", type="primary"):
    payload = {
        "Trip_Distance_km": float(distance),
        "Time_of_Day": str(time_of_day),
        "Day_of_Week": str(day_of_week),
        "Passenger_Count": float(passengers),
        "Traffic_Conditions": str(traffic),
        "Weather": str(weather),
        "Base_Fare": float(base_fare),
        "Per_Km_Rate": float(per_km),
        "Per_Minute_Rate": float(per_minute),
        "Trip_Duration_Minutes": float(duration)
    }
    
    try:
        api_url = "http://127.0.0.1:8000/predict" 
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            fare = result["estimated_fare"]
            
            # Show response cleanly
            st.success(f"### 💰 Model Predicted Fare: ${fare:,.2f}")
            
            # Mini debug expander so you can see exactly what went out
            with st.expander("🔍 View Raw JSON Payload Sent to FastAPI"):
                st.json(payload)
        else:
            st.error(f"Backend Error: Status code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Connection Refused! Ensure your FastAPI server terminal (`app.py`) is running on port 8000.")