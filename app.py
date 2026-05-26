import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Taxi Price Prediction API")

# self-contained predictor
class LocalTaxiPredictor:
    def __init__(self, model_path='final_model.pkl'):
        self.model = joblib.load(model_path)
        # The exact 15 features your model expects in order
        self.feature_names = [
            'Trip_Distance_km', 'Passenger_Count', 'Traffic_Conditions', 'Base_Fare', 
            'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Time_of_Day_Morning', 
            'Day_of_Week_Weekend', 'Weather_Rain', 'Weather_Snow', 'Distance_Per_Minute', 
            'Is_Long_Trip', 'Cost_Interaction', 'Passenger_Density'
        ]

    def predict(self, input_data: dict) -> float:
        # Create a single row dictionary with baseline keys to prevent missing keys
        processed_data = {}
        
        # Numeric extraction
        processed_data['Trip_Distance_km'] = float(input_data.get('Trip_Distance_km', 0))
        processed_data['Passenger_Count'] = float(input_data.get('Passenger_Count', 1))
        processed_data['Base_Fare'] = float(input_data.get('Base_Fare', 0))
        processed_data['Per_Km_Rate'] = float(input_data.get('Per_Km_Rate', 0))
        processed_data['Per_Minute_Rate'] = float(input_data.get('Per_Minute_Rate', 0))
        
        duration = float(input_data.get('Trip_Duration_Minutes', 1))
        duration = duration if duration > 0 else 1
        processed_data['Trip_Duration_Minutes'] = duration

        # Ordinal mapping for Traffic Conditions
        traffic_map = {'Low': 0.0, 'Medium': 1.0, 'High': 2.0}
        traffic_str = input_data.get('Traffic_Conditions', 'Medium')
        processed_data['Traffic_Conditions'] = traffic_map.get(traffic_str, 1.0)

        # One-Hot Encoding calculations
        processed_data['Time_of_Day_Morning'] = 1.0 if input_data.get('Time_of_Day') == 'Morning' else 0.0
        processed_data['Day_of_Week_Weekend'] = 1.0 if input_data.get('Day_of_Week') == 'Weekend' else 0.0
        processed_data['Weather_Rain'] = 1.0 if input_data.get('Weather') == 'Rain' else 0.0
        processed_data['Weather_Snow'] = 1.0 if input_data.get('Weather') == 'Snow' else 0.0

        # Feature Engineering calculations
        processed_data['Distance_Per_Minute'] = processed_data['Trip_Distance_km'] / duration
        processed_data['Is_Long_Trip'] = 1.0 if processed_data['Trip_Distance_km'] > 50.0 else 0.0
        processed_data['Cost_Interaction'] = processed_data['Per_Km_Rate'] * processed_data['Per_Minute_Rate']
        processed_data['Passenger_Density'] = processed_data['Passenger_Count'] / duration

        # Turn it into a DataFrame ordered exactly by what scikit-learn expects
        df = pd.DataFrame([processed_data])
        final_features = df[self.feature_names].astype(float)
        
        return float(self.model.predict(final_features)[0])

# Initialize the predictor
predictor = LocalTaxiPredictor()

# Define API Input Schema
class TripInput(BaseModel):
    Trip_Distance_km: float
    Time_of_Day: str
    Day_of_Week: str
    Passenger_Count: float
    Traffic_Conditions: str
    Weather: str
    Base_Fare: float
    Per_Km_Rate: float
    Per_Minute_Rate: float
    Trip_Duration_Minutes: float

# API Endpoint
@app.post("/predict")
def predict_fare(trip: TripInput):
    data = trip.model_dump()
    prediction = predictor.predict(data)
    return {"estimated_fare": round(prediction, 4)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)