from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the app
app1 = FastAPI()  # Changed the app name to app1

scaler = joblib.load('scaler1.joblib')    # Replace with the path to your saved scaler if scaling was used
# Load the trained model and scaler
model = joblib.load('svm_model1.joblib')  # Replace with the path to your saved model

# Define the input data schema
class ModelInput(BaseModel):
    N: float  # Nitrogen
    P: float  # Phosphorus
    K: float  # Potassium
    temperature: float
    humidity: float
    pH: float
    rainfall: float

# Define the prediction endpoint
@app1.post("/predict")
def predict(input_data: ModelInput):
    # Convert input data to a NumPy array
    data = np.array([[input_data.N, input_data.P, input_data.K, 
                      input_data.temperature, input_data.humidity, 
                      input_data.pH, input_data.rainfall]])
    
    # Scale the input data (if scaling was used during training)
    scaled_data = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(scaled_data)

    # Return the prediction result
    return {"prediction": int(prediction[0])}
