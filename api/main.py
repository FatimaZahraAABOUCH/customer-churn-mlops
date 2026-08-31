import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

model = joblib.load("models/churn_model.joblib") 

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running"
    }

@app.post("/predict")
def predict(customer: CustomerData):

    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0, 1]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }