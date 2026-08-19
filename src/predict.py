import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict_churn(model, customer_data):
    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0, 1]

    return prediction, probability

if __name__ == "__main__":

    model = load_model("models/churn_model.joblib")


    customer_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.50,
        "TotalCharges": 271.50
    }

    prediction, probability = predict_churn(
        model,
        customer_data
    )

    print("Prediction:", prediction)
    print("Churn probability:", probability)