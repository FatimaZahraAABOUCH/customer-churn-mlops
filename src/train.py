import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from data_preprocessing import clean_data, build_preprocessor

def train_model(x_train, y_train):

    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]
    )

    model.fit(x_train, y_train)
    
    return model

def load_and_prepare_data(data_path):

    df = pd.read_csv(data_path)

    df = clean_data(df)

    x = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return x, y

if __name__ == "__main__":

    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    x, y = load_and_prepare_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = train_model(x_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        model,
        "models/churn_model.joblib"
    )

    print("Model trained and saved successfully!")