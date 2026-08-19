import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from train import load_and_prepare_data

def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":

    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_path = "models/churn_model.joblib"

    x, y = load_and_prepare_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = joblib.load(model_path)

    evaluate_model(
        model,
        x_test,
        y_test
    )