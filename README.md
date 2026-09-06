# Customer Churn Prediction - End-to-End ML Project

An end-to-end Machine Learning project for predicting customer churn in a telecommunications company.

The project covers the complete ML lifecycle, from exploratory data analysis and model development to API serving, automated testing, and Docker containerization.

## Project Objectives

The main objectives of this project are to:

- Analyze customer behavior and identify factors associated with churn.
- Build and evaluate machine learning models for churn prediction.
- Prioritize the detection of customers at risk of churn.
- Create a reproducible training and inference pipeline.
- Expose the trained model through a REST API using FastAPI.
- Validate the application with automated tests.
- Containerize the prediction service using Docker.

## Dataset

The project uses the Telco Customer Churn dataset, containing information about 7,043 telecommunications customers.

The dataset includes customer demographics, subscribed services, account information, contract characteristics, payment methods, monthly charges, total charges, and the target variable `Churn`.

### Data Preparation

The main preprocessing steps include:

- Converting `TotalCharges` to a numeric variable.
- Handling missing `TotalCharges` values for customers with zero tenure.
- Removing `customerID` from the model features.
- Encoding the target variable `Churn` as 0 (No) and 1 (Yes).
- Standardizing numerical features.
- One-hot encoding categorical features.
- Using a stratified train/test split to preserve the churn distribution.

The dataset is moderately imbalanced, with approximately 26.5% churners and 73.5% non-churners.

## Exploratory Data Analysis

Exploratory analysis highlighted several customer characteristics associated with higher churn rates:

- **Tenure:** New customers showed substantially higher churn rates than long-term customers.
- **Contract:** Month-to-month customers had the highest churn rate, while customers with two-year contracts had the lowest.
- **Internet Service:** Fiber optic customers showed a higher churn rate than DSL customers.
- **Technical Support:** Customers without TechSupport were more likely to churn.
- **Online Security:** Customers without OnlineSecurity also showed higher churn.
- **Payment Method:** Electronic check customers had the highest observed churn rate among payment methods.
- **Paperless Billing:** Customers using paperless billing showed a higher observed churn rate.

These findings represent associations in the dataset and should not be interpreted as causal relationships.

## Model Development and Evaluation

Two baseline classification models were evaluated:

- Logistic Regression
- Random Forest

Model performance was assessed using stratified 5-fold cross-validation with several metrics, including accuracy, precision, recall, F1-score, and ROC-AUC.

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.802 | 0.652 | 0.544 | 0.592 | 0.846 |
| Random Forest | 0.786 | 0.628 | 0.482 | 0.545 | 0.820 |

Logistic Regression provided the strongest overall baseline performance, particularly in terms of ROC-AUC and F1-score.

### Handling Class Imbalance

Because identifying customers at risk of churn is particularly important for a retention use case, recall was considered a key metric.

A Logistic Regression model using `class_weight="balanced"` was therefore evaluated.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Standard Logistic Regression | 0.802 | 0.652 | 0.544 | 0.592 | 0.846 |
| Balanced Logistic Regression | 0.748 | 0.517 | 0.801 | 0.628 | 0.846 |

The balanced model significantly improved recall, allowing the model to identify a larger proportion of churners, at the cost of additional false positives and lower precision.

This trade-off was considered appropriate for a churn prevention scenario, where missing a customer who is likely to leave can be more costly than investigating some additional false positives.

### Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV` with stratified cross-validation.

The search included different values of the Logistic Regression regularization parameter `C` and different solvers.

The selected configuration was:

- `C = 0.1`
- `solver = "liblinear"`
- `class_weight = "balanced"`

Hyperparameter tuning produced only marginal improvements, indicating that the Logistic Regression model was relatively stable across the tested configurations.

### Final Model Performance

The selected model achieved the following performance on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.741 |
| Precision | 0.508 |
| Recall | 0.786 |
| F1-score | 0.617 |
| ROC-AUC | 0.841 |

The final confusion matrix was:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 750 | 285 |
| Actual Churn | 80 | 294 |

The model correctly identified 294 of the 374 churners in the test set, corresponding to a recall of approximately 78.6%.

## Project Architecture

The project is organized to separate experimentation, model training, inference, API serving, and testing.

```text
customer-churn-mlops/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_data_preprocessing.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── tests/
│   ├── test_api.py
│   └── test_data_preprocessing.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

The machine learning workflow follows the architecture:

```text
Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Serialized ML Pipeline
   ↓
FastAPI
   ↓
POST /predict
   ↓
Prediction + Churn Probability
```

## Model Pipeline

The final model is implemented as a scikit-learn `Pipeline` combining preprocessing and classification.

Numerical features are standardized using `StandardScaler`, while categorical features are transformed using `OneHotEncoder`.

The transformed features are then passed to the selected balanced Logistic Regression classifier.

Bundling preprocessing and prediction into the same pipeline ensures that the same transformations used during training are automatically applied during inference.

The trained pipeline is serialized using `joblib`.

## REST API with FastAPI

The trained model is exposed through a REST API built with FastAPI.

The main prediction endpoint is:

```text
POST /predict
```

The API:

1. Receives customer characteristics as JSON.
2. Validates the request using a Pydantic schema.
3. Converts the validated input into a pandas DataFrame.
4. Sends the data through the saved scikit-learn pipeline.
5. Returns the predicted class and churn probability as JSON.

Example response:

```json
{
  "prediction": "Churn",
  "churn_probability": 0.9046
}
```

FastAPI also provides interactive API documentation through Swagger UI at `/docs`.

## Automated Testing

Automated tests were implemented with `pytest`.

The test suite currently validates:

- Data cleaning and `TotalCharges` preprocessing.
- API health endpoint.
- Successful prediction requests.
- Validation and rejection of invalid input data.

Tests can be executed with:

```bash
python -m pytest -v
```

The current test suite contains four passing tests.

## Continuous Integration

A Continuous Integration workflow is implemented using GitHub Actions.

On every push or pull request to the `main` branch, the workflow automatically:

- Sets up the Python environment.
- Installs project dependencies.
- Trains the machine learning model.
- Runs the automated test suite.

This ensures that the project remains reproducible and that code changes do not break the existing pipeline.

## Docker

The prediction API is containerized using Docker to provide a reproducible runtime environment.

The Docker image contains:

- Python runtime
- Project dependencies
- FastAPI application
- ML preprocessing and inference code
- Serialized trained model

Build the image with:

```bash
docker build -t customer-churn-api .
```

Run the container with:

```bash
docker run --rm -p 8000:8000 customer-churn-api
```

The API can then be accessed locally on port `8000`, with Swagger documentation available at `/docs`.

A `.dockerignore` file is used to exclude development artifacts such as the virtual environment, Git metadata, notebooks, raw data, tests, and cache files from the Docker build context.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/FatimaZahraAABOUCH/customer-churn-mlops.git
cd customer-churn-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

This generates the trained model artifact:

```text
models/churn_model.joblib
```

### 4. Evaluate the model

```bash
python src/evaluate.py
```

### 5. Run the API locally

```bash
uvicorn api.main:app --reload
```

The interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 6. Run the automated tests

```bash
python -m pytest -v
```

### 7. Run with Docker

Build the Docker image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 customer-churn-api
```

## Technologies

- Python
- pandas
- scikit-learn
- Logistic Regression
- Random Forest
- FastAPI
- Pydantic
- pytest
- Docker
- Git

## Future Improvements

Potential improvements include:

- Cloud deployment of the prediction API
- Model and data monitoring
- Experiment tracking and model versioning