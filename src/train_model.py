import mlflow
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.data_preprocessing import load_and_preprocess
import pandas as pd
import joblib


def train():
    X_train, X_test, y_train, y_test = load_and_preprocess('data/telco.csv')
    X_train_original = X_train.copy()
   # Assume these are the raw categorical and numerical columns
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod']

    numerical_cols = ['SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    model.fit(X_train, y_train)


    # Save full pipeline
    joblib.dump(model, 'model.joblib')

    mlflow.log_param("model", "xgboost")
    signature = infer_signature(X_test, model.predict(X_test))
    #mlflow.sklearn.log_model(model, artifact_path="xgb_model")
    mlflow.sklearn.log_model(model, name="model", signature=signature)



if __name__ == "__main__":
    train()
