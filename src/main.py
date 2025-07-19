from fastapi import FastAPI
import pandas as pd
from joblib import load
from src.train_model import train
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of prediction")

@app.post("/predict")
@PREDICTION_LATENCY.time()
def predict(data: dict):
    model = load('model.joblib')
    
    REQUEST_COUNT.inc()
    
    df = pd.DataFrame([data])

    # Get prediction (0 = No churn, 1 = Churn)
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]

    print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    print(f"Churn Probability: {probability:.2f}")

    return {"churn": int(prediction)}
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
@app.get("/train")
def train_main():
    train()
