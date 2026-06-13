from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.lightgbm
import pandas as pd
import numpy as np
import sys
sys.path.append("../src")
from preprocess import engineer_features, get_feature_columns

app = FastAPI(title="House Price Predictor", version="1.0")

model = mlflow.lightgbm.load_model("mlflow_runs/model")


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def root():
    return {"status": "ok", "model": "california-house-price-lgbm"}


@app.post("/predict")
def predict(data: HouseFeatures):
    df = pd.DataFrame([data.dict()])
    df = engineer_features(df)
    X = df[get_feature_columns()]
    prediction = model.predict(X)[0]
    return {
        "predicted_median_house_value": round(float(prediction) * 100_000, 2)
    }
