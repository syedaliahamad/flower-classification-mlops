
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["feature_names"]
class_labels = saved["class_labels"]

app = FastAPI(title="Flower Classification API")


class Flower(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Flower Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "features": features,
        "class_labels": class_labels,
        "hyperparameters": saved["hyperparameters"]
    }


@app.post("/predict")
def predict(flower: Flower):
    values = [
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]

    input_data = pd.DataFrame([values], columns=features)
    prediction = model.predict(input_data)[0]

    return {"predicted_species": str(prediction)}
