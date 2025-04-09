from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from RandomForest3400 import SimpleRandomForest

housing = fetch_california_housing()
feature_names = housing.feature_names

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("california_housing_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

class HouseFeatures(BaseModel):
    features: list[float]

@app.post("/predict/")
async def predict_price(data: HouseFeatures):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        return {"error": str(e)}         

