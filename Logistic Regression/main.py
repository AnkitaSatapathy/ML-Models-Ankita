from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from LogisticRegression3400 import LogisticRegressionCustom

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

class CancerInput(BaseModel):
    mean_radius: float

@app.post("/predict")
def predict(data: CancerInput):
    input_value = np.array([[data.mean_radius]])
    min_value, max_value = 6.981, 28.11  
    input_value = (input_value - min_value) / (max_value - min_value)
    probability = float(model.predict_proba(input_value)[0])  
    prediction = int(probability < 0.6)                                                                  
    return {"probability": round(probability, 2), "prediction": prediction}

