from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from SVM3400 import CustomSVM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("svm_titanic_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

class InputData(BaseModel):
    features: list

@app.post("/predict/")
async def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    features_scaled = scaler.transform(features)  
    prediction = model.predict(features_scaled)[0]                                  
    class_name = "Survived" if prediction == 1 else "Not Survived"
    return {"prediction": class_name}
