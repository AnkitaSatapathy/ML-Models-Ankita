from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from LinearRegression3400 import SimpleLinearRegressionCustom

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("model.pkl", "rb") as file:
    model = pickle.load(file)

class DiabetesInput(BaseModel):
    bmi: float

@app.post("/predict")
def predict(data: DiabetesInput):
    bmi_value = np.array(data.bmi).reshape(-1, 1)
    prediction = model.predict(bmi_value)
    return {"diabetes_progression": round(float(prediction[0]), 2)}

