from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
import os

app = FastAPI()

# Safe loading using relative paths
BASE_DIR = os.path.dirname(__file__)
model = tf.keras.models.load_model("churn_model.h5", compile=False)
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
input_cols = joblib.load(os.path.join(BASE_DIR, "input_cols.pkl"))

class CustomerData(BaseModel):
    customerID: str
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

def preprocess(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    df.drop("customerID", axis=1, inplace=True)
    df.replace('No phone service', 'No', inplace=True)
    df.replace('No internet service', 'No', inplace=True)

    yes_no_cols = [
        "partner", "dependents", "phoneservice", "multiplelines",
        "onlinesecurity", "onlinebackup", "deviceprotection", "techsupport",
        "streamingtv", "streamingmovies", "paperlessbilling"
    ]
    df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].replace({"Male": 1, "Female": 0})

    num_cols = ["tenure", "monthlycharges", "totalcharges"]
    df[num_cols] = scaler.transform(df[num_cols])

    cat_cols = ["internetservice", "contract", "paymentmethod"]
    df = pd.get_dummies(df, columns=cat_cols)
    df = df.reindex(columns=input_cols, fill_value=0)
    return df

@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is live!"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        X = preprocess(data)
        prediction = model.predict(X)[0][0]
        result = "Churn" if prediction > 0.5 else "No Churn"
        return {
            "customerID": data.customerID,
            "prediction": result,
            "probability": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
