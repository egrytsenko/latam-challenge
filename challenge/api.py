"""
Challenge: Flights Delays - ML Model to predict the probability of delay for a flight
           taking off or landing at SCL airport.

Description: For this challenge I would choose the model 6.b.i (XGBoost with Feature Importance
and Balance)! So, let's transcribe it to challenge/model.py as well and test it!

API Implementation

Author: "Eugenio Grytsenko" <yevgry@gmail.com>
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List

from .model import DelayModel

# Initialize the API
app = FastAPI()

# Initialize the model
model = DelayModel()


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictRequest(BaseModel):
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        df = pd.DataFrame([flight.dict() for flight in request.flights])

        # Check some values from API request
        # FIXME: add more validations (maybe Pydantic validators)
        if not all(df['MES'].between(1, 12)):
            raise ValueError("MES value out of range")

        processed_features = model.preprocess(df)
        prediction = model.predict(processed_features)
        return {"predict": prediction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))