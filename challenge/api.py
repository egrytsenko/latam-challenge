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
    """
    A model representing flight data for prediction.
    Fields:
    - OPERA: Name of the airline that operates.
    - TIPOVUELO: Type of flight, I=International, N=National.
    - MES: Number of the month of operation of the flight.
    """
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictRequest(BaseModel):
    """
    A model representing a request for flight delay prediction.
    Contains a list of Flight objects.
    """
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint.
    Returns a JSON object with the status 'OK' indicating that the API is operational.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    """
    Endpoint for predicting flight delays.
    Takes a PredictRequest object containing flight data and returns predictions.

    Args:
    - request: A PredictRequest object containing flight data.

    Returns:
    - A dictionary with the prediction results.
    """
    try:
        df = pd.DataFrame([flight.dict() for flight in request.flights])

        # Check some values from API request
        # FIXME: add more validations (maybe Pydantic validators)
        if not all(df['MES'].between(1, 12)):
            raise ValueError("MES value out of range")

        # Preprocess data & make predictions
        processed_features = model.preprocess(df)
        prediction = model.predict(processed_features)

        # Successful response from model
        return {"predict": prediction}
    except ValueError as e:
        # Catch errors from input validation
        raise HTTPException(status_code=400, detail=str(e))