import fastapi
from typing import List
from challenge.model import DelayModel
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum

app = fastapi.FastAPI()

# Define the path to the model file
model_file = 'model.joblib'

# Create model instance
model = DelayModel()
# Load the model if it exists. Else it trains it
model.load_or_fit(model_file)


VALID_OPERA_VALUES = [
    'Grupo LATAM', 
    'Sky Airline', 
    'Aerolineas Argentinas', 
    'Copa Air',
    'Latin American Wings', 
    'Avianca', 
    'JetSmart SPA', 
    'Gol Trans',
    'American Airlines', 
    'Air Canada', 
    'Iberia', 
    'Delta Air', 
    'Air France',
    'Aeromexico', 
    'United Airlines', 
    'Oceanair Linhas Aereas', 
    'Alitalia',
    'K.L.M.', 
    'British Airways', 
    'Qantas Airways', 
    'Lacsa', 
    'Austral',
    'Plus Ultra Lineas Aereas'
]

# Creating OperaEnum dynamically
opera_values_dict = {value.replace(' ', '_').replace('.', '').replace('-', '_'): value for value in VALID_OPERA_VALUES}
OperaEnum = Enum("OperaEnum", opera_values_dict)

class TipoVueloEnum(str, Enum):
    """Enum class to contain all valid TIPOVUELO values"""
    N = 'N'
    I = 'I'

class FlightItem(BaseModel):
    OPERA: OperaEnum
    TIPOVUELO: TipoVueloEnum
    MES: int = Field(..., ge=1, le=12)  # ge = greater than or equal to, le = less than or equal to


class Flights(BaseModel):
    flights: List[FlightItem]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(flight_data: Flights) -> dict:
    # Parse the input JSON into a DataFrame
    data = pd.DataFrame([flight.dict() for flight in flight_data.flights])
    
    # Preprocess the input data and predict the target
    features = model.preprocess(data)
    predictions = model.predict(features)
    
    return {"predict": predictions}
