from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle

app = FastAPI()

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def root():
    return {"message": " Predictive Model API is live!"}
class UserInput(BaseModel):
    site_area: float=Field(...,alias="site area",gt=0)
    structure_type: str = Field(..., alias="structure type")
    water_consumption:float=Field(...,alias="water consumption")
    recycling_rate: float = Field(..., alias="recycling rate")
    utilisation_rate: float = Field(..., alias="utilisation rate")
    air_quality_index: float = Field(..., alias="air quality index")
    issue_resolution: float = Field(..., alias="issue reolution time")
    resident_count: int = Field(..., alias="resident count",ge=0)

    class Config:
        populate_by_name = True  # Allow alias-based input


@app.post('/predict')
def pred_failure_type(input_data: UserInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.dict(by_alias=True)])

    
    prediction = model.predict(input_df)[0]

    return {"predicted_value": float(prediction)}
