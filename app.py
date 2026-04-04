import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class BiharVoter(BaseModel):
    Age_Group: str
    Gender: str
    Geography: str
    Education: str
    Occupation: str
    Caste: str

class MaharashtraVoter(BaseModel):
    region: str
    district: str
    geography: str
    gender: str
    age_band: str
    age: float
    caste: str
    occupation: str

bihar_model = joblib.load("models/Bihar_voter_prediction.pkl")
maharastra_model = joblib.load("models/Maharastra.pkl")

@app.get("/")
async def root():
    status = "active"
    available_models = ["bihar", "maharashtra"]
    response = {
        "status": status,
        "models": available_models
    }
    return response

@app.post("/bihar")
async def predict_bihar(data: BiharVoter):
    voter_dict = data.model_dump()
    voter_list = [voter_dict]
    voter_df = pd.DataFrame(voter_list)
    
    predictions = bihar_model.predict(voter_df)
    voted_party = predictions[0]
    
    result = {"voted_party": voted_party}
    return result

@app.post("/maharashtra")
async def predict_maharashtra(data: MaharashtraVoter):
    voter_dict = data.model_dump()
    voter_list = [voter_dict]
    voter_df = pd.DataFrame(voter_list)
    
    predictions = maharastra_model.predict(voter_df)
    voted_party = predictions[0]
    
    if hasattr(voted_party, "item"):
        voted_party = int(voted_party)
        
    result = {"voted_party": voted_party}
    return result