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
#maharastra_model = joblib.load("models/Maharastra.pkl")

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
async def predict_bihar(voter_data: BiharVoter):
   
    try:
        
        voter_info = voter_data.model_dump()
        
        info = {
            "Age_Group": voter_info["Age_Group"],
            "Gender": voter_info["Gender"].title(),
            "Geography": voter_info["Geography"].title(),
            "Education": voter_info["Education"].title(),
            "Occupation": voter_info["Occupation"].title(),
            "Caste": voter_info["Caste"].title()
        }

        
        input_df = pd.DataFrame([info])
        input_df = input_df[['Age_Group', 'Gender', 'Geography', 'Education', 'Occupation', 'Caste']]

        prediction = bihar_model.predict(input_df)
        predicted_party = prediction[0]

        return {
            "status": "success",
            "prediction": {
                "voted_party": predicted_party
            },
            "parameters_used": info
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": f"Mapping error: {str(e)}.",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"error occurred: {str(e)}"
        }

# @app.post("/maharashtra")
# async def predict_maharashtra(data: MaharashtraVoter):
#     voter_dict = data.model_dump()
#     voter_list = [voter_dict]
#     voter_df = pd.DataFrame(voter_list)
    
#     predictions = maharastra_model.predict(voter_df)
#     voted_party = predictions[0]
    
#     if hasattr(voted_party, "item"):
#         voted_party = int(voted_party)
        
#     result = {"voted_party": voted_party}
#     return result