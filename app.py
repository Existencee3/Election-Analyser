import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper function to serve HTML files from the template folder
def get_html_page(filename: str):
    path = os.path.join("template", filename)
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="<h1>404 Not Found</h1>", status_code=404)

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
    return FileResponse("template/index.html")

@app.get("/{filename}.html")
async def serve_html(filename: str):
    return get_html_page(f"{filename}.html")

@app.post("/bihar")
async def predict_bihar(voter_data: BiharVoter):
   
    try:
        
        voter_info = voter_data.model_dump()
        
        normalized_info = {
            "Age_Group": voter_info["Age_Group"],
            "Gender": voter_info["Gender"].title(),
            "Geography": voter_info["Geography"].title(),
            "Education": voter_info["Education"].title(),
            "Occupation": voter_info["Occupation"].title(),
            "Caste": voter_info["Caste"].title()
        }

        
        input_df = pd.DataFrame([normalized_info])
        input_df = input_df[['Age_Group', 'Gender', 'Geography', 'Education', 'Occupation', 'Caste']]

        prediction = bihar_model.predict(input_df)
        predicted_party = prediction[0]

        return {
            "status": "success",
            "prediction": {
                "voted_party": predicted_party
            },
            "parameters_used": normalized_info
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