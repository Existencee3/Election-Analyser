"""
Run with:
    uvicorn api:app --reload --port 5000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pickle, os, traceback
import pandas as pd

app = FastAPI(title="Election Analyzer API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_model_cache: dict = {}

def load_model(filename: str):
    if filename in _model_cache:
        return _model_cache[filename]
    path = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(path):
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.lower() == filename.lower():
                    path = os.path.join(MODEL_DIR, f)
                    break
    
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    _model_cache[filename] = model
    return model


class VoterPredictionInput(BaseModel):
    state: str
    Age_Group: str
    Gender: str
    Geography: str
    Caste: str
    Education: Optional[str] = None
    Occupation: str

class MahaVoterPredictionInput(BaseModel):
    state: str
    Age: int
    District: str
    Gender: str
    Geography: str
    Caste: str
    Occupation: str

@app.get("/")
def root():
    index_path = os.path.join(TEMPLATE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "ok", "message": "index.html not found in template/"}

@app.get("/bihar")
def bihar_page():
    return FileResponse(os.path.join(TEMPLATE_DIR, "bihar.html"))

@app.get("/maharashtra")
def maharashtra_page():
    return FileResponse(os.path.join(TEMPLATE_DIR, "maharashtra.html"))


@app.get("/health")
def health():
    return {"status": "ok", "message": "Election Analyzer API is running"}

@app.get("/models/status")
def model_status():
    files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
    pkl_files = [f for f in files if f.endswith(".pkl")]
    return {
        "available_models": pkl_files,
        "model_dir": MODEL_DIR
    }

@app.post("/bihar/predict_voter")
def bihar_voter_predict(data: VoterPredictionInput):
    model = load_model("bihar_voter_prediction.pkl")
    if model is None:
        raise HTTPException(status_code=503, detail="Bihar model not found")
    try:
        input_dict = {
            "Age_Group": data.Age_Group.strip(),
            "Gender": data.Gender.strip(),
            "Geography": data.Geography.strip(),
            "Education": (data.Education.strip() if data.Education else ""),
            "Occupation": data.Occupation.strip(),
            "Caste": data.Caste.strip()
        }
        columns = ["Age_Group", "Gender", "Geography", "Education", "Occupation", "Caste"]
        features = pd.DataFrame([input_dict])[columns]
        prediction = model.predict(features)[0]
        proba = None
        estimator = model
        if hasattr(model, "steps"): 
            estimator = model.steps[-1][1]
        if hasattr(estimator, "predict_proba"):
            proba_vals = model.predict_proba(features)[0]
            classes = estimator.classes_
            proba = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, proba_vals)}
        return {
            "status": "success",
            "predicted_party": str(prediction),
            "probabilities": proba
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/maharashtra/predict_voter")
def maha_voter_predict(data: MahaVoterPredictionInput):
    model = load_model("maharashtra_voter_prediction.pkl")
    if model is None:
        raise HTTPException(status_code=503, detail="Maharashtra model not found")
    try:
        input_dict = {
            "age": data.Age,
            "gender": data.Gender.strip(),
            "district": data.District.strip(),
            "geography": data.Geography.strip(),
            "caste": data.Caste.strip(),
            "occupation": data.Occupation.strip()
        }
        if input_dict["caste"] == "OBC":
            input_dict["caste"] = "Other OBC"
        elif input_dict["caste"] == "General":
            input_dict["caste"] = "Other General"
        
        columns = ["age", "gender", "district", "geography", "caste", "occupation"]
        features = pd.DataFrame([input_dict])[columns]
        prediction = model.predict(features)[0]
        proba = None
        estimator = model
        if hasattr(model, "steps"): 
            estimator = model.steps[-1][1]
        if hasattr(estimator, "predict_proba"):
            proba_vals = model.predict_proba(features)[0]
            classes = estimator.classes_
            proba = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, proba_vals)}
        return {
            "status": "success",
            "predicted_party": str(prediction),
            "probabilities": proba
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def generic_voter_predict(data: VoterPredictionInput):
    state = (data.state or "bihar").lower()
    if state == "maharashtra":
        return maha_voter_predict(data)
    else:
        return bihar_voter_predict(data)