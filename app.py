"""
Run with:
    uvicorn api:app --reload --port 5000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, os, traceback
import pandas as pd

app = FastAPI(title="Election Analyzer API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_model_cache: dict = {}

def load_model(filename: str):
    if filename in _model_cache:
        return _model_cache[filename]
    path = os.path.join(MODEL_DIR, filename)
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
    Education: str
    Occupation: str

@app.get("/")
def root():
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
        raise HTTPException(503, "Bihar voter prediction model not found. Place 'bihar_voter_prediction.pkl' in the models/ folder.")
    try:
        features = pd.DataFrame([{
            "Age_Group": data.Age_Group,
            "Gender": data.Gender,
            "Geography": data.Geography,
            "Caste": data.Caste,
            "Education": data.Education,
            "Occupation": data.Occupation,
        }])
        prediction = model.predict(features)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba_vals = model.predict_proba(features)[0]
            classes = model.classes_ if hasattr(model, "classes_") else []
            proba = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, proba_vals)}
        return {
            "status": "success",
            "predicted_party": str(prediction),
            "probabilities": proba
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {traceback.format_exc()}")

@app.post("/maharashtra/predict_voter")
def maha_voter_predict(data: VoterPredictionInput):
    model = load_model("maharashtra_voter_prediction.pkl")
    if model is None:
        raise HTTPException(503, "Maharashtra voter prediction model not found. Place 'maharashtra_voter_prediction.pkl' in the models/ folder.")
    try:
        features = pd.DataFrame([{
            "Age_Group": data.Age_Group,
            "Gender": data.Gender,
            "Geography": data.Geography,
            "Caste": data.Caste,
            "Education": data.Education,
            "Occupation": data.Occupation,
        }])
        prediction = model.predict(features)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba_vals = model.predict_proba(features)[0]
            classes = model.classes_ if hasattr(model, "classes_") else []
            proba = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, proba_vals)}
        return {
            "status": "success",
            "predicted_party": str(prediction),
            "probabilities": proba
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {traceback.format_exc()}")

@app.post("/predict")
def generic_voter_predict(data: VoterPredictionInput):
    state = (data.state or "bihar").lower()
    if state == "maharashtra":
        return maha_voter_predict(data)
    return bihar_voter_predict(data)
