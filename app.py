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
import numpy as np

# ── Strategy tips lookup ──────────────────────────────────────────────────────
AGE_TIPS = {
    "18-19":      "Run social media campaigns (Instagram/YouTube); focus on education & first jobs.",
    "20-29":      "Prioritise employment, startup support, and digital schemes.",
    "30-39":      "Focus on housing, job security, and children's education.",
    "40-49":      "Highlight economic stability, MSP for farmers, health insurance.",
    "50-59":      "Push pension security, healthcare, and agricultural loan waivers.",
    "60-69":      "Promote senior welfare, pension hike, free medical camps.",
    "70 & Above": "Focus on elder care, pension reliability, and pilgrimage schemes.",
}
EDU_TIPS = {
    "Not Gone to School":    "Hold community meetings; use local language and visuals.",
    "Upto 9th":              "Use pamphlets and radio; keep messages simple.",
    "10th Pass":             "Highlight skill development (ITI/polytechnic) and job schemes.",
    "12th Pass":             "Push government job opportunities and coaching support.",
    "Graduate":              "Focus on white-collar employment and digital services.",
    "Post-Graduate":         "Highlight research funding and governance reforms.",
    "Professional Education":"Engage via policy papers, industry events, and tax relief.",
}
OCC_TIPS = {
    "Farmer":              "Announce MSP hike, irrigation schemes, crop insurance.",
    "Labour":              "Promise minimum wage increase and MNREGA expansion.",
    "Student":             "Offer free coaching, exam fee waivers, scholarships.",
    "Housewife":           "Highlight women SHG support, gas subsidy, cash-transfer schemes.",
    "Skilled Professional": "Promise easier business licenses and GST simplification.",
    "Unemployed":          "Announce employment guarantee and skill training programs.",
    "Government Employee": "Focus on pay-commission benefits and job security.",
    "Business":            "Highlight lower taxes and ease-of-doing-business policies.",
}

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

if os.path.exists(MODEL_DIR):
    app.mount("/models", StaticFiles(directory=MODEL_DIR), name="models")

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

# ── Load data for suggestions ─────────────────────────────────────────────────
BIHAR_DATA_PATH = os.path.join(MODEL_DIR, "bihar_election_dataset.csv")
bihar_df = pd.read_csv(BIHAR_DATA_PATH) if os.path.exists(BIHAR_DATA_PATH) else None

def get_suggestion_for_voter(party_name, voter_profile: dict):
    """
    Adapted from Streamlit code to calculate targeted suggestions.
    """
    if bihar_df is None:
        return []

    df = bihar_df
    SEGMENT_MAP = {
        "Caste":      voter_profile["Caste"],
        "Age_Group":  voter_profile["Age_Group"],
        "Gender":     voter_profile["Gender"],
        "Geography":  voter_profile["Geography"],
        "Education":  voter_profile["Education"],
        "Occupation": voter_profile["Occupation"],
    }

    suggestions = []
    for col, val in SEGMENT_MAP.items():
        if not val: continue
        seg = df[df[col] == val]
        if len(seg) < 20: # Lower threshold than Streamlit for variety
            continue

        party_pct = (seg["Voted_Party"] == party_name).mean() * 100
        total     = len(seg)

        # best rival in this segment
        rival_counts = seg[seg["Voted_Party"] != party_name]["Voted_Party"].value_counts()
        if rival_counts.empty:
            continue
        rival      = rival_counts.idxmax()
        rival_pct  = (seg["Voted_Party"] == rival).mean() * 100
        gap        = round(rival_pct - party_pct, 1)

        # build targeted tip
        if col == "Caste":
            tip = (f"Among {val} voters, {party_name} gets {party_pct:.0f}% vs "
                   f"{rival}'s {rival_pct:.0f}%. Field candidates from the {val} community, "
                   f"launch targeted welfare schemes, and engage local {val} leaders.")
        elif col == "Age_Group":
            base = AGE_TIPS.get(val, "Tailor outreach to this age group.")
            tip  = f"Among {val} voters, {party_name} gets {party_pct:.0f}% vs {rival}'s {rival_pct:.0f}%. {base}"
        elif col == "Gender":
            base = ("Launch women-centric welfare schemes (cash transfers, SHGs, safety)."
                    if val == "Female"
                    else "Address male voter concerns around employment and security.")
            tip  = f"Among {val} voters, {party_name} gets {party_pct:.0f}% vs {rival}'s {rival_pct:.0f}%. {base}"
        elif col == "Geography":
            tip  = (f"In {val} areas, {party_name} gets {party_pct:.0f}% vs {rival}'s {rival_pct:.0f}%. "
                    f"Increase candidate visits, local infrastructure promises, and booth-level outreach.")
        elif col == "Education":
            base = EDU_TIPS.get(val, "Tailor messaging to this education level.")
            tip  = f"Among {val}-educated voters, {party_name} gets {party_pct:.0f}% vs {rival}'s {rival_pct:.0f}%. {base}"
        elif col == "Occupation":
            base = OCC_TIPS.get(val, "Address key concerns of this group.")
            tip  = f"Among {val} voters, {party_name} gets {party_pct:.0f}% vs {rival}'s {rival_pct:.0f}%. {base}"
        else:
            tip = f"{party_name} should focus more on {val} voters."

        suggestions.append({
            "dimension":   col,
            "value":       val,
            "party_pct":   round(party_pct, 1),
            "rival":       rival,
            "rival_pct":   round(rival_pct, 1),
            "gap":         gap,
            "total":       total,
            "tip":         tip,
            "winning":     bool(party_pct >= rival_pct),
        })

    # Sort: biggest gap first
    suggestions.sort(key=lambda x: x["gap"], reverse=True)
    return suggestions


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
        
        suggestions = get_suggestion_for_voter(str(prediction), input_dict)

        return {
            "status": "success",
            "predicted_party": str(prediction),
            "probabilities": proba,
            "suggestions": suggestions
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
    
@app.get("/{full_path:path}")
def serve_index_fallback(full_path: str):
    index_path = os.path.join(TEMPLATE_DIR, "index.html")
    return FileResponse(index_path, media_type="text/html")