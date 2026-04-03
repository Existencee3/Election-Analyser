import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

class SurveyData(BaseModel):
    region: str
    district: str
    geography: str
    gender: str
    age_band: str
    age: float
    caste: str
    occupation: str
    cm_preference: str
    voting_reason: str

# model = joblib.load("models/model.pkl")

uvicorn.run(app, host="0.0.0.0", port=8000)