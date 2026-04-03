import numpy
import pandas
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
@app.get("/")
async def home():
    pass