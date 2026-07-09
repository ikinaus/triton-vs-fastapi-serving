import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import onnxruntime as ort
import pickle

from pydantic import BaseModel
from typing import Optional, List

from features import FeatureExtractor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTOR_PATH = os.path.join(CURRENT_DIR, 'extractor.pkl')
MODEL_PATH = os.path.join(CURRENT_DIR, 'model.onnx')

categorical_features = ['Sex', 'Embarked', 'Deck', 'TicketPrefix', 'Title']

@asynccontextmanager
async def lifespan(app:FastAPI):
    
    with open(EXTRACTOR_PATH, 'rb') as f:
        app.state.extractor = pickle.load(f)
    app.state.session = ort.InferenceSession(MODEL_PATH)
    yield

    del app.state.extractor
    del app.state.session

app = FastAPI(lifespan=lifespan)

class PassangerSchema(BaseModel):
    Pclass: int
    Name: str
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Ticket: str
    Fare: Optional[float] = None
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None

@app.post("/predict")
async def predict_single(request: Request, payload: PassangerSchema):
    extractor = request.app.state.extractor
    session = request.app.state.session

    data_dict = payload.model_dump()
    df = pd.DataFrame([data_dict])
    df_prep = extractor.transform(df)

    onnx_inputs = {}
    for col in df_prep.columns:
        if col in categorical_features:
            onnx_inputs[col] = df_prep[col].astype(str).values.reshape(-1, 1)
        else:
            onnx_inputs[col] = df_prep[col].values.reshape(-1, 1).astype(np.float32)

    output_names = [out.name for out in session.get_outputs()]
    onnx_outputs = session.run(output_names, onnx_inputs)

    label = int(onnx_outputs[0][0])
    prob_survived = float(onnx_outputs[1][0][1])

    return {
        'survived': label,
        'proba': prob_survived 
    }


@app.post("/predict_batch")
async def predict_batch(request: Request, payload: List[PassangerSchema]):
    extractor = request.app.state.extractor
    session = request.app.state.session

    df = pd.DataFrame([item.model_dump() for item in payload])
    df_prep = extractor.transform(df)

    onnx_inputs = {}
    for col in df_prep.columns:
        if col in categorical_features:
            onnx_inputs[col] = df_prep[col].astype(str).values.reshape(-1, 1)
        else:
            onnx_inputs[col] = df_prep[col].values.reshape(-1, 1).astype(np.float32)

    output_names = [out.name for out in session.get_outputs()]
    onnx_outputs = session.run(output_names, onnx_inputs)

    labels = onnx_outputs[0].flatten().astype(int)
    probs = onnx_outputs[1]

    results = []
    for i in range(len(payload)):
        results.append({
            "survived": int(labels[i]),
            "proba": float(probs[i][1])
        })

    return results