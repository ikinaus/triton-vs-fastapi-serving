import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import onnxruntime as ort
import pickle

from pydantic import BaseModel
from typing import Optional, List

from features import FeatureExtractor, CATEGORICAL_FEATURES, NUMERIC_FEATURES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTOR_PATH = os.path.join(CURRENT_DIR, 'extractor.pkl')
MODEL_PATH = os.path.join(CURRENT_DIR, 'model.onnx')

def build_onnx_inputs(records: list) -> dict:
    inputs = {}
    for name in CATEGORICAL_FEATURES:
        inputs[name] = np.array([r[name] for r in records], dtype=str).reshape(-1, 1)
    for name in NUMERIC_FEATURES:
        column = [r[name] if r[name] is not None else np.nan for r in records]
        inputs[name] = np.array(column, dtype=np.float32).reshape(-1, 1)
    return inputs

@asynccontextmanager
async def lifespan(app:FastAPI):

    with open(EXTRACTOR_PATH, 'rb') as f:
        app.state.extractor = pickle.load(f)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    app.state.session = ort.InferenceSession(MODEL_PATH, sess_options)
    yield

    del app.state.extractor
    del app.state.session

app = FastAPI(lifespan=lifespan)

class PassengerSchema(BaseModel):
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
def predict_single(request: Request, payload: PassengerSchema):
    extractor = request.app.state.extractor
    session = request.app.state.session

    records = extractor.transform_online([payload.model_dump()])
    onnx_inputs = build_onnx_inputs(records)

    output_names = [out.name for out in session.get_outputs()]
    onnx_outputs = session.run(output_names, onnx_inputs)

    label = bool(onnx_outputs[0][0])
    prob_survived = float(onnx_outputs[1][0][1])

    return {
        'survived': label,
        'proba': prob_survived
    }


@app.post("/predict_batch")
def predict_batch(request: Request, payload: List[PassengerSchema]):

    if not payload:
        return []

    extractor = request.app.state.extractor
    session = request.app.state.session

    records = extractor.transform_online([item.model_dump() for item in payload])
    onnx_inputs = build_onnx_inputs(records)

    output_names = [out.name for out in session.get_outputs()]
    onnx_outputs = session.run(output_names, onnx_inputs)

    labels = onnx_outputs[0].flatten().astype(int)
    probs = onnx_outputs[1]

    results = []
    for i in range(len(payload)):
        results.append({
            "survived": bool(labels[i]),
            "proba": float(probs[i][1])
        })

    return results
