import os
import numpy as np
import tritonclient.grpc.aio as grpcclient
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTOR_PATH = os.path.join(CURRENT_DIR, 'extractor.pkl')
MODEL_PATH = os.path.join(CURRENT_DIR, 'model.onnx')
TRITON_URL = os.environ.get('TRITON_URL', 'localhost:8001')

FIELD_SPECS = {
    "Pclass":   ("INT32", np.int32),
    "Name":     ("BYTES", object),
    "Sex":      ("BYTES", object),
    "Age":      ("FP32",  np.float32),
    "SibSp":    ("INT32", np.int32),
    "Parch":    ("INT32", np.int32),
    "Ticket":   ("BYTES", object),
    "Fare":     ("FP32",  np.float32),
    "Cabin":    ("BYTES", object),
    "Embarked": ("BYTES", object),
}

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.triton = grpcclient.InferenceServerClient(url=TRITON_URL)
    yield
    await app.state.triton.close()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict_single(request: Request, payload: PassengerSchema) -> Dict:
    row = payload.model_dump()

    inputs = []
    for name, (triton_type, np_dtype) in FIELD_SPECS.items():
        value = row[name]
        if value is None:
            continue

        arr = np.array([[value]], dtype=np_dtype)

        inp = grpcclient.InferInput(name, list(arr.shape), triton_type)
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    outputs = [
        grpcclient.InferRequestedOutput('label'),
        grpcclient.InferRequestedOutput('probabilities'),
    ]

    client = request.app.state.triton
    result = await client.infer(
        model_name="titanic_ensemble",
        inputs=inputs,
        outputs=outputs,
    )
    label = result.as_numpy('label')
    probs = result.as_numpy('probabilities')

    return {
        'survived': bool(label[0]),
        'proba': float(probs[0][1]),
    }

@app.post("/predict_batch")
async def predict_batch(request: Request, payload: List[PassengerSchema]) -> List[Dict]:
    rows = [p.model_dump() for p in payload]

    inputs = []
    for name, (triton_type, np_dtype) in FIELD_SPECS.items():
        values = [row[name] if row[name] is not None else np.nan for row in rows]
        arr = np.array(values, dtype=np_dtype).reshape(-1, 1)

        inp = grpcclient.InferInput(name, list(arr.shape), triton_type)
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    outputs = [
        grpcclient.InferRequestedOutput('label'),
        grpcclient.InferRequestedOutput('probabilities'),
    ]

    client = request.app.state.triton
    result = await client.infer(
        model_name="titanic_ensemble",
        inputs=inputs,
        outputs=outputs,
    )
    label = result.as_numpy('label')
    probs = result.as_numpy('probabilities')

    results = []
    for i in range(len(payload)):
        results.append({
            'survived': bool(label[i]),
            'proba': float(probs[i, 1]),
        })

    return results