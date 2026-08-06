import os
import numpy as np
import tritonclient.grpc.aio as grpcclient
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, List

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

async def infer_passengers(client, rows: List[Dict]) -> List[Dict]:
    inputs = []
    for name, (triton_type, np_dtype) in FIELD_SPECS.items():
        values = [row[name] for row in rows]

        if all(v is None for v in values):
            continue

        arr = np.array(
            [v if v is not None else np.nan for v in values],
            dtype=np_dtype,
        ).reshape(-1, 1)

        inp = grpcclient.InferInput(name, list(arr.shape), triton_type)
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    outputs = [
        grpcclient.InferRequestedOutput('label'),
        grpcclient.InferRequestedOutput('probabilities'),
    ]

    result = await client.infer(
        model_name="titanic_ensemble",
        inputs=inputs,
        outputs=outputs,
    )
    label = result.as_numpy('label')
    probs = result.as_numpy('probabilities')

    return [
        {
            'survived': bool(label[i]),
            'proba': float(probs[i, 1]),
        }
        for i in range(len(rows))
    ]


@app.post("/predict")
async def predict_single(request: Request, payload: PassengerSchema) -> Dict:
    results = await infer_passengers(request.app.state.triton, [payload.model_dump()])
    return results[0]


@app.post("/predict_batch")
async def predict_batch(request: Request, payload: List[PassengerSchema]) -> List[Dict]:
    if not payload:
        return []
    rows = [p.model_dump() for p in payload]
    return await infer_passengers(request.app.state.triton, rows)