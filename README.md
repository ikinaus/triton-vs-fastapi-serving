# triton-vs-fastapi-serving

One ONNX classifier, one identical HTTP contract, two serving backends — to compare what
each deployment path actually costs and gives you:

- **FastAPI** — ONNX Runtime in-process, preprocessing in Python.
- **Triton** — NVIDIA Triton Inference Server, preprocessing and the model wired
  together as an ensemble, called over gRPC.

Both paths are held to byte-identical outputs, including how missing values are handled
end to end — the interesting part, since the two runtimes disagree on null semantics by
default.

## Run

Two mutually exclusive profiles. Both expose the same ports, so run one at a time.

Triton stack (Triton Inference Server + FastAPI wrapper + web UI):

```bash
docker compose --profile triton up -d --build
```

FastAPI-only stack (ONNX Runtime in-process + web UI):

```bash
docker compose --profile fastapi up -d --build
```

Stop:

```bash
docker compose --profile triton --profile fastapi down
```

## Services

| Service | URL | Profile |
|---|---|---|
| Web UI | http://localhost:8501 | both |
| API (Swagger UI) | http://localhost:8080/docs | both |
| Triton HTTP | http://localhost:8000 | triton |
| Triton gRPC | localhost:8001 | triton |
| Triton metrics | http://localhost:8002/metrics | triton |

Triton readiness check: `curl http://localhost:8000/v2/health/ready`
