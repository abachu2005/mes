# API reference

The full OpenAPI schema is served live at
[`/api/openapi.json`](https://huggingface.co/spaces/abachu2005/mes/api/openapi.json) and
interactive docs at `/api/docs`.

## Core endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/healthz` | Liveness + version |
| `GET`  | `/api/models`  | List ONNX models + benchmarks |
| `POST` | `/api/demo/seed` | Seed two demo participants + sessions |
| `GET`  | `/api/participants` | List participants |
| `POST` | `/api/participants` | Create a participant |
| `GET`  | `/api/participants/{id}` | Get one |
| `GET`  | `/api/participants/{id}/longitudinal` | MES across all sessions |
| `GET`  | `/api/sessions` | List (filter by `?participant_id=`) |
| `POST` | `/api/sessions` | Upload an EEG file (multipart) |
| `GET`  | `/api/sessions/{id}` | Status + progress |
| `GET`  | `/api/sessions/{id}/score` | Final MES + features |
| `GET`  | `/api/sessions/{id}/report.pdf` | Downloadable report |
| `DELETE` | `/api/sessions/{id}` | Remove session + score |

## Upload example

```bash
curl -X POST https://huggingface.co/spaces/abachu2005/mes/api/sessions \
  -F "file=@my_recording.edf" \
  -F "participant_id=PARTICIPANT_UUID" \
  -F "task=right_hand" \
  -F "target_limb=Right hand"
```

Polling the returned session `id` will show progress; once `status=done`,
fetch `/api/sessions/{id}/score`.

## Python client

The repo ships with an installable client in `mes_core.cli`:

```bash
mes version
mes score path/to/rec.edf --task right_hand --out result.json
```
