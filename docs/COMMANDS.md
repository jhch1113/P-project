# Commands

## Docker Compose (권장, 3서비스)

```bash
cd /home/neuro/drowsiness_project
docker compose up --build -d
```

| 서비스 | 포트 | 역할 |
|--------|------|------|
| orchestrator | **8000** | 대시보드, 융합, `/ingest/*` |
| camera | **8001** | RealSense, `/video_feed` |
| eeg | **8002** (host network) | Muse2 LSL + 모델 추론, `/eeg/*` |

통합 대시보드: `http://<host>:8000`

## Environment

모델 파일(기본 경로는 컨테이너 내 `/app/pp_nrsc/MUSE_activity_model.keras` 또는 `./models` 마운트 후 `MUSE_MODEL_PATH=/models/...`):

```bash
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
```

## Run locally (monolith)

```
export SERVICE_MODE=monolith EEG_MODE=muse2
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Health checks

```
# 오케스트레이터 / 모놀리스
curl http://localhost:8000/metrics/all
curl http://localhost:8000/debug/raw

# EEG 실험 API — eeg 컨테이너(8002) 또는 monolith(8000)에서만
curl http://localhost:8002/eeg/health
```

## EEG monitor UI

`SERVICE_MODE=eeg`(포트 8002) 또는 `monolith`(8000)에서만 사용:

```
http://localhost:8002/eeg/monitor
```

## EEG streaming session (simple test)

Start a session:

```
curl -s -X POST http://localhost:8002/eeg/session/start
```

Send 1-second chunks (5+ chunks needed for the first window):

```
python3 - <<'PY'
import json, time, urllib.request

sid = "PASTE_SESSION_ID"
url = f"http://localhost:8002/eeg/session/{sid}/append"

payload = {
    "af7": [0.0] * 256,
    "af8": [0.0] * 256,
    "apply_minmax": False,
}

for _ in range(6):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        print(r.read().decode())
    time.sleep(1)
PY
```

End the session:

```
curl -s -X POST http://localhost:8002/eeg/session/PASTE_SESSION_ID/end
```

## Utility scripts

```
python3 deprecated/scripts/scan_muse.py
python3 deprecated/scripts/test_get_eeg.py
python3 deprecated/scripts/multi_sensor_api.py
```
