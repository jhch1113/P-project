# Commands

## Environment

Set the model path before starting the server:

```
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
```

## Run the server

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Health checks

```
curl http://localhost:8000/metrics/all
curl http://localhost:8000/eeg/health
```

## EEG monitor UI

Open in a browser:

```
http://localhost:8000/eeg/monitor
```

## EEG streaming session (simple test)

Start a session:

```
curl -s -X POST http://localhost:8000/eeg/session/start
```

Send 1-second chunks (5+ chunks needed for the first window):

```
python3 - <<'PY'
import json, time, urllib.request

sid = "PASTE_SESSION_ID"
url = f"http://localhost:8000/eeg/session/{sid}/append"

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
curl -s -X POST http://localhost:8000/eeg/session/PASTE_SESSION_ID/end
```

## Utility scripts

```
python3 scripts/scan_muse.py
python3 scripts/test_get_eeg.py
python3 scripts/multi_sensor_api.py
```
