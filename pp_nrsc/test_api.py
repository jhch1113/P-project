"""FastAPI 앱 스모크 테스트.
서버 띄우지 않고 TestClient로 직접 호출해서 모든 엔드포인트가 도는지 검증.
"""
import os
import sys

os.environ["MUSE_MODEL_PATH"] = "/mnt/user-data/uploads/MUSE_activity_model.keras"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

# 외부 의존성(예: fastapi)이 설치되지 않은 환경에서는 테스트를 건너뜁니다.
try:
    from fastapi.testclient import TestClient
except Exception as e:  # ModuleNotFoundError 등
    print("외부 패키지 누락: fastapi 또는 관련 패키지가 설치되어 있지 않습니다.")
    print("로컬에서 전체 테스트를 실행하려면 다음을 실행하십시오:")
    print("  python3 -m pip install -r requirements.txt")
    sys.exit(0)

sys.path.insert(0, "/home/claude")
from muse_inference_api import app, FS, SEQ_LEN

client = TestClient(app)


def make_fake_eeg(n_seconds: float, seed: int = 0) -> tuple[list, list]:
    """그럴듯한 EEG 신호 생성: 알파(10Hz) + 베타(20Hz) + 노이즈."""
    rng = np.random.default_rng(seed)
    n = int(n_seconds * FS)
    t = np.arange(n) / FS
    af7 = (
        15 * np.sin(2 * np.pi * 10 * t)
        + 5 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 8, n)
    ).astype(np.float32)
    af8 = (
        12 * np.sin(2 * np.pi * 10 * t + 0.5)
        + 4 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 8, n)
    ).astype(np.float32)
    return af7.tolist(), af8.tolist()


# -------- 1) /health --------
print("=" * 60)
print("[1] GET /health")
r = client.get("/health")
print("status:", r.status_code)
print("body:", r.json())
assert r.status_code == 200 and r.json()["status"] == "ok"

# -------- 2) /predict/batch --------
print("\n" + "=" * 60)
print("[2] POST /predict/batch (10초 신호)")
af7, af8 = make_fake_eeg(10, seed=1)
r = client.post(
    "/predict/batch",
    json={"af7": af7, "af8": af8, "apply_minmax": True},
)
print("status:", r.status_code)
body = r.json()
print(f"  n_samples={body['n_samples']}, n_windows={body['n_windows']}")
print(f"  prob_raw range: {body['prob_raw_min']:.4f} ~ {body['prob_raw_max']:.4f}")
print(f"  첫 윈도우: {body['results'][0]}")
print(f"  마지막 윈도우: {body['results'][-1]}")
# 10초 - 5초 윈도우 + 1초 stride => 6개
assert body["n_windows"] == 6, f"기대 6, 실제 {body['n_windows']}"
assert r.status_code == 200

# -------- 3) /predict/csv --------
print("\n" + "=" * 60)
print("[3] POST /predict/csv (CSV 업로드)")
import io
import pandas as pd

df = pd.DataFrame({"AF7": af7, "AF8": af8})
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()
r = client.post(
    "/predict/csv",
    files={"file": ("test.csv", csv_bytes, "text/csv")},
)
print("status:", r.status_code)
body = r.json()
print(f"  n_windows={body['n_windows']}, prob_raw_max={body['prob_raw_max']:.4f}")
assert r.status_code == 200
assert body["n_windows"] == 6

# -------- 4) HTTP 세션 스트리밍 --------
print("\n" + "=" * 60)
print("[4] /session/start -> /session/{sid}/append (1초씩 7번) -> /session/{sid}/end")
r = client.post("/session/start")
sid = r.json()["session_id"]
print(f"  session_id={sid}")

all_results = []
for sec in range(7):
    af7c, af8c = make_fake_eeg(1, seed=100 + sec)
    r = client.post(
        f"/session/{sid}/append",
        json={"af7": af7c, "af8": af8c, "apply_minmax": False},
    )
    body = r.json()
    new_count = len(body["new_results"])
    print(
        f"  [t={sec+1}s] buffered={body['buffered_samples']}, "
        f"new_results={new_count}"
        + (f" -> {body['new_results'][0]}" if new_count else "")
    )
    all_results.extend(body["new_results"])
    assert r.status_code == 200

# 총 7초 보냈으니 5초 모인 시점부터 결과 시작 -> 5,6,7초 = 윈도우 3개
assert len(all_results) == 3, f"기대 3, 실제 {len(all_results)}"

r = client.post(f"/session/{sid}/end")
print(f"  end: {r.json()}")
assert r.status_code == 200

# -------- 5) WebSocket 스트리밍 --------
print("\n" + "=" * 60)
print("[5] WebSocket /ws/stream")
with client.websocket_connect("/ws/stream") as ws:
    total_new = 0
    for sec in range(7):
        af7c, af8c = make_fake_eeg(1, seed=200 + sec)
        ws.send_json({"af7": af7c, "af8": af8c, "apply_minmax": False})
        msg = ws.receive_json()
        nr = len(msg["new_results"])
        total_new += nr
        print(f"  [t={sec+1}s] buffered={msg['buffered_samples']}, new_results={nr}")
        assert msg["type"] == "result"
print(f"  WS 총 new_results 수: {total_new} (기대: 3)")
assert total_new == 3

# -------- 6) 에러 케이스 --------
print("\n" + "=" * 60)
print("[6] 에러 케이스: 너무 짧은 신호")
af7s, af8s = make_fake_eeg(2)
r = client.post("/predict/batch", json={"af7": af7s, "af8": af8s})
print(f"  status={r.status_code} (400 기대), detail={r.json().get('detail')}")
assert r.status_code == 400

print("\n" + "=" * 60)
print("✅ 전체 통과")
