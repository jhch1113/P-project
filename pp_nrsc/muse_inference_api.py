"""
MUSE 2 Sleep/Awake Inference API
================================
EEG (AF7, AF8) -> 전처리 -> 추론 -> 확률/상태 반환까지만 책임짐.
점수화 로직은 호출하는 쪽에서 붙이면 됨.

엔드포인트
----------
- GET  /health                    : 헬스체크 + 모델 로드 상태
- POST /predict/batch             : JSON으로 전체 신호 보내고 모든 윈도우 예측 받기
- POST /predict/csv               : CSV 파일 업로드 (AF7, AF8 컬럼 필수)
- POST /session/start             : 스트리밍 세션 생성 -> session_id 반환
- POST /session/{sid}/append      : 1초(또는 임의 길이) 청크 누적 + 가능하면 즉시 추론
- POST /session/{sid}/end         : 세션 종료 + 누적된 모든 결과 반환
- WS   /ws/stream                 : WebSocket 실시간 스트리밍 (1초 단위 추론)
"""

from __future__ import annotations

import io
import os
import time
import uuid
import asyncio
from threading import Lock
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from scipy.signal import butter, filtfilt

try:
    from .eeg_data_source import create_reader
    from .drowsiness_scorer import DrowsinessScorer
    from .advanced_preprocessing import AdvancedPreprocessor
    from .advanced_postprocessing import AdvancedPostprocessor
except ImportError:  # Allow running as a standalone script
    from eeg_data_source import create_reader
    from drowsiness_scorer import DrowsinessScorer
    from advanced_preprocessing import AdvancedPreprocessor
    from advanced_postprocessing import AdvancedPostprocessor

# ============================================================
# 1. 설정 (학습 코드와 동일)
# ============================================================
FS = 256                       # 샘플레이트
SEQ_LEN = FS * 5               # 5초 = 1280 샘플
STRIDE = FS                    # 1초 stride

# 환경변수로 모델 경로 덮어쓸 수 있게
MODEL_PATH = os.environ.get(
    "MUSE_MODEL_PATH",
    "/mnt/user-data/uploads/MUSE_activity_model.keras",
)

# Hysteresis 임계값 (학습 코드 기본값)
HYS_HIGH_DEFAULT = 0.55
HYS_LOW_DEFAULT = 0.45

# 스트리밍 시 전처리에 사용할 버퍼 길이 (filtfilt 경계 효과 줄이려고 5초보다 넉넉히 잡음)
BUFFER_SEC = 10
BUFFER_LEN = FS * BUFFER_SEC

# 세션 만료 (초)
SESSION_TTL = 60 * 30


# ============================================================
# 2. 핵심 함수 (학습 코드와 동일한 전처리/후처리)
# ============================================================
# Butter 필터 계수는 한 번만 계산해서 재사용
_NYQ = 0.5 * FS
_BUTTER_B, _BUTTER_A = butter(4, [0.5 / _NYQ, 40 / _NYQ], btype="band")

# 고급 전처리기 초기화
_advanced_preprocessor = AdvancedPreprocessor(verbose=False)


def preprocess_basic(x: np.ndarray) -> np.ndarray:
    """기본 전처리 (호환성용)"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"입력은 (N, 2) shape이어야 합니다. 받은 shape: {x.shape}")

    if x.shape[0] < 33:
        raise ValueError(f"신호가 너무 짧습니다 (N={x.shape[0]}). 최소 33 샘플 필요.")

    x = filtfilt(_BUTTER_B, _BUTTER_A, x, axis=0)
    x = x - np.median(x, axis=0)
    x = np.clip(x, -100, 100)
    return ((x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)).astype(np.float32)


def preprocess(x: np.ndarray, use_advanced: bool = True) -> Tuple[np.ndarray, float]:
    """
    향상된 전처리 (Muse2 노이즈 대응)
    
    Args:
        x: (N, 2) 입력 신호 (AF7, AF8)
        use_advanced: True면 고급 전처리, False면 기본 전처리 사용
    
    Returns:
        (처리된_신호, 품질_점수)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"입력은 (N, 2) shape이어야 합니다. 받은 shape: {x.shape}")

    if x.shape[0] < 33:
        raise ValueError(f"신호가 너무 짧습니다 (N={x.shape[0]}). 최소 33 샘플 필요.")

    if not use_advanced:
        # 기본 전처리 (호환성)
        return preprocess_basic(x), 1.0
    
    # 고급 전처리: 각 채널 개별 처리
    af7_proc = _advanced_preprocessor.process(x[:, 0], remove_artifacts=True, aggressive=False)
    af8_proc = _advanced_preprocessor.process(x[:, 1], remove_artifacts=True, aggressive=False)
    
    quality = _advanced_preprocessor.get_quality_score()
    
    # 결과를 (N, 2)로 스택
    return np.stack([af7_proc, af8_proc], axis=1).astype(np.float32), quality


def hysteresis(pred: np.ndarray, high: float = HYS_HIGH_DEFAULT, low: float = HYS_LOW_DEFAULT) -> np.ndarray:
    """
    히스테리시스 필터 적용
    high 이상이면 1, low 이하면 0, 사이 구간은 직전 상태 유지.
    
    참고: advanced_postprocessing에서 더 견고한 버전 제공
    """
    state = 0
    out = []
    for p in pred:
        if p > high:
            state = 1
        elif p < low:
            state = 0
        out.append(state)
    return np.array(out, dtype=np.int32)


def hysteresis_with_quality(pred: np.ndarray, 
                           quality_scores: np.ndarray = None,
                           high: float = HYS_HIGH_DEFAULT, 
                           low: float = HYS_LOW_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    품질 점수를 고려한 개선된 히스테리시스
    
    Returns:
        (states, confidences)
    """
    if quality_scores is None:
        quality_scores = np.ones_like(pred)
    
    postprocessor = AdvancedPostprocessor(history_size=5, confidence_threshold=0.5)
    states = []
    confidences = []
    
    for p, q in zip(pred, quality_scores):
        state, confidence, _ = postprocessor.process(p, q, high, low)
        states.append(state)
        confidences.append(confidence)
    
    return np.array(states, dtype=np.int32), np.array(confidences, dtype=np.float32)


def make_windows(x: np.ndarray, seq_len: int = SEQ_LEN, stride: int = STRIDE) -> np.ndarray:
    """전처리된 (N, 2) -> (W, 1280, 2) 윈도우 배열."""
    n = len(x)
    if n < seq_len:
        return np.empty((0, seq_len, 2), dtype=np.float32)
    starts = list(range(0, n - seq_len + 1, stride))
    return np.stack([x[s : s + seq_len] for s in starts], axis=0).astype(np.float32)


def minmax_scale(p: np.ndarray) -> np.ndarray:
    """학습 코드의 preds_scaled. 분포가 좁을 때 결과를 0~1로 펼침."""
    if len(p) == 0:
        return p
    return (p - p.min()) / (p.max() - p.min() + 1e-6)


# ============================================================
# 3. 모델 로딩 (싱글턴, 스레드 안전)
# ============================================================
_model: Optional[tf.keras.Model] = None
_model_lock = Lock()


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(f"모델 파일을 찾을 수 없음: {MODEL_PATH}")
                _model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
                # warmup
                dummy = np.zeros((1, SEQ_LEN, 2), dtype=np.float32)
                _model.predict(dummy, verbose=0)
    return _model


# 추론 자체는 thread-safe하지 않은 케이스가 있어서 lock으로 감쌈
_predict_lock = Lock()


def predict_windows(windows: np.ndarray, batch_size: int = 128) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0,), dtype=np.float32)
    model = get_model()
    with _predict_lock:
        preds = model.predict(windows, batch_size=batch_size, verbose=0).flatten()
    return preds.astype(np.float32)


# ============================================================
# 4. Pydantic 스키마
# ============================================================
class BatchPredictRequest(BaseModel):
    af7: list[float] = Field(..., description="AF7 채널 raw 신호 (256Hz 기준)")
    af8: list[float] = Field(..., description="AF8 채널 raw 신호 (256Hz 기준)")
    apply_minmax: bool = Field(True, description="원본 학습 코드처럼 확률을 min-max로 펼치기")
    hys_high: float = Field(HYS_HIGH_DEFAULT, ge=0.0, le=1.0)
    hys_low: float = Field(HYS_LOW_DEFAULT, ge=0.0, le=1.0)


class WindowResult(BaseModel):
    t_start_sec: float
    t_end_sec: float
    prob_raw: float
    prob_scaled: Optional[float] = None
    state: int  # 0: sleep, 1: awake (히스테리시스 적용 후)


class BatchPredictResponse(BaseModel):
    n_samples: int
    n_windows: int
    fs: int = FS
    seq_len: int = SEQ_LEN
    stride: int = STRIDE
    prob_raw_min: float
    prob_raw_max: float
    results: list[WindowResult]


class StartSessionResponse(BaseModel):
    session_id: str
    fs: int = FS
    seq_len: int = SEQ_LEN
    stride: int = STRIDE
    note: str = "AF7, AF8 청크를 /session/{sid}/append로 보내세요."


class AppendRequest(BaseModel):
    af7: list[float]
    af8: list[float]
    apply_minmax: bool = False  # 스트리밍에서는 현재 윈도우 단독이라 보통 False
    hys_high: float = HYS_HIGH_DEFAULT
    hys_low: float = HYS_LOW_DEFAULT


class AppendResponse(BaseModel):
    session_id: str
    buffered_samples: int
    new_results: list[WindowResult]


# ============================================================
# 5. 스트리밍 세션 (in-memory, 단일 프로세스 가정)
# ============================================================
class StreamSession:
    """1초 단위 추론을 위한 롤링 버퍼."""

    def __init__(self, sid: str):
        self.sid = sid
        self.buffer = np.empty((0, 2), dtype=np.float32)  # raw
        self.total_samples = 0                            # 누적 샘플 수 (절대 인덱스용)
        self.next_window_start = 0                        # 다음에 추론할 윈도우의 절대 시작 인덱스
        self.hyst_state = 0                               # hysteresis 상태 유지
        self.created_at = time.time()
        self.last_active = time.time()
        self.lock = Lock()

    def append(self, raw_chunk: np.ndarray, hys_high: float, hys_low: float, apply_minmax: bool) -> list[WindowResult]:
        """
        raw_chunk: (M, 2) — 새로 들어온 raw EEG.
        반환: 이번 호출로 새로 가능해진 윈도우들의 결과.
        """
        if raw_chunk.ndim != 2 or raw_chunk.shape[1] != 2:
            raise ValueError(f"chunk shape이 (M, 2) 여야 함. 받은 shape: {raw_chunk.shape}")

        with self.lock:
            self.last_active = time.time()
            self.buffer = np.concatenate([self.buffer, raw_chunk.astype(np.float32)], axis=0)
            self.total_samples += len(raw_chunk)

            # 버퍼는 최근 BUFFER_LEN 샘플만 유지
            if len(self.buffer) > BUFFER_LEN:
                self.buffer = self.buffer[-BUFFER_LEN:]

            # buffer의 절대 시작 인덱스 = total_samples - len(buffer)
            buf_abs_start = self.total_samples - len(self.buffer)

            # 추론 가능한 윈도우 (절대 인덱스 기준): next_window_start, +stride, ... <= total_samples - SEQ_LEN
            results: list[WindowResult] = []
            new_window_starts: list[int] = []
            ws = self.next_window_start
            while ws + SEQ_LEN <= self.total_samples:
                # 이 윈도우가 현재 버퍼 안에 들어와 있어야 함
                if ws < buf_abs_start:
                    # 버퍼 밖으로 밀려난 경우(거의 없음 - chunk가 BUFFER_SEC보다 클 때) skip
                    ws += STRIDE
                    continue
                new_window_starts.append(ws)
                ws += STRIDE
            self.next_window_start = ws

            if not new_window_starts:
                return results

            # 버퍼 전체를 한 번 전처리 (filtfilt 경계 효과 최소화)
            # 단, 버퍼가 너무 짧으면 (5초 미만) 추론 불가 — 위 while에서 이미 걸러짐
            try:
                buf_pre, quality_score = preprocess(self.buffer, use_advanced=True)
            except ValueError:
                return results

            # 각 윈도우를 슬라이스 (버퍼 내부의 상대 인덱스로 변환)
            windows = []
            for ws_abs in new_window_starts:
                rel_start = ws_abs - buf_abs_start
                windows.append(buf_pre[rel_start : rel_start + SEQ_LEN])
            windows_arr = np.stack(windows, axis=0)

            preds = predict_windows(windows_arr)

            if apply_minmax and len(preds) > 1:
                preds_scaled = minmax_scale(preds)
            else:
                preds_scaled = None

            # hysteresis는 세션 상태를 이어가야 하므로 직접 적용
            for i, ws_abs in enumerate(new_window_starts):
                p_raw = float(preds[i])
                p_for_hys = float(preds_scaled[i]) if preds_scaled is not None else p_raw
                if p_for_hys > hys_high:
                    self.hyst_state = 1
                elif p_for_hys < hys_low:
                    self.hyst_state = 0
                results.append(
                    WindowResult(
                        t_start_sec=ws_abs / FS,
                        t_end_sec=(ws_abs + SEQ_LEN) / FS,
                        prob_raw=p_raw,
                        prob_scaled=(float(preds_scaled[i]) if preds_scaled is not None else None),
                        state=int(self.hyst_state),
                    )
                )
            return results


_sessions: dict[str, StreamSession] = {}
_sessions_lock = Lock()


def _gc_sessions():
    now = time.time()
    with _sessions_lock:
        dead = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL]
        for sid in dead:
            del _sessions[sid]


# ============================================================
# 6. FastAPI 앱
# ============================================================
app = FastAPI(
    title="MUSE Sleep/Awake Inference API",
    description="EEG (AF7, AF8) → 전처리 → 추론까지. 점수화 로직은 호출자가 붙임.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page():
        """브라우저 실시간 모니터링 페이지."""
        return """
<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Muse Realtime Monitor</title>
    <style>
        :root {
            --bg: #0b1220;
            --card: #121a2b;
            --fg: #e9eefb;
            --muted: #8ea0c7;
            --ok: #16a34a;
            --warn: #f59e0b;
            --risk: #ef4444;
            --line: #22314f;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;
            background: radial-gradient(circle at 20% 10%, #1b2b4a 0, var(--bg) 40%);
            color: var(--fg);
            min-height: 100vh;
            padding: 24px;
        }
        .wrap { max-width: 1100px; margin: 0 auto; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent), var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 16px;
        }
        .big { font-size: 44px; font-weight: 800; letter-spacing: 0.5px; }
        .muted { color: var(--muted); }
        .pill { display: inline-block; border-radius: 999px; padding: 8px 12px; font-weight: 700; }
        .ok { background: rgba(22,163,74,.2); color: #7ff5a3; }
        .warn { background: rgba(245,158,11,.2); color: #ffd17c; }
        .risk { background: rgba(239,68,68,.2); color: #ff9a9a; }
        .alert {
            margin-top: 12px;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid transparent;
            transition: all .2s;
        }
        .alert.on {
            background: rgba(239,68,68,.18);
            border-color: rgba(239,68,68,.55);
            box-shadow: 0 0 0 2px rgba(239,68,68,.2) inset;
        }
        #log {
            height: 260px;
            overflow: auto;
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px;
            line-height: 1.4;
            background: #0d1424;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid var(--line);
        }
        button {
            border: 0;
            border-radius: 10px;
            padding: 10px 14px;
            cursor: pointer;
            font-weight: 700;
            background: #1f6feb;
            color: #fff;
        }
        button.stop { background: #8b2a2a; }
        @media (max-width: 900px) {
            .row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Muse 실시간 모니터링</h1>
        <p class="muted">헤드셋 데이터 -> 추론 -> 점수화 -> 경고를 브라우저에서 확인합니다.</p>

        <div class="card" style="margin-bottom:16px; display:flex; gap:8px; align-items:center;">
            <button id="startBtn">시작</button>
            <button id="stopBtn" class="stop">중지</button>
            <span id="conn" class="muted">연결 대기</span>
        </div>

        <div class="row">
            <div class="card">
                <div class="muted">현재 상태</div>
                <div id="riskLevel" class="pill ok" style="margin-top:8px;">안전</div>
                <div style="height:8px"></div>
                <div class="muted">점수</div>
                <div id="score" class="big">0.0</div>
                <div class="muted">prob_raw(awake)</div>
                <div id="probRaw" class="big" style="font-size:28px">0.000</div>
                <div id="alert" class="alert">경고 없음</div>
            </div>

            <div class="card">
                <div class="muted">실시간 로그</div>
                <div id="log"></div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        const conn = document.getElementById('conn');
        const scoreEl = document.getElementById('score');
        const probRawEl = document.getElementById('probRaw');
        const riskEl = document.getElementById('riskLevel');
        const logEl = document.getElementById('log');
        const alertEl = document.getElementById('alert');

        function appendLog(msg) {
            const t = new Date().toLocaleTimeString();
            logEl.textContent += `[${t}] ${msg}\n`;
            logEl.scrollTop = logEl.scrollHeight;
        }

        function applyRisk(level) {
            riskEl.textContent = level;
            riskEl.className = 'pill ' + (level === '위험' ? 'risk' : level === '주의' ? 'warn' : 'ok');
        }

        function start() {
            if (ws && ws.readyState < 2) return;
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            const basePath = location.pathname.replace(/\/monitor$/, '');
            ws = new WebSocket(`${proto}://${location.host}${basePath}/ws/live-muse`);

            ws.onopen = () => {
                conn.textContent = '연결됨';
                appendLog('실시간 스트림 시작');
            };
            ws.onclose = () => {
                conn.textContent = '연결 종료';
                appendLog('스트림 종료');
            };
            ws.onerror = () => {
                conn.textContent = '연결 오류';
            };
            ws.onmessage = (ev) => {
                const data = JSON.parse(ev.data);
                if (data.type === 'status') {
                    appendLog(data.message);
                    return;
                }
                if (data.type === 'error') {
                    appendLog('ERROR: ' + data.detail);
                    return;
                }
                if (data.type === 'tick') {
                    scoreEl.textContent = Number(data.smoothed_score).toFixed(1);
                    probRawEl.textContent = Number(data.prob_raw).toFixed(3);
                    applyRisk(data.risk_level);
                    if (data.should_alert) {
                        alertEl.textContent = '경고 발생: 즉시 휴식 권장';
                        alertEl.classList.add('on');
                    } else {
                        alertEl.textContent = '경고 없음';
                        alertEl.classList.remove('on');
                    }
                    appendLog(`risk=${data.risk_level} score=${Number(data.smoothed_score).toFixed(1)} alert=${data.should_alert}`);
                }
            };
        }

        function stop() {
            if (ws) ws.close();
        }

        document.getElementById('startBtn').onclick = start;
        document.getElementById('stopBtn').onclick = stop;
    </script>
</body>
</html>
"""


@app.on_event("startup")
def _startup():
    # 모델 미리 로드 (실패하면 바로 알게)
    get_model()


@app.get("/health")
def health():
    try:
        m = get_model()
        return {
            "status": "ok",
            "model_path": MODEL_PATH,
            "input_shape": list(m.input_shape),
            "output_shape": list(m.output_shape),
            "fs": FS,
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ----- Batch (JSON) -----
@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if len(req.af7) != len(req.af8):
        raise HTTPException(400, f"AF7/AF8 길이가 다름: {len(req.af7)} vs {len(req.af8)}")
    if req.hys_low > req.hys_high:
        raise HTTPException(400, "hys_low는 hys_high 이하여야 합니다.")

    raw = np.stack([req.af7, req.af8], axis=1).astype(np.float32)
    return _run_batch(raw, req.apply_minmax, req.hys_high, req.hys_low)


# ----- Batch (CSV 업로드) -----
@app.post("/predict/csv", response_model=BatchPredictResponse)
async def predict_csv(
    file: UploadFile = File(...),
    apply_minmax: bool = True,
    hys_high: float = HYS_HIGH_DEFAULT,
    hys_low: float = HYS_LOW_DEFAULT,
):
    if hys_low > hys_high:
        raise HTTPException(400, "hys_low는 hys_high 이하여야 합니다.")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"CSV 파싱 실패: {e}")

    if "AF7" not in df.columns or "AF8" not in df.columns:
        raise HTTPException(400, f"CSV에 AF7, AF8 컬럼이 필요합니다. 받은 컬럼: {list(df.columns)}")

    raw = df[["AF7", "AF8"]].values.astype(np.float32)
    return _run_batch(raw, apply_minmax, hys_high, hys_low)


def _run_batch(raw: np.ndarray, apply_minmax: bool, hys_high: float, hys_low: float) -> BatchPredictResponse:
    if len(raw) < SEQ_LEN:
        raise HTTPException(400, f"신호가 너무 짧음. 최소 {SEQ_LEN} 샘플 ({SEQ_LEN/FS}초) 필요. 받은 샘플 수: {len(raw)}")

    pre, quality_score = preprocess(raw, use_advanced=True)
    windows = make_windows(pre)
    preds = predict_windows(windows)

    if apply_minmax:
        preds_scaled = minmax_scale(preds)
        states = hysteresis(preds_scaled, hys_high, hys_low)
    else:
        preds_scaled = None
        states = hysteresis(preds, hys_high, hys_low)

    results: list[WindowResult] = []
    for i in range(len(preds)):
        ws_abs = i * STRIDE
        results.append(
            WindowResult(
                t_start_sec=ws_abs / FS,
                t_end_sec=(ws_abs + SEQ_LEN) / FS,
                prob_raw=float(preds[i]),
                prob_scaled=(float(preds_scaled[i]) if preds_scaled is not None else None),
                state=int(states[i]),
            )
        )

    return BatchPredictResponse(
        n_samples=int(len(raw)),
        n_windows=int(len(preds)),
        prob_raw_min=float(preds.min()) if len(preds) else 0.0,
        prob_raw_max=float(preds.max()) if len(preds) else 0.0,
        results=results,
    )


# ----- Streaming (HTTP 세션) -----
@app.post("/session/start", response_model=StartSessionResponse)
def session_start():
    _gc_sessions()
    sid = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[sid] = StreamSession(sid)
    return StartSessionResponse(session_id=sid)


@app.post("/session/{sid}/append", response_model=AppendResponse)
def session_append(sid: str, req: AppendRequest):
    if len(req.af7) != len(req.af8):
        raise HTTPException(400, f"AF7/AF8 길이가 다름: {len(req.af7)} vs {len(req.af8)}")
    if req.hys_low > req.hys_high:
        raise HTTPException(400, "hys_low는 hys_high 이하여야 합니다.")

    with _sessions_lock:
        sess = _sessions.get(sid)
    if sess is None:
        raise HTTPException(404, f"session_id 없음: {sid}")

    chunk = np.stack([req.af7, req.af8], axis=1).astype(np.float32)
    try:
        new_results = sess.append(chunk, req.hys_high, req.hys_low, req.apply_minmax)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return AppendResponse(
        session_id=sid,
        buffered_samples=sess.total_samples,
        new_results=new_results,
    )


@app.post("/session/{sid}/end")
def session_end(sid: str):
    with _sessions_lock:
        sess = _sessions.pop(sid, None)
    if sess is None:
        raise HTTPException(404, f"session_id 없음: {sid}")
    return {"session_id": sid, "total_samples": sess.total_samples, "ended": True}


# ----- Streaming (WebSocket) -----
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    프로토콜 (JSON 메시지):
      클라이언트 -> 서버:
        {"af7": [...], "af8": [...], "apply_minmax": false, "hys_high": 0.55, "hys_low": 0.45}
      서버 -> 클라이언트:
        {"type": "result", "buffered_samples": int, "new_results": [WindowResult, ...]}
        {"type": "error",  "detail": "..."}
    """
    await websocket.accept()
    sess = StreamSession(uuid.uuid4().hex)
    try:
        while True:
            msg = await websocket.receive_json()
            af7 = msg.get("af7")
            af8 = msg.get("af8")
            if af7 is None or af8 is None or len(af7) != len(af8):
                await websocket.send_json({"type": "error", "detail": "af7/af8 누락 또는 길이 불일치"})
                continue
            try:
                chunk = np.stack([af7, af8], axis=1).astype(np.float32)
                new_results = sess.append(
                    chunk,
                    float(msg.get("hys_high", HYS_HIGH_DEFAULT)),
                    float(msg.get("hys_low", HYS_LOW_DEFAULT)),
                    bool(msg.get("apply_minmax", False)),
                )
            except ValueError as e:
                await websocket.send_json({"type": "error", "detail": str(e)})
                continue

            await websocket.send_json(
                {
                    "type": "result",
                    "buffered_samples": sess.total_samples,
                    "new_results": [r.model_dump() for r in new_results],
                }
            )
    except WebSocketDisconnect:
        return


@app.websocket("/ws/live-muse")
async def ws_live_muse(websocket: WebSocket):
    """
    서버가 Muse LSL에서 직접 EEG를 읽어 추론/점수화한 결과를 브라우저로 푸시.
    """
    await websocket.accept()
    reader = create_reader("muse2")
    sess = StreamSession(uuid.uuid4().hex)
    scorer = DrowsinessScorer(
        window_size=60,
        drowsy_threshold=0.70,
        accumulated_time_limit=25.0,
        instant_alert_threshold=0.95,
        prob_is_awake=True,
    )

    try:
        if not reader.connect():
            await websocket.send_json({"type": "error", "detail": "Muse 연결 실패"})
            return

        await websocket.send_json({"type": "status", "message": "Muse 연결 성공"})

        while True:
            # blocking I/O는 event loop를 막지 않게 별도 스레드로 실행
            chunk = await asyncio.to_thread(reader.read_chunk, 2.0)
            if chunk is None:
                await websocket.send_json({"type": "status", "message": "데이터 대기 중..."})
                continue

            raw = np.stack([chunk.af7, chunk.af8], axis=1).astype(np.float32)
            new_results = sess.append(raw, HYS_HIGH_DEFAULT, HYS_LOW_DEFAULT, False)
            if not new_results:
                continue

            latest = new_results[-1]
            score_obj = scorer.score(
                prob_raw=float(latest.prob_raw),
                prob_scaled=(float(latest.prob_scaled) if latest.prob_scaled is not None else None),
                state=int(latest.state),
            )

            await websocket.send_json(
                {
                    "type": "tick",
                    "sequence_id": chunk.sequence_id,
                    "prob_raw": float(latest.prob_raw),
                    "prob_scaled": (float(latest.prob_scaled) if latest.prob_scaled is not None else None),
                    "state": int(latest.state),
                    "instant_score": float(score_obj.instant_score),
                    "smoothed_score": float(score_obj.smoothed_score),
                    "risk_level": score_obj.risk_level,
                    "should_alert": bool(score_obj.should_alert),
                }
            )

    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "detail": str(e)})
    finally:
        reader.disconnect()


# ============================================================
# 7. 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "muse_inference_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
