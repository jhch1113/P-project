"""
api/eeg_processor.py
====================
Muse2 EEG 데이터 수신·가공 인터페이스.

설계 원칙:
  - BaseEEGProcessor : 추상 인터페이스 (의존성 역전 원칙)
  - StubEEGProcessor : 테스트용 시뮬레이터 (Muse2 없이 개발·검증 가능)
  - Muse2LSLProcessor: 실제 Muse2 연동 구현체 (pylsl 기반, 플러그인 준비)

Muse2 → pylsl 실장 순서:
  1. pip install muselsl pylsl
  2. muselsl stream  (터미널에서 Muse2 BT 스트리밍 시작)
  3. Muse2LSLProcessor를 선택하여 실행

채널 배치 (Muse2):
  ch0 = TP9  | ch1 = AF7 | ch2 = AF8 | ch3 = TP10
  프론탈(AF7, AF8)이 인지·졸음 상태에 가장 민감하다.

주파수 대역:
  Delta 0.5–4Hz | Theta 4–8Hz | Alpha 8–12Hz | Beta 13–30Hz
"""

import logging
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from app.core.config import EEGConfig, FusionConfig, adjust_model_drowsy_prob
from app.core.state import EEGMetrics, SharedEEGState

logger = logging.getLogger(__name__)

# NumPy 2.0에서 np.trapz가 np.trapezoid로 대체되었다.
# 두 버전 모두에서 동작하도록 호환 심볼을 확보한다.
try:
    from numpy import trapezoid as _trapezoid  # NumPy >= 2.0
except ImportError:  # pragma: no cover - NumPy < 2.0 경로
    from numpy import trapz as _trapezoid


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------
class BaseEEGProcessor(ABC):
    """
    EEG 프로세서 추상 인터페이스.
    실제 구현체(Muse2LSLProcessor)와 테스트 스텁(StubEEGProcessor) 모두
    이 인터페이스를 구현함으로써 상위 레이어(Fusion, Routes)가
    구현체에 의존하지 않는다.
    """

    @abstractmethod
    def start(self) -> threading.Thread:
        """백그라운드 스레드 시작."""

    @abstractmethod
    def stop(self) -> None:
        """백그라운드 스레드 종료 요청."""

    @abstractmethod
    def is_connected(self) -> bool:
        """장치 연결 상태 반환."""


# ---------------------------------------------------------------------------
# Signal Preprocessing (detrend → bandpass → notch)
# ---------------------------------------------------------------------------
def preprocess_eeg(raw_data: np.ndarray, cfg: EEGConfig) -> np.ndarray:
    """
    다채널 EEG 원시 신호를 PSD 추정 전 정제한다.

    처리 순서 (의도된 순서이며 변경 금지):
      1. 선형 detrend  : DC 오프셋 + 선형 드리프트(전극 분극·발한) 제거
      2. 대역통과(BP)  : [low, high]Hz Butterworth, zero-phase(filtfilt)
                         → 저주파 누설로 인한 Theta 과대평가 차단,
                           고주파 EMG 억제
      3. 노치(Notch)   : 60Hz 전원 노이즈 제거 (Beta 대역 오염 방지)

    Parameters
    ----------
    raw_data : shape (n_channels, n_samples), float, μV
    cfg      : EEGConfig

    Returns
    -------
    np.ndarray : 동일 shape의 정제된 신호 (float64)

    Notes
    -----
    filtfilt는 신호 길이가 필터 패딩보다 길어야 한다. 너무 짧으면
    원본을 그대로 반환하여 런타임 오류를 방지한다(졸음 판정은 카메라가 백업).
    """
    assert raw_data.ndim == 2, f"raw_data는 2D여야 함. 수신 형상: {raw_data.shape}"
    fs = cfg.sample_rate
    nyq = fs / 2.0

    # 대역 경계의 물리적 유효성 검증 (0 < low < high < Nyquist)
    assert 0.0 < cfg.bandpass_low < cfg.bandpass_high < nyq, (
        f"대역통과 경계 오류: low={cfg.bandpass_low}, high={cfg.bandpass_high}, "
        f"Nyquist={nyq}"
    )

    n_samples = raw_data.shape[1]
    # 4차 SOS filtfilt의 안정적 동작에 필요한 최소 길이(보수적). 미만이면 스킵.
    min_len = 3 * (2 * cfg.filter_order + 1)
    if n_samples < min_len:
        logger.debug(
            "샘플 수(%d)가 필터 최소 길이(%d) 미만 — 전처리 스킵.", n_samples, min_len
        )
        return raw_data.astype(np.float64, copy=True)

    # float64로 승격 (필터 수치 안정성)
    data = raw_data.astype(np.float64, copy=True)

    # NaN/Inf 방어: 비정상 표본을 0으로 치환하여 필터 발산 차단
    if not np.all(np.isfinite(data)):
        logger.warning("EEG 입력에 비유한(NaN/Inf) 값 발견 — 0으로 치환.")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) 선형 detrend (채널축=1)
    data = sp_signal.detrend(data, axis=1, type="linear")

    # 2) 대역통과 (zero-phase, SOS로 수치 안정성 확보)
    sos = sp_signal.butter(
        cfg.filter_order,
        [cfg.bandpass_low, cfg.bandpass_high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    data = sp_signal.sosfiltfilt(sos, data, axis=1)

    # 3) 60Hz 노치 (Nyquist 미만일 때만 적용)
    if 0.0 < cfg.notch_freq < nyq:
        b_notch, a_notch = sp_signal.iirnotch(cfg.notch_freq, cfg.notch_quality, fs=fs)
        data = sp_signal.filtfilt(b_notch, a_notch, data, axis=1)

    assert data.shape == raw_data.shape, (
        f"전처리 후 형상 불일치: {data.shape} != {raw_data.shape}"
    )
    return data


# ---------------------------------------------------------------------------
# Band Power Utilities
# ---------------------------------------------------------------------------
def _band_power(
    data: np.ndarray,
    sample_rate: int,
    band: tuple,
    nperseg: int,
    noverlap: int,
) -> float:
    """
    Welch's method로 특정 주파수 대역의 절대 파워(μV²/Hz) 계산.

    Parameters
    ----------
    data        : 1D EEG 시계열, shape (N,)
    sample_rate : 샘플링 주파수 (Hz)
    band        : (f_low, f_high) 대역 경계값
    nperseg     : Welch 세그먼트 길이
    noverlap    : 세그먼트 오버랩

    Returns
    -------
    float: 해당 대역 평균 파워 (μV²/Hz)
    """
    assert data.ndim == 1, f"1D EEG 데이터 필요. 수신 형상: {data.shape}"
    if len(data) < nperseg:
        return 0.0
    freqs, psd = sp_signal.welch(
        data,
        fs=sample_rate,
        nperseg=min(nperseg, len(data)),
        noverlap=min(noverlap, len(data) // 2),
    )
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return float(_trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0


def extract_eeg_features(
    raw_data: np.ndarray,  # shape (n_channels, n_samples)
    cfg: EEGConfig,
    frontal_channels: Optional[tuple] = None,
) -> dict:
    """
    다채널 EEG 원시 데이터에서 졸음 관련 특징 추출.

    Parameters
    ----------
    raw_data         : shape (n_channels, n_samples), μV
    cfg              : EEGConfig
    frontal_channels : 프론탈 채널 인덱스 (None이면 cfg.frontal_channels 사용)

    Returns
    -------
    dict: alpha_power, theta_power, beta_power, alpha_beta_ratio, relative_theta
    """
    assert raw_data.ndim == 2, f"raw_data 형상: {raw_data.shape}"
    ch_idx = frontal_channels if frontal_channels is not None else cfg.frontal_channels

    delta_list, theta_list, alpha_list, beta_list, gamma_list = [], [], [], [], []
    for ch in ch_idx:
        assert ch < raw_data.shape[0], (
            f"채널 인덱스 {ch}가 데이터 채널 수 {raw_data.shape[0]}를 초과합니다."
        )
        ch_data = raw_data[ch]
        delta_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.delta_band, cfg.nperseg, cfg.noverlap)
        )
        theta_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.theta_band, cfg.nperseg, cfg.noverlap)
        )
        alpha_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.alpha_band, cfg.nperseg, cfg.noverlap)
        )
        beta_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.beta_band, cfg.nperseg, cfg.noverlap)
        )
        gamma_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.gamma_band, cfg.nperseg, cfg.noverlap)
        )

    delta = float(np.mean(delta_list))
    theta = float(np.mean(theta_list))
    alpha = float(np.mean(alpha_list))
    beta = float(np.mean(beta_list))
    gamma = float(np.mean(gamma_list))

    # [Fix] relative_theta는 '전 대역 대비 상대 파워'(표준 정의)로 계산한다.
    # EEG는 1/f 스펙트럼이라 분모에서 Delta를 빼면 theta가 비정상적으로 커져
    # relative_theta가 상시 0.8+로 포화된다(허위 졸음 경보 유발).
    # Delta~Gamma 전 대역 합을 분모로 사용하여 생리적 범위(~0.1–0.3)로 정규화한다.
    total = delta + theta + alpha + beta + gamma

    alpha_beta_ratio = alpha / max(beta, 1e-8)
    relative_theta = theta / max(total, 1e-8)

    return {
        "alpha_power": alpha,
        "theta_power": theta,
        "beta_power": beta,
        "alpha_beta_ratio": alpha_beta_ratio,
        "relative_theta": relative_theta,
    }


# ---------------------------------------------------------------------------
# Stub EEG Processor (테스트용 시뮬레이터)
# ---------------------------------------------------------------------------
class StubEEGProcessor(BaseEEGProcessor):
    """
    Muse2 미연결 상태를 나타내는 더미 프로세서.

    모든 EEG 값을 0으로 고정하고 is_connected=False를 유지한다.
    Fusion 엔진은 is_connected=False를 보고 EEG 가중치를 완전히 무시하며,
    최종 판정은 카메라 점수만으로 산출된다.
    실제 Muse2 연결 시 Muse2LSLProcessor로 교체한다.
    """

    _UPDATE_INTERVAL = 5.0  # EEG 없으므로 업데이트 주기를 느리게 설정 (CPU 절약)

    _ZERO_METRICS = EEGMetrics(
        alpha_power=0.0,
        theta_power=0.0,
        beta_power=0.0,
        alpha_beta_ratio=0.0,
        relative_theta=0.0,
        blink_rate=0.0,
        is_connected=False,
        signal_quality=0.0,
        timestamp=0.0,
    )

    def __init__(self, shared_state: SharedEEGState, cfg: EEGConfig) -> None:
        self._state = shared_state
        self._cfg = cfg
        self._stop_event = threading.Event()
        # 초기 상태를 즉시 0으로 설정
        self._state.update(self._ZERO_METRICS)
        logger.warning(
            "StubEEGProcessor 활성 — Muse2 미연결 상태. "
            "EEG 값 전체 0 고정, 최종 판정은 카메라 전용으로 동작합니다."
        )

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run, daemon=True, name="EEGStubLoop")
        t.start()
        return t

    def stop(self) -> None:
        self._stop_event.set()

    def is_connected(self) -> bool:
        return False

    def _run(self) -> None:
        # 값 변경 없이 주기적으로 대기만 함 (스레드 생존 유지용)
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._UPDATE_INTERVAL)

    @staticmethod
    def _drowsiness_envelope(phase: float) -> float:
        """0–1 범위의 졸음 수준 엔벨로프 (phase: 0–1, 120초 주기)."""
        # 0–0.5: 정상 유지 → 0.5–0.75: 졸음 상승 → 0.75–1.0: 회복
        if phase < 0.5:
            return 0.05 + 0.1 * math.sin(phase * math.pi)
        elif phase < 0.75:
            return 0.15 + 0.85 * math.sin((phase - 0.5) / 0.25 * math.pi / 2)
        else:
            return 1.0 - (phase - 0.75) / 0.25 * 0.95


# ---------------------------------------------------------------------------
# Muse2 LSL Processor (실제 연동 — pylsl 필요)
# ---------------------------------------------------------------------------
class Muse2LSLProcessor(BaseEEGProcessor):
    """
    Lab Streaming Layer(LSL)를 통해 Muse2 EEG 스트림을 실시간 수신하고
    졸음 관련 EEG 특징을 추출한다.

    사전 준비:
      1. pip install muselsl pylsl scipy
      2. 터미널: muselsl stream   (Muse2 BT 연결 + LSL 스트리밍 시작)
      3. main.py에서 StubEEGProcessor → Muse2LSLProcessor로 교체

    스트림 구조:
      채널 수 : 4 (TP9, AF7, AF8, TP10)
      샘플률  : 256 Hz
      단위    : μV
    """

    # 버퍼는 AI 모델 윈도우(5초=1280샘플)에 filtfilt 경계 여유(1초)를 더해
    # 6초로 잡는다. DSP 밴드파워는 이 버퍼 전체를 사용(주파수 해상도 향상).
    _MODEL_WINDOW_SEC = 5.0
    _BUFFER_SEC = 6.0

    def __init__(self, shared_state: SharedEEGState, cfg: EEGConfig) -> None:
        self._state = shared_state
        self._cfg = cfg
        self._stop_event = threading.Event()
        self._connected = False

        # 원시 데이터 버퍼: (n_channels, n_samples)
        _buf_len = int(cfg.sample_rate * self._BUFFER_SEC)
        self._buffer = np.zeros((4, _buf_len), dtype=np.float32)
        self._buf_lock = threading.Lock()

        # 모델 추론에 필요한 최소 실표본 수(윈도우 길이). 누적 전엔 추론 스킵.
        self._model_window_len = int(cfg.sample_rate * self._MODEL_WINDOW_SEC)
        self._samples_seen = 0

        # AI 모델 모듈(pp_nrsc.muse_inference_api)은 지연 임포트한다.
        # 임포트/추론 실패 시 model_available=False로 두고 DSP 폴백을 보장한다.
        self._muse_api = None
        self._model_init_failed = False

        self._fusion_cfg = FusionConfig()
        self._model_baseline_samples: list[float] = []
        self._model_baseline_value: Optional[float] = None

        # 마지막 특징 업데이트 타임스탬프
        self._last_update: float = 0.0

        # LSL 자동 재-resolve 타이밍 (환경변수로 조정 가능)
        self._resolve_retry_sec = float(
            os.environ.get("EEG_LSL_RESOLVE_RETRY_SEC", "3.0")
        )
        self._stale_timeout_sec = float(
            os.environ.get("EEG_LSL_STALE_TIMEOUT_SEC", "5.0")
        )
        self._probe_timeout_sec = float(
            os.environ.get("EEG_LSL_PROBE_TIMEOUT_SEC", "4.0")
        )
        self._reconnect_count = 0

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run, daemon=True, name="Muse2LSLLoop")
        t.start()
        logger.info("Muse2LSLProcessor 스레드 시작. LSL 스트림 탐색 중...")
        return t

    def stop(self) -> None:
        self._stop_event.set()

    def is_connected(self) -> bool:
        return self._connected

    def _run(self) -> None:
        try:
            # pylsl 동적 임포트 (설치되지 않은 환경에서도 모듈 로드 가능)
            import pylsl  # type: ignore
        except ImportError:
            logger.error(
                "pylsl가 설치되어 있지 않습니다. "
                "pip install pylsl 후 재시도하세요. StubEEGProcessor를 사용합니다."
            )
            self._publish_disconnected()
            return

        # 감독 루프: muselsl 재시작 등으로 LSL 스트림이 교체되어도 앱이 stale
        # 상태에 빠지지 않도록, 스트림 단절/오류를 감지하면 자동으로 다시
        # resolve 하여 inlet을 재생성한다. stop() 호출 시에만 종료한다.
        while not self._stop_event.is_set():
            inlet = None
            try:
                inlet = self._resolve_and_open(pylsl)
                if inlet is None:
                    self._publish_disconnected()
                    self._stop_event.wait(timeout=self._resolve_retry_sec)
                    continue
                # 새 스트림 연결: 버퍼/누적표본 초기화로 과거 데이터와 단절
                self._reset_stream_buffer()
                self._reconnect_count += 1
                logger.info(
                    "LSL 스트림 세션 시작 (누적 재연결 %d회).",
                    self._reconnect_count,
                )
                self._consume_stream(inlet)
            except Exception:
                logger.exception("Muse2 LSL 세션 오류 — 스트림 재-resolve를 시도합니다.")
                self._publish_disconnected()
                self._stop_event.wait(timeout=self._resolve_retry_sec)
            finally:
                self._connected = False
                if inlet is not None:
                    try:
                        inlet.close_stream()
                    except Exception:
                        logger.debug("LSL inlet close 중 예외(무시).", exc_info=True)

        logger.info("Muse2LSLProcessor 감독 루프 종료.")

    def _resolve_and_open(self, pylsl):
        """
        LSL 스트림을 resolve 하고 inlet을 생성한다.

        muselsl 재시작 시 '이름만 같은 유령 스트림'이 남을 수 있으므로,
        연결 직후 _probe_stream_alive()로 실제 표본 유입을 확인한다.
        미발견·프로브 실패 시 None.
        """
        logger.info(
            "LSL 스트림 탐색 중: name=%s, type=%s",
            self._cfg.lsl_stream_name,
            self._cfg.lsl_stream_type,
        )
        # name 단일 속성 resolve 대신 전체 스캔 후 name+type 필터 (유령 스트림 완화)
        all_streams = pylsl.resolve_streams(wait_time=self._cfg.connection_timeout)
        streams = [
            s
            for s in all_streams
            if s.name() == self._cfg.lsl_stream_name
            and s.type() == self._cfg.lsl_stream_type
        ]
        if not streams:
            # 구형 muselsl/환경 호환: name-only 폴백
            streams = pylsl.resolve_byprop(
                "name",
                self._cfg.lsl_stream_name,
                timeout=2.0,
            )
        if not streams:
            logger.warning(
                "Muse2 LSL 스트림 미발견 — muselsl 스트리밍 상태 확인 필요. "
                "%.0f초 후 재탐색합니다.",
                self._resolve_retry_sec,
            )
            return None

        inlet = pylsl.StreamInlet(streams[0], max_buflen=30)
        n_channels = int(inlet.info().channel_count())
        assert n_channels >= 4, (
            f"Muse2는 4채널이 필요합니다. 수신 채널 수: {n_channels}"
        )

        if not self._probe_stream_alive(inlet):
            logger.warning(
                "LSL 스트림 resolve 됐으나 %.1f초 내 표본 없음(유령/stale) — inlet 폐기 후 재탐색.",
                self._probe_timeout_sec,
            )
            try:
                inlet.close_stream()
            except Exception:
                pass
            return None

        self._connected = True
        try:
            stream_uid = streams[0].uid()
        except Exception:
            stream_uid = "unknown"
        logger.info(
            "Muse2 LSL 스트림 연결·프로브 성공 (채널 %d, uid=%s).",
            n_channels,
            stream_uid,
        )
        return inlet

    def _probe_stream_alive(self, inlet) -> bool:
        """연결 직후 실제 EEG 청크가 도착하는지 확인 (muselsl 재시작 대응)."""
        deadline = time.time() + self._probe_timeout_sec
        while time.time() < deadline:
            samples, _ = inlet.pull_chunk(timeout=0.2, max_samples=32)
            if samples:
                return True
        return False

    def _reset_stream_buffer(self) -> None:
        """새 스트림 연결 시 버퍼/누적표본을 초기화한다(stale 윈도우 추론 방지)."""
        with self._buf_lock:
            self._buffer[:] = 0.0
        self._samples_seen = 0
        self._last_update = 0.0
        self._model_baseline_samples.clear()
        self._model_baseline_value = None

    def _resolve_model_baseline(self, model_prob: float, model_ok: bool) -> float:
        """자동 awake baseline 수집·확정 후 융합에 쓸 baseline 값을 반환한다."""
        cfg = self._fusion_cfg
        if model_ok and self._model_baseline_value is None:
            self._model_baseline_samples.append(model_prob)
            calib_n = max(
                1,
                int(
                    cfg.model_baseline_calib_sec
                    / max(self._cfg.feature_update_interval, 0.1)
                ),
            )
            if len(self._model_baseline_samples) >= calib_n:
                self._model_baseline_value = float(
                    np.median(self._model_baseline_samples)
                )
                logger.info(
                    "AI 모델 awake baseline 확정: %.3f (n=%d, %.0fs)",
                    self._model_baseline_value,
                    len(self._model_baseline_samples),
                    cfg.model_baseline_calib_sec,
                )
        if self._model_baseline_value is not None:
            return self._model_baseline_value
        return cfg.model_awake_baseline_default

    def _consume_stream(self, inlet) -> None:
        """
        inlet에서 표본을 수신하며 특징을 추출/발행한다.

        _STALE_TIMEOUT_SEC 동안 신규 표본이 없으면(스트림 사망 추정) 반환하여
        상위 감독 루프가 재-resolve 하도록 한다. inlet 오류는 상위에서 처리.
        """
        last_data_time = time.time()
        while not self._stop_event.is_set():
            # 논블로킹 샘플 수집 (최대 32샘플)
            samples, _ = inlet.pull_chunk(timeout=0.2, max_samples=32)
            now = time.time()

            if samples:
                arr = np.array(samples, dtype=np.float32).T  # (n_ch, n_samp)
                arr = arr[:4]  # TP9, AF7, AF8, TP10만 사용
                n_new = arr.shape[1]
                with self._buf_lock:
                    self._buffer = np.roll(self._buffer, -n_new, axis=1)
                    self._buffer[:, -n_new:] = arr
                # 실제 수신 표본 누적량 (모델 추론 가능 시점 판단용)
                self._samples_seen += n_new
                last_data_time = now
            elif now - last_data_time > self._stale_timeout_sec:
                logger.warning(
                    "LSL 스트림에서 %.1f초간 데이터 없음 — 스트림 사망 추정, 재-resolve.",
                    now - last_data_time,
                )
                return

            # 주기적으로 특징 추출
            if now - self._last_update >= self._cfg.feature_update_interval:
                self._extract_and_publish()
                self._last_update = now

    def _extract_and_publish(self) -> None:
        with self._buf_lock:
            raw = self._buffer.copy()

        # 신호 품질은 '원시' 신호 기준으로 평가한다(포화/접촉 불량은 필터 전에 판단).
        quality = self._estimate_signal_quality(raw)

        # 전처리(detrend→대역통과→노치) 후 특징/깜빡임을 추출하여
        # 저주파 누설로 인한 Theta 과대평가와 60Hz/EMG 오염을 제거한다.
        clean = preprocess_eeg(raw, self._cfg)

        features = extract_eeg_features(clean, self._cfg)

        # AI 모델(MUSE_activity_model) 추론: 원시 AF7/AF8 신호를 모델 자체
        # 전처리로 처리하여 P(졸음)을 산출한다. 실패 시 (0.0, False) 폴백.
        model_prob, model_ok = self._run_model_inference(raw)
        baseline = self._resolve_model_baseline(model_prob, model_ok) if model_ok else 0.0
        model_adj = (
            adjust_model_drowsy_prob(model_prob, baseline) if model_ok else 0.0
        )

        metrics = EEGMetrics(
            alpha_power=features["alpha_power"],
            theta_power=features["theta_power"],
            beta_power=features["beta_power"],
            alpha_beta_ratio=features["alpha_beta_ratio"],
            relative_theta=features["relative_theta"],
            blink_rate=self._estimate_blink_rate(clean),
            model_drowsy_prob=model_prob,
            model_drowsy_prob_adj=model_adj,
            model_awake_baseline=baseline if model_ok else 0.0,
            model_available=model_ok,
            is_connected=True,
            signal_quality=quality,
            timestamp=time.time(),
        )
        self._state.update(metrics)

    def _get_muse_api(self):
        """pp_nrsc.muse_inference_api 지연 임포트 (1회). 실패 시 None."""
        if self._muse_api is not None or self._model_init_failed:
            return self._muse_api
        try:
            from pp_nrsc import muse_inference_api as api  # type: ignore
            # 시작 시 main.py가 warmup하지만, 독립 동작도 보장하기 위해 확인.
            api.get_model()
            self._muse_api = api
            logger.info("AI 모델(MUSE_activity_model) 추론 모듈 로드 완료.")
        except Exception:
            self._model_init_failed = True
            logger.exception(
                "AI 모델 로드 실패 — DSP 특징 기반으로 폴백합니다(모델 미사용)."
            )
        return self._muse_api

    def _run_model_inference(self, raw: np.ndarray) -> tuple:
        """
        원시 버퍼(4ch)에서 AF7/AF8을 추출해 AI 모델로 P(졸음)을 추론한다.

        Returns
        -------
        (prob, available) : prob는 0–1 졸음 확률, available은 유효 추론 여부.

        주의: 모델은 자체 전처리(preprocess)를 수행하므로 '원시' 신호를 넘긴다
        (이중 필터링 방지). 버퍼에 실표본이 윈도우 길이만큼 차기 전엔 스킵한다.
        """
        if self._samples_seen < self._model_window_len:
            return 0.0, False

        api = self._get_muse_api()
        if api is None:
            return 0.0, False

        try:
            af7, af8 = self._cfg.frontal_channels  # (1, 2) = AF7, AF8
            x = np.stack([raw[af7], raw[af8]], axis=1)  # (N, 2)

            x_proc, _quality = api.preprocess(x, use_advanced=True)
            windows = api.make_windows(x_proc)          # (W, 1280, 2)
            if windows.shape[0] == 0:
                return 0.0, False

            preds = api.predict_windows(windows)        # (W,) sigmoid 확률
            if preds.size == 0:
                return 0.0, False

            # 가장 최신 윈도우의 확률을 실시간 졸음 확률로 사용
            prob = float(np.clip(preds[-1], 0.0, 1.0))
            return prob, True
        except Exception:
            logger.exception("AI 모델 추론 중 오류 — 이번 주기는 DSP 폴백.")
            return 0.0, False

    def _estimate_blink_rate(self, data: np.ndarray) -> float:
        """
        전두엽 채널(AF7=ch1, AF8=ch2)의 큰 진폭 피크를 눈 깜빡임으로 간주.
        단순 임계값 기반 추정 (고도화 필요 시 ICA 아티팩트 제거 적용 권장).
        """
        frontal = np.mean(data[list(self._cfg.frontal_channels)], axis=0)
        threshold = float(np.std(frontal) * 3.5)
        peaks, _ = sp_signal.find_peaks(np.abs(frontal), height=threshold, distance=50)
        duration_sec = data.shape[1] / self._cfg.sample_rate
        return float(len(peaks) / max(duration_sec / 60.0, 1e-6))

    @staticmethod
    def _estimate_signal_quality(data: np.ndarray) -> float:
        """
        신호 품질 추정: 포화(±500μV 초과) 비율과 분산을 기반으로 0–1 반환.
        낮을수록 신호 불량 (전극 접촉 불량, 움직임 아티팩트 등).
        """
        saturated = float(np.mean(np.abs(data) > 500.0))
        variance_ok = float(np.clip(np.mean(np.std(data, axis=1)) / 20.0, 0.0, 1.0))
        return float(np.clip((1.0 - saturated) * variance_ok, 0.0, 1.0))

    def _publish_disconnected(self) -> None:
        self._state.update(EEGMetrics(is_connected=False, signal_quality=0.0))
