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
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from core.config import EEGConfig
from core.state import EEGMetrics, SharedEEGState

logger = logging.getLogger(__name__)


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
    return float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0


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

    alpha_list, theta_list, beta_list = [], [], []
    for ch in ch_idx:
        assert ch < raw_data.shape[0], (
            f"채널 인덱스 {ch}가 데이터 채널 수 {raw_data.shape[0]}를 초과합니다."
        )
        ch_data = raw_data[ch]
        alpha_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.alpha_band, cfg.nperseg, cfg.noverlap)
        )
        theta_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.theta_band, cfg.nperseg, cfg.noverlap)
        )
        beta_list.append(
            _band_power(ch_data, cfg.sample_rate, cfg.beta_band, cfg.nperseg, cfg.noverlap)
        )

    alpha = float(np.mean(alpha_list))
    theta = float(np.mean(theta_list))
    beta = float(np.mean(beta_list))
    total = alpha + theta + beta

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

    _BUFFER_SEC = 2.0   # 특징 추출에 사용할 원시 데이터 창 크기 (초)

    def __init__(self, shared_state: SharedEEGState, cfg: EEGConfig) -> None:
        self._state = shared_state
        self._cfg = cfg
        self._stop_event = threading.Event()
        self._connected = False

        # 원시 데이터 버퍼: (n_channels, n_samples)
        _buf_len = int(cfg.sample_rate * self._BUFFER_SEC)
        self._buffer = np.zeros((4, _buf_len), dtype=np.float32)
        self._buf_lock = threading.Lock()

        # 마지막 특징 업데이트 타임스탬프
        self._last_update: float = 0.0

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

            logger.info(
                f"LSL 스트림 탐색 중: name={self._cfg.lsl_stream_name}, "
                f"type={self._cfg.lsl_stream_type}"
            )
            streams = pylsl.resolve_byprop(
                "name",
                self._cfg.lsl_stream_name,
                timeout=self._cfg.connection_timeout,
            )
            if not streams:
                logger.error(
                    "Muse2 LSL 스트림을 찾을 수 없습니다. "
                    "muselsl stream 명령으로 Muse2를 먼저 연결하세요."
                )
                self._publish_disconnected()
                return

            inlet = pylsl.StreamInlet(streams[0], max_buflen=30)
            self._connected = True
            logger.info("Muse2 LSL 스트림 연결 성공.")

            n_channels = int(inlet.info().channel_count())
            assert n_channels >= 4, (
                f"Muse2는 4채널이 필요합니다. 수신 채널 수: {n_channels}"
            )
            buf_len = self._buffer.shape[1]

            while not self._stop_event.is_set():
                # 논블로킹 샘플 수집 (최대 32샘플)
                samples, _ = inlet.pull_chunk(timeout=0.1, max_samples=32)
                if samples:
                    arr = np.array(samples, dtype=np.float32).T  # (n_ch, n_samp)
                    arr = arr[:4]  # TP9, AF7, AF8, TP10만 사용
                    n_new = arr.shape[1]
                    with self._buf_lock:
                        self._buffer = np.roll(self._buffer, -n_new, axis=1)
                        self._buffer[:, -n_new:] = arr

                # 주기적으로 특징 추출
                now = time.time()
                if now - self._last_update >= self._cfg.feature_update_interval:
                    self._extract_and_publish()
                    self._last_update = now

        except ImportError:
            logger.error(
                "pylsl가 설치되어 있지 않습니다. "
                "pip install pylsl 후 재시도하세요. StubEEGProcessor를 사용합니다."
            )
            self._publish_disconnected()
        except Exception:
            logger.exception("Muse2LSLProcessor 오류 발생.")
            self._publish_disconnected()
        finally:
            self._connected = False
            logger.info("Muse2LSLProcessor 종료.")

    def _extract_and_publish(self) -> None:
        with self._buf_lock:
            data = self._buffer.copy()

        features = extract_eeg_features(data, self._cfg)
        quality = self._estimate_signal_quality(data)

        metrics = EEGMetrics(
            alpha_power=features["alpha_power"],
            theta_power=features["theta_power"],
            beta_power=features["beta_power"],
            alpha_beta_ratio=features["alpha_beta_ratio"],
            relative_theta=features["relative_theta"],
            blink_rate=self._estimate_blink_rate(data),
            is_connected=True,
            signal_quality=quality,
            timestamp=time.time(),
        )
        self._state.update(metrics)

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
