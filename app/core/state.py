"""
core/state.py
=============
시스템 전반에서 공유되는 데이터 컨테이너.

  CameraMetrics   : 카메라 프레임 1장의 처리 결과 (불변 전달 객체)
  EEGMetrics      : EEG 특징 추출 결과 (불변 전달 객체)
  SharedCameraState : 비전 스레드 ↔ API 간 스레드 안전 공유 상태
  SharedEEGState    : EEG 스레드 ↔ API 간 스레드 안전 공유 상태
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Transfer Objects (불변 전달 객체)
# ---------------------------------------------------------------------------
@dataclass
class CameraMetrics:
    """카메라 기반 졸음 지표 스냅샷."""
    ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    pitch: float = 0.0
    status: str = "Waiting..."
    threshold: float = 0.25
    is_calibrated: bool = False

    def to_dict(self) -> dict:
        return {
            "ear": round(self.ear, 4),
            "mar": round(self.mar, 4),
            "perclos": round(self.perclos, 2),
            "pitch": round(self.pitch, 2),
            "status": self.status,
            "threshold": round(self.threshold, 4),
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class EEGMetrics:
    """
    Muse2 EEG 기반 졸음 지표 스냅샷.

    모든 power 값은 프론탈 채널(AF7, AF8) 평균 절대 파워 (μV²/Hz).
    relative_* 는 전체 파워 합 대비 해당 대역 비율 (0–1).
    """
    # 절대 파워 (μV²/Hz)
    alpha_power: float = 0.0
    theta_power: float = 0.0
    beta_power: float = 0.0

    # 파생 비율 (핵심 졸음 마커)
    alpha_beta_ratio: float = 1.0   # 높을수록 졸음 증가
    relative_theta: float = 0.0     # 전체 파워 대비 theta 비율

    # 눈 깜빡임 (EOG 아티팩트 기반 추정)
    blink_rate: float = 15.0        # 회/분 (정상 12–20)

    # 연결 상태
    is_connected: bool = False
    signal_quality: float = 0.0     # 0.0(불량) – 1.0(양호)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "alpha_power": round(self.alpha_power, 4),
            "theta_power": round(self.theta_power, 4),
            "beta_power": round(self.beta_power, 4),
            "alpha_beta_ratio": round(self.alpha_beta_ratio, 3),
            "relative_theta": round(self.relative_theta, 3),
            "blink_rate": round(self.blink_rate, 1),
            "is_connected": self.is_connected,
            "signal_quality": round(self.signal_quality, 2),
        }


# ---------------------------------------------------------------------------
# Shared State Containers (스레드 안전)
# ---------------------------------------------------------------------------
class SharedCameraState:
    """
    VisionProcessingLoop(프로듀서) ↔ FastAPI 라우터(컨슈머) 간 교환.

    threading.Event 블로킹 대기로 CPU 스핀 없이 새 프레임을 기다린다.
    락 보유 시간을 최소화하여 프레임 처리 지연을 방지한다.
    """

    # 마지막 프레임 수신 후 이 시간(초)이 지나면 카메라 비활성 판정
    ACTIVE_TIMEOUT_SEC: float = 3.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._metrics: CameraMetrics = CameraMetrics()
        self._new_frame_event = threading.Event()
        self._last_frame_time: float = 0.0   # 마지막 프레임 수신 시각 (time.time())
        self._reset_calib_event = threading.Event()  # 캘리브레이션 초기화 요청 이벤트

    def request_calibration_reset(self) -> None:
        """API 등에서 캘리브레이션 초기화를 요청할 때 호출합니다."""
        self._reset_calib_event.set()

    def check_and_clear_reset_calibration(self) -> bool:
        """비전 스레드에서 주기적으로 확인하여, 요청이 있으면 True 반환 후 플래그를 해제합니다."""
        if self._reset_calib_event.is_set():
            self._reset_calib_event.clear()
            return True
        return False

    def update(self, frame: bytes, metrics: CameraMetrics) -> None:
        """비전 스레드에서 호출 — 프레임과 메트릭을 원자적으로 갱신."""
        with self._lock:
            self._frame = frame
            self._metrics = metrics
            self._last_frame_time = time.time()
        self._new_frame_event.set()  # 락 해제 후 set → 지연 최소화

    def update_metrics(self, metrics: CameraMetrics) -> None:
        """프레임 없이 메트릭만 갱신 (원격 서비스 인입용)."""
        with self._lock:
            self._metrics = metrics
            self._last_frame_time = time.time()

    def get_frame(self, timeout: float = 2.0) -> Optional[bytes]:
        """
        새 프레임이 올 때까지 블로킹 대기.
        timeout 초 내 미도착 시 None 반환 (CPU 소비 0%).
        """
        arrived = self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        if not arrived:
            return None
        with self._lock:
            return self._frame

    def get_metrics(self) -> CameraMetrics:
        with self._lock:
            return self._metrics

    def is_active(self) -> bool:
        """최근 ACTIVE_TIMEOUT_SEC 이내에 프레임을 수신했는지 여부."""
        with self._lock:
            if self._last_frame_time == 0.0:
                return False
            return (time.time() - self._last_frame_time) < self.ACTIVE_TIMEOUT_SEC

    def seconds_since_last_frame(self) -> float:
        """마지막 프레임 수신 후 경과 시간(초). 한 번도 수신 없으면 -1."""
        with self._lock:
            if self._last_frame_time == 0.0:
                return -1.0
            return time.time() - self._last_frame_time


class SharedEEGState:
    """
    EEGProcessor(프로듀서) ↔ FastAPI 라우터/Fusion(컨슈머) 간 교환.
    카메라 상태와 동일한 패턴을 사용하여 일관성을 유지한다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: EEGMetrics = EEGMetrics()

    def update(self, metrics: EEGMetrics) -> None:
        with self._lock:
            self._metrics = metrics

    def get_metrics(self) -> EEGMetrics:
        with self._lock:
            return self._metrics
