"""
core/config.py
==============
시스템 전체의 불변 설정값을 dataclass로 관리.
하드코딩된 매직 넘버를 완전히 제거하여 재현성과 실험 용이성을 보장한다.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Camera / Vision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    stream_index: int = 1       # IR 채널 인덱스 (1 = 좌적외선)
    jpeg_quality: int = 85      # 스트리밍 JPEG 인코딩 품질


@dataclass(frozen=True)
class DrowsinessConfig:
    # ── 캘리브레이션 ──────────────────────────────────────────────────────────
    calibration_duration: float = 5.0   # 초
    ear_threshold_factor: float = 0.80  # mean * factor (하한 보조 기준)
    ear_threshold_std_k: float = 1.5    # mean - k*σ  (주 통계 기준)
    ear_threshold_min: float = 0.15     # 절대 하한 (비현실적 낮은 값 방지)
    ear_threshold_max: float = 0.35     # 절대 상한 (비현실적 높은 값 방지)

    # ── 감지 임계값 ───────────────────────────────────────────────────────────
    mar_threshold: float = 0.60
    pitch_threshold: float = 25.0       # 고개를 숙인 상태로 판정하는 각도 기준
    pitch_danger_threshold: float = 50.0 # 고개를 완전히 떨군 것으로 간주하는 위험 각도 기준 (둔감화)
    head_drop_frames: int = 12
    perclos_danger_pct: float = 70.0    # PERCLOS 위험 판정 기준(%)

    # ── 시간 창 ───────────────────────────────────────────────────────────────
    perclos_window_sec: float = 60.0

    # ── 이동 평균 버퍼 크기 ───────────────────────────────────────────────────
    ear_buffer_size: int = 5
    mar_buffer_size: int = 5
    pitch_buffer_size: int = 10

    # ── EMA 가중치 (클수록 최신값 민감) ──────────────────────────────────────
    pitch_alpha: float = 0.7
    mar_alpha: float = 0.6


# ---------------------------------------------------------------------------
# EEG (Muse2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EEGConfig:
    """
    Muse2 EEG 수신 및 특징 추출 설정.

    Muse2 채널 배치:
      ch0 = TP9  (좌측두엽)
      ch1 = AF7  (좌전두엽) ← 인지 상태에 가장 민감
      ch2 = AF8  (우전두엽) ← 인지 상태에 가장 민감
      ch3 = TP10 (우측두엽)

    주파수 대역 (Hz):
      Delta  0.5–4   | Theta  4–8  | Alpha  8–12
      Beta  13–30    | Gamma  30–50
    """
    sample_rate: int = 256          # Muse2 EEG 샘플링 주파수 (Hz)
    lsl_stream_name: str = "Muse"   # Lab Streaming Layer 스트림 이름
    lsl_stream_type: str = "EEG"

    # 대역 경계값 (Hz)
    delta_band: tuple = (0.5, 4.0)
    theta_band: tuple = (4.0, 8.0)
    alpha_band: tuple = (8.0, 12.0)
    beta_band: tuple = (13.0, 30.0)
    gamma_band: tuple = (30.0, 50.0)

    # PSD 추정 파라미터 (Welch 방법)
    nperseg: int = 256              # 세그먼트 길이 (samples)
    noverlap: int = 128             # 오버랩 (samples)

    # 프론탈 채널 인덱스 (AF7=1, AF8=2)
    frontal_channels: tuple = (1, 2)

    # 특징 업데이트 주기 (초)
    feature_update_interval: float = 1.0

    # 연결 타임아웃 (초)
    connection_timeout: float = 10.0

    # 기준선 보정용 창 크기 (초)
    baseline_window_sec: float = 30.0


# ---------------------------------------------------------------------------
# Fusion (다중모달 졸음 융합)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusionConfig:
    """
    카메라 점수와 EEG 점수를 합산하는 가중치 설정.

    EEG 연결 시   : final = camera_weight * cam + eeg_weight * eeg
    EEG 미연결 시 : final = cam  (가중치 무시, 단일 모달 동작)
    """
    # 최종 점수 가중치 (합계 = 1.0)
    camera_weight: float = 0.55
    eeg_weight: float = 0.45

    # 카메라 서브-점수 내부 가중치 (합계 = 1.0)
    # Pitch가 너무 민감하게 반응하여 False Alarm을 유발할 수 있으므로,
    # PERCLOS와 EAR(눈의 개폐 상태)에 높은 가중치를 부여하고 Head Pitch 비중을 하향 조정합니다.
    # 사용자가 이 가중치들을 쉽게 조절하여 민감도를 튜닝할 수 있습니다.
    ear_weight: float = 0.30       # (기존 0.35 -> 0.40) 눈 감김은 가장 직관적인 지표
    perclos_weight: float = 0.40   # (기존 0.35 -> 0.40) 장기적인 눈 감김 유지 비율
    mar_weight: float = 0.15       # (기존 유지) 하품 빈도
    pitch_weight: float = 0.15     # (기존 0.15 -> 0.05) 머리 숙임의 가중치를 대폭 낮춰 민감도 완화

    # EEG 서브-점수 내부 가중치 (합계 = 1.0)
    alpha_beta_weight: float = 0.50
    rel_theta_weight: float = 0.40
    blink_rate_weight: float = 0.10

    # 졸음 판정 임계값 (최종 점수 기준, 0–1)
    threshold_caution: float = 0.25
    threshold_warning: float = 0.45
    threshold_drowsy: float = 0.65

    # 카메라 서브-점수 정규화 기준값 (fusion 엔진에서 사용)
    pitch_threshold: float = 25.0
    pitch_danger_threshold: float = 50.0

    # EEG 정상 Alpha/Beta 비율 기준선 (개인 캘리브레이션 전 초기값)
    eeg_alpha_beta_baseline: float = 1.5
    eeg_alpha_beta_danger: float = 4.0  # 이 값 이상이면 score = 1.0

    # 정상 하품 빈도 (회/분)
    normal_blink_rate: float = 15.0
    danger_blink_rate: float = 40.0
