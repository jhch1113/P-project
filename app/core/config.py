"""
core/config.py
==============
실시간 졸음 파이프라인의 하이퍼파라미터 단일 진입점.

- FusionConfig     : 융합 가중치, 단계 threshold, 모델 awake baseline
- DrowsinessConfig : 카메라 EAR/MAR/Pitch/PERCLOS
- EEGConfig        : Muse DSP·필터·특징 주기
- CameraConfig     : 캡처 해상도/FPS

환경변수·필드 설명 표: docs/HYPERPARAMETERS.md
점수 산출(정규화·합산) 로직: app/api/fusion.py
"""

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Camera / Vision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraConfig:
    width: int = int(os.environ.get("CAM_WIDTH", "640"))
    height: int = int(os.environ.get("CAM_HEIGHT", "480"))
    # 캡처 FPS. CPU 바운드 비전 파이프라인(MediaPipe)이 30fps를 따라가지
    # 못하면 RealSense 큐에 프레임이 누적되어 지연이 계속 커진다. 따라서
    # 처리 능력에 맞춘 보수적 기본값(15)을 사용하고, 환경변수로 조정 가능하다.
    # (실시간 지연 제거의 핵심은 아래 vision_loop의 '최신 프레임 drain' 로직이다.)
    fps: int = int(os.environ.get("CAM_FPS", "15"))
    stream_index: int = 1       # IR 채널 인덱스 (1 = 좌적외선)
    jpeg_quality: int = int(os.environ.get("CAM_JPEG_QUALITY", "80"))  # 스트리밍 JPEG 인코딩 품질
    use_webcam: bool = os.environ.get("USE_WEBCAM", "0") == "1"


@dataclass(frozen=True)
class DrowsinessConfig:
    # ── 캘리브레이션 ──────────────────────────────────────────────────────────
    calibration_duration: float = 5.0   # 초
    # 눈 감김: smoothed_ear < threshold. factor는 '뜬 눈 EAR 대비 비율'이므로
    # 낮을수록(0.55~0.65) 실제 감은 눈만 잡고 PERCLOS 오검출이 줄어든다.
    # (구 0.80+max() 조합은 임계값이 뜬 눈보다 높아져 PERCLOS가 상시 포화됨)
    ear_threshold_factor: float = 0.62  # threshold ≈ mean_open × factor
    ear_threshold_std_k: float = 2.0    # mean - k×σ (통계 보조, min()으로 병합)
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

    # ── 전처리(필터) 파라미터 ────────────────────────────────────────────────
    # [Critical] 원시 Muse2 신호에는 DC 오프셋·저주파 드리프트(발한/전극 분극)와
    # 60Hz 전원 노이즈, EMG(근전도)가 섞여 있다. 필터 없이 PSD를 구하면 저주파
    # 누설이 Theta(4–8Hz) 대역을 부풀려 relative_theta가 비정상적으로 커진다.
    # 따라서 detrend → 대역통과 → 노치 순으로 반드시 전처리한다.
    bandpass_low: float = 1.0       # 고역 통과 차단주파수 (Hz) — DC/드리프트 제거
    bandpass_high: float = 40.0     # 저역 통과 차단주파수 (Hz) — 고주파 EMG 억제
    notch_freq: float = 60.0        # 전원 노이즈 노치 (대한민국 60Hz)
    notch_quality: float = 30.0     # 노치 Q 인자 (높을수록 좁은 대역 제거)
    filter_order: int = 4           # Butterworth 차수 (zero-phase filtfilt 사용)

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

    # 졸음 판정 임계값 (최종 점수 0–1). 환경변수로 덮어쓸 수 있다.
    # 권장: 깨어 있는 상태에서 final_score 분포의 P90~P95를 CAUTION으로 잡고,
    # 실제 졸음 유도 시험에서 P50~P70을 WARNING으로 검증한다.
    threshold_caution: float = float(
        os.environ.get("FUSION_THRESHOLD_CAUTION", "0.32")
    )
    threshold_warning: float = float(
        os.environ.get("FUSION_THRESHOLD_WARNING", "0.52")
    )
    threshold_drowsy: float = float(
        os.environ.get("FUSION_THRESHOLD_DROWSY", "0.72")
    )

    # AI 모델 P(졸음)의 '깨어 있음' 기준선. 자동 캘리브레이션 전 폴백(실측 ~0.50–0.55).
    # effective = (prob - baseline) / (1 - baseline) 로 개인 편향을 제거한다.
    model_awake_baseline_default: float = float(
        os.environ.get("EEG_MODEL_AWAKE_BASELINE", "0.50")
    )
    # 자동 기준선 수집 시간(초). 이 구간의 모델 출력 중앙값을 baseline으로 고정.
    model_baseline_calib_sec: float = float(
        os.environ.get("EEG_MODEL_BASELINE_CALIB_SEC", "30")
    )

    # 카메라 서브-점수 정규화 기준값 (fusion 엔진에서 사용)
    pitch_threshold: float = 25.0
    pitch_danger_threshold: float = 50.0

    # EEG 정상 Alpha/Beta 비율 기준선 (개인 캘리브레이션 전 초기값)
    eeg_alpha_beta_baseline: float = 1.5
    eeg_alpha_beta_danger: float = 4.0  # 이 값 이상이면 score = 1.0

    # 정상 하품 빈도 (회/분)
    normal_blink_rate: float = 15.0
    danger_blink_rate: float = 40.0


def adjust_model_drowsy_prob(raw: float, baseline: float) -> float:
    """
    AI 모델 P(졸음)에서 개인·세션별 '깨어 있음' 오프셋을 제거한다.

    effective = clip((raw - baseline) / (1 - baseline), 0, 1)
    baseline≈0.5이고 raw≈0.55이면 effective≈0.10 수준으로 내려간다.
    """
    raw_f = max(0.0, min(1.0, float(raw)))
    b = max(0.0, min(0.95, float(baseline)))
    if b <= 1e-6:
        return raw_f
    return max(0.0, min(1.0, (raw_f - b) / (1.0 - b + 1e-6)))
