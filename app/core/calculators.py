"""
core/calculators.py
===================
단일 책임 원칙에 따라 분리된 수치 계산기 모음.
각 클래스는 하나의 측정값 계산에만 집중한다.

  EARCalculator       : Eye Aspect Ratio (눈 개방비)
  MARCalculator       : Mouth Aspect Ratio (입 개방비)
  HeadPoseEstimator   : solvePnP 기반 머리 자세 추정
  CalibrationManager  : 개인화 EAR 임계값 통계적 캘리브레이션
  PERCLOSCalculator   : Sliding-window 기반 PERCLOS
"""

import logging
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from app.core.config import DrowsinessConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 랜드마크 인덱스 상수 (MediaPipe FaceMesh 478점 기준)
# ---------------------------------------------------------------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 머리 자세 추정용 3D 얼굴 모델 좌표 (일반 인체 모델, 단위: mm)
FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),           # 코 끝       → landmark[1]
        (0.0, -330.0, -65.0),      # 턱          → landmark[152]
        (-225.0, 170.0, -135.0),   # 왼쪽 눈 외각 → landmark[263]
        (225.0, 170.0, -135.0),    # 오른쪽 눈 외각→ landmark[33]
        (-150.0, -150.0, -125.0),  # 왼쪽 입 꼬리 → landmark[291]
        (150.0, -150.0, -125.0),   # 오른쪽 입 꼬리→ landmark[61]
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# EAR Calculator
# ---------------------------------------------------------------------------
class EARCalculator:
    """
    Eye Aspect Ratio = (V1 + V2) / (2 * H)

    V1, V2: 수직 랜드마크 쌍 간 거리
    H     : 수평 랜드마크 쌍 간 거리
    """

    @staticmethod
    def compute(landmarks, indices: list) -> float:
        assert len(indices) == 6, (
            f"EAR 계산에는 정확히 6개 랜드마크 인덱스가 필요합니다. 수신={len(indices)}"
        )
        coords = np.array(
            [(landmarks[i].x, landmarks[i].y) for i in indices], dtype=np.float32
        )
        assert coords.shape == (6, 2), f"EAR coords 형상 불일치: {coords.shape}"
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# MAR Calculator
# ---------------------------------------------------------------------------
class MARCalculator:
    """
    Mouth Aspect Ratio = mean(V1, V2, V3) / H

    세 쌍의 수직 거리 평균을 수평 거리로 나눈 값.
    하품 감지에 사용 (값이 클수록 입이 크게 열림).
    """

    @staticmethod
    def compute(landmarks) -> float:
        def _pt(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x, landmarks[idx].y], dtype=np.float32)

        horiz = np.linalg.norm(_pt(61) - _pt(291))
        if horiz < 1e-6:
            return 0.0
        v_pairs = [(13, 14), (81, 178), (311, 402)]
        vert = float(
            np.mean([np.linalg.norm(_pt(a) - _pt(b)) for a, b in v_pairs])
        )
        return vert / float(horiz)


# ---------------------------------------------------------------------------
# Head Pose Estimator
# ---------------------------------------------------------------------------
class HeadPoseEstimator:
    """
    OpenCV solvePnP + RQDecomp3x3으로 머리 자세(Pitch/Yaw/Roll) 추정.

    핵심 개선: RealSense SDK에서 실제 카메라 내부 파라미터(intrinsics)를
    수신하여 적용한다. focal_length ≈ image_width 근사값 대비
    Pitch 각도 정확도가 현저히 향상된다.
    """

    def __init__(self, width: int, height: int) -> None:
        f = float(width)
        cx, cy = width / 2.0, height / 2.0
        self._camera_matrix = np.array(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        logger.warning(
            "HeadPoseEstimator: 추정 내부 파라미터 사용 중. "
            "update_from_realsense() 호출 후 실제 intrinsics로 대체됩니다."
        )

    def update_from_realsense(self, intrinsics: rs.intrinsics) -> None:
        """RealSense SDK에서 수신한 실제 카메라 행렬로 갱신."""
        self._camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        coeffs = list(intrinsics.coeffs)[:4]
        self._dist_coeffs = np.array(coeffs, dtype=np.float64).reshape(4, 1)
        logger.info(
            f"HeadPoseEstimator intrinsics 갱신 완료: "
            f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
            f"ppx={intrinsics.ppx:.2f}, ppy={intrinsics.ppy:.2f}"
        )

    def estimate(
        self, landmarks, img_w: int, img_h: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        image_points = np.array(
            [
                (landmarks[1].x * img_w,   landmarks[1].y * img_h),
                (landmarks[152].x * img_w, landmarks[152].y * img_h),
                (landmarks[263].x * img_w, landmarks[263].y * img_h),
                (landmarks[33].x * img_w,  landmarks[33].y * img_h),
                (landmarks[291].x * img_w, landmarks[291].y * img_h),
                (landmarks[61].x * img_w,  landmarks[61].y * img_h),
            ],
            dtype=np.float64,
        )
        assert image_points.shape == (6, 2), (
            f"image_points 형상 오류: {image_points.shape}"
        )
        success, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None, None
        rmat, _ = cv2.Rodrigues(rvec)
        # cv2.RQDecomp3x3의 첫 반환값(angles)은 이미 '도(°)' 단위의 오일러 각이다.
        # 과거 ×360을 곱하던 로직은 25°를 9000°로 폭증시켜 거짓 HEAD DROP을
        # 유발했으므로 제거한다. (angles[0]=pitch, [1]=yaw, [2]=roll)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])


# ---------------------------------------------------------------------------
# Calibration Manager
# ---------------------------------------------------------------------------
class CalibrationManager:
    """
    개인화 EAR 임계값 자동 캘리브레이션 (통계 기반).

    판정: smoothed_ear < ear_threshold → 눈 감김(PERCLOS 누적).

    두 후보 중 **더 낮은** 값을 선택한다(뜬 눈을 감김으로 오분류하지 않도록):
      1) mean - k×σ   : 캘리브레이션 분산 반영
      2) mean × factor: 뜬 눈 EAR의 일정 비율(일반적으로 0.55~0.65)
    임계값이 mean×0.75를 넘지 않게 상한을 둔다(PERCLOS 포화 방지).
    결과는 [ear_threshold_min, ear_threshold_max]로 클리핑.
    """

    def __init__(self, cfg: DrowsinessConfig) -> None:
        self._cfg = cfg
        self._ear_samples: list = []
        self._pitch_samples: list = []
        self._start_time: Optional[float] = None
        self.is_done: bool = False
        self.ear_threshold: float = 0.25  # 캘리브레이션 완료 전 초기 안전값
        self.pitch_baseline: float = 0.0  # 정면 응시 시의 기본 Pitch 각도

    def reset(self) -> None:
        """캘리브레이션 상태를 초기화하여 재보정을 시작합니다."""
        self._ear_samples.clear()
        self._pitch_samples.clear()
        self._start_time = None
        self.is_done = False
        self.ear_threshold = 0.25
        self.pitch_baseline = 0.0
        logger.info("캘리브레이션 초기화됨: 다음 프레임부터 재수집 시작.")

    def update(self, ear: float, pitch: Optional[float], current_time: float) -> Optional[str]:
        """
        매 프레임 호출.
        - 진행 중: 카운트다운 상태 문자열 반환
        - 완료/이미 완료: None 반환
        """
        if self.is_done:
            return None
        if self._start_time is None:
            self._start_time = current_time
            logger.info("EAR 및 Pitch 캘리브레이션 시작.")
        elapsed = current_time - self._start_time
        
        self._ear_samples.append(ear)
        if pitch is not None:
            self._pitch_samples.append(pitch)
            
        remaining = max(0, int(self._cfg.calibration_duration - elapsed))
        if elapsed >= self._cfg.calibration_duration:
            self._finalize()
            return None
        return f"Calibrating... {remaining}s"

    def _finalize(self) -> None:
        # EAR 보정
        arr_ear = np.array(self._ear_samples, dtype=np.float32)
        mean_ear = float(np.mean(arr_ear))
        std_ear = float(np.std(arr_ear))
        
        threshold_stat = mean_ear - self._cfg.ear_threshold_std_k * std_ear
        threshold_factor = mean_ear * self._cfg.ear_threshold_factor
        # 낮은 임계값 → 뜬 눈(ear≈mean)은 감김으로 잡히지 않음 → PERCLOS 정상화
        raw_ear = min(threshold_stat, threshold_factor)
        # 안전 상한: 임계값이 뜬 눈 평균의 75%를 넘으면 오검출 위험이 크다
        raw_ear = min(raw_ear, mean_ear * 0.75)
        self.ear_threshold = float(
            np.clip(raw_ear, self._cfg.ear_threshold_min, self._cfg.ear_threshold_max)
        )

        # Pitch 보정 (평균 자세를 0도로 영점 조절하기 위한 기준값)
        if self._pitch_samples:
            arr_pitch = np.array(self._pitch_samples, dtype=np.float32)
            self.pitch_baseline = float(np.mean(arr_pitch))
        else:
            self.pitch_baseline = 0.0

        self.is_done = True
        logger.info(
            "캘리브레이션 완료: EAR mean=%.4f std=%.4f | thr=%.4f "
            "(stat=%.4f factor=%.4f cap=%.4f) | Pitch baseline=%.2f° (n=%d)",
            mean_ear,
            std_ear,
            self.ear_threshold,
            threshold_stat,
            threshold_factor,
            mean_ear * 0.75,
            self.pitch_baseline,
            len(self._ear_samples),
        )


# ---------------------------------------------------------------------------
# PERCLOS Calculator
# ---------------------------------------------------------------------------
class PERCLOSCalculator:
    """
    PERCLOS (Percentage of Eye Closure) — NHTSA 기준 졸음 지표.
    sliding-window 내 눈 감김 프레임 비율(%)을 실시간 계산.
    """

    def __init__(self, window_sec: float) -> None:
        self._window_sec = window_sec
        self._history: deque = deque()  # (timestamp, is_closed)

    def update(self, is_closed: bool, current_time: float) -> float:
        self._history.append((current_time, is_closed))
        cutoff = current_time - self._window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        if not self._history:
            return 0.0
        closed_cnt = sum(1 for _, c in self._history if c)
        return (closed_cnt / len(self._history)) * 100.0
