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

from core.config import DrowsinessConfig

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
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        # RQDecomp3x3 반환값은 정규화 값 → ×360으로 도(°) 변환
        return angles[0] * 360.0, angles[1] * 360.0, angles[2] * 360.0


# ---------------------------------------------------------------------------
# Calibration Manager
# ---------------------------------------------------------------------------
class CalibrationManager:
    """
    개인화 EAR 임계값 자동 캘리브레이션 (통계 기반).

    두 가지 기준을 동시에 적용하여 더 보수적인(높은) 값을 선택:
      1) mean - k * σ      : 개인별 눈 개방 편차를 반영한 주 기준
      2) mean * factor     : 통계값이 너무 낮을 때의 안전망
    결과를 [min, max]로 클리핑하여 비현실적 임계값 방지.
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
        
        # 두 기준 중 더 높은(보수적) 값 선택 → 과소검출 방지
        threshold_stat = mean_ear - self._cfg.ear_threshold_std_k * std_ear
        threshold_factor = mean_ear * self._cfg.ear_threshold_factor
        raw_ear = max(threshold_stat, threshold_factor)
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
            f"캘리브레이션 완료: EAR mean={mean_ear:.4f}, thr={self.ear_threshold:.4f} | "
            f"Pitch baseline={self.pitch_baseline:.2f}° (n={len(self._ear_samples)})"
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
