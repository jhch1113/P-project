"""
core/monitor.py
===============
DriverMonitorCV: 단일 RGB 프레임 입력 → CameraMetrics 출력.

하위 계산기(EAR/MAR/HeadPose/Calibration/PERCLOS)를 조합하여
프레임 단위 졸음 지표를 산출한다. 카메라/스트리밍 로직은 관여하지 않는다.
"""

import logging
import time
from collections import deque
from typing import Optional, Tuple

import mediapipe as mp
import numpy as np

from core.calculators import (
    CalibrationManager,
    EARCalculator,
    HeadPoseEstimator,
    LEFT_EYE_IDX,
    MARCalculator,
    PERCLOSCalculator,
    RIGHT_EYE_IDX,
)
from core.config import CameraConfig, DrowsinessConfig
from core.state import CameraMetrics

logger = logging.getLogger(__name__)


class DriverMonitorCV:
    """
    프레임 처리 파이프라인:
      RGB 이미지 → FaceMesh → EAR/MAR/Pitch 계산 → 캘리브레이션/감지 → CameraMetrics

    상태 보유: 이동 평균 버퍼, EMA 값, 카운터.
    실제 수치 계산은 전용 계산기에 위임 (단일 책임 원칙).
    """

    def __init__(self, drw_cfg: DrowsinessConfig, cam_cfg: CameraConfig) -> None:
        self._cfg = drw_cfg

        _mp_fm = mp.solutions.face_mesh
        self._face_mesh = _mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            # IR 이미지는 가시광 대비 피부 텍스처·명암이 달라 MediaPipe 신뢰도가
            # 구조적으로 낮게 산출된다. 0.5 → 0.3으로 낮춰 IR 환경에서 검출 안정성 확보.
            # tracking confidence는 일단 검출 후 연속 추적이므로 더 낮게 설정해도 무방.
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        logger.info("MediaPipe FaceMesh 초기화 완료 (IR 모드: confidence=0.3).")

        self._ear_calc = EARCalculator()
        self._mar_calc = MARCalculator()
        self.head_pose = HeadPoseEstimator(cam_cfg.width, cam_cfg.height)
        self._calibration = CalibrationManager(drw_cfg)
        self._perclos = PERCLOSCalculator(drw_cfg.perclos_window_sec)

        # 이동 평균 버퍼
        self._ear_buf: deque = deque(maxlen=drw_cfg.ear_buffer_size)
        self._mar_buf: deque = deque(maxlen=drw_cfg.mar_buffer_size)
        self._pitch_buf: deque = deque(maxlen=drw_cfg.pitch_buffer_size)

        # EMA 상태
        self._pitch_ema: Optional[float] = None
        self._mar_ema: Optional[float] = None

        # 머리 숙임 연속 프레임 카운터
        self._head_drop_counter: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ema(new_val: float, prev: Optional[float], alpha: float) -> float:
        return new_val if prev is None else alpha * new_val + (1.0 - alpha) * prev

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset_calibration(self) -> None:
        """현재까지의 캘리브레이션 데이터를 버리고 다시 시작하도록 요청합니다."""
        self._calibration.reset()

    def process_frame(
        self, image_rgb: np.ndarray
    ) -> Tuple[CameraMetrics, Optional[object]]:
        """
        Parameters
        ----------
        image_rgb : np.ndarray, shape (H, W, 3), dtype uint8

        Returns
        -------
        (CameraMetrics, face_landmarks | None)
        """
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, (
            f"RGB 3채널 이미지가 필요합니다. 수신 형상: {image_rgb.shape}"
        )
        assert image_rgb.dtype == np.uint8, (
            f"이미지 dtype은 uint8이어야 합니다. 수신: {image_rgb.dtype}"
        )

        img_h, img_w = image_rgb.shape[:2]

        # MediaPipe 공식 권장 패턴: process() 전 writeable=False → 내부 복사 생략
        # writeable=True 복원은 이후 draw_landmarks()가 직접 배열에 쓸 수 있도록 하기 위함
        image_rgb.flags.writeable = False
        results = self._face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        snapshot = CameraMetrics(
            threshold=self._calibration.ear_threshold,
            is_calibrated=self._calibration.is_done,
        )

        if not results.multi_face_landmarks:
            logger.debug(
                "MediaPipe 얼굴 미검출 — "
                f"shape={image_rgb.shape}, dtype={image_rgb.dtype}, "
                f"min={int(image_rgb.min())}, max={int(image_rgb.max())}, "
                f"mean={float(image_rgb.mean()):.1f}"
            )
            snapshot.status = "No Face"
            return snapshot, None

        landmarks = results.multi_face_landmarks[0].landmark
        current_time = time.time()

        # ── EAR ──────────────────────────────────────────────────────
        left_ear = self._ear_calc.compute(landmarks, LEFT_EYE_IDX)
        right_ear = self._ear_calc.compute(landmarks, RIGHT_EYE_IDX)
        self._ear_buf.append((left_ear + right_ear) / 2.0)
        smoothed_ear = float(np.mean(self._ear_buf))
        snapshot.ear = smoothed_ear

        # ── MAR ──────────────────────────────────────────────────────
        self._mar_buf.append(self._mar_calc.compute(landmarks))
        self._mar_ema = self._ema(
            float(np.mean(self._mar_buf)), self._mar_ema, self._cfg.mar_alpha
        )
        snapshot.mar = self._mar_ema

        # ── Head Pose ─────────────────────────────────────────────────
        pitch, _, _ = self.head_pose.estimate(landmarks, img_w, img_h)
        if pitch is not None:
            self._pitch_buf.append(pitch)
            self._pitch_ema = self._ema(
                float(np.mean(self._pitch_buf)), self._pitch_ema, self._cfg.pitch_alpha
            )
            snapshot.pitch = self._pitch_ema

        # ── Calibration 진행 중 ────────────────────────────────────────
        calib_status = self._calibration.update(smoothed_ear, self._pitch_ema, current_time)
        snapshot.threshold = self._calibration.ear_threshold
        if calib_status is not None:
            snapshot.status = calib_status
            return snapshot, results.multi_face_landmarks

        # ── 캘리브레이션 완료 후 감지 ──────────────────────────────────
        snapshot.is_calibrated = True
        is_eye_closed = smoothed_ear < self._calibration.ear_threshold
        snapshot.perclos = self._perclos.update(is_eye_closed, current_time)

        # 영점 조절된 Pitch(상대 각도) 사용
        # 고개를 숙일 때 양의 방향으로 값이 커진다고 가정. 
        # (만약 음수 방향이라면 절대값 abs() 처리가 필요할 수 있으나 solvePnP 기본 특성상 pitch 변위 사용)
        if self._pitch_ema is not None:
            adjusted_pitch = self._pitch_ema - self._calibration.pitch_baseline
            snapshot.pitch = adjusted_pitch  # 대시보드 및 Fusion에 전달되는 값은 보정된 상대 각도
            
            if adjusted_pitch > self._cfg.pitch_threshold:
                self._head_drop_counter += 1
            else:
                self._head_drop_counter = 0
        else:
            self._head_drop_counter = 0

        warnings: list[str] = []
        if is_eye_closed:
            warnings.append("EYES CLOSED")
        if self._mar_ema > self._cfg.mar_threshold:
            warnings.append("YAWNING")
        if self._head_drop_counter >= self._cfg.head_drop_frames:
            warnings.append("HEAD DROP")
        if snapshot.perclos >= self._cfg.perclos_danger_pct:
            warnings.append(f"PERCLOS≥{self._cfg.perclos_danger_pct:.0f}%")

        snapshot.status = ", ".join(warnings) if warnings else "NORMAL"
        return snapshot, results.multi_face_landmarks
