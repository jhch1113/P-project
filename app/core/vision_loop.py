"""
core/vision_loop.py
===================
VisionProcessingLoop: RealSense IR 카메라 캡처 + DriverMonitorCV 구동.

백그라운드 daemon 스레드로 실행되며, stop() 호출 시 graceful 종료.
처리 결과는 SharedCameraState를 통해 API 레이어로 전달된다.
"""

import logging
import os
import threading
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from app.core.config import CameraConfig, DrowsinessConfig
from app.core.monitor import DriverMonitorCV
from app.core.state import SharedCameraState

logger = logging.getLogger(__name__)


class VisionProcessingLoop:
    """
    단일 책임: RealSense 파이프라인 생명주기 관리 + 프레임별 DriverMonitorCV 호출.
    비즈니스 로직(수치 계산)은 모두 DriverMonitorCV에 위임한다.
    """

    def __init__(
        self,
        shared_state: SharedCameraState,
        cam_cfg: CameraConfig,
        drw_cfg: DrowsinessConfig,
    ) -> None:
        self._state = shared_state
        self._cam_cfg = cam_cfg
        self._drw_cfg = drw_cfg
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run, daemon=True, name="VisionLoop")
        t.start()
        logger.info("VisionProcessingLoop 백그라운드 스레드 시작.")
        return t

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("VisionProcessingLoop 종료 요청 전송.")

    # ------------------------------------------------------------------
    # Private: RealSense 파이프라인 빌드
    # ------------------------------------------------------------------
    def _start_pipeline(self) -> Tuple[rs.pipeline, rs.pipeline_profile]:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.infrared,
            self._cam_cfg.stream_index,
            self._cam_cfg.width,
            self._cam_cfg.height,
            rs.format.y8,
            self._cam_cfg.fps,
        )
        profile = pipeline.start(config)

        # [Critical Fix] RealSense의 레이저 에미터는 단순 조명이 아니라 '도트 패턴(Structured Light)'을 방사합니다.
        # 이 도트 패턴이 얼굴에 맺히면 MediaPipe FaceMesh(CNN)의 특징 추출이 완전히 실패하여 "No Face" 현상이 발생합니다.
        # 따라서 2D 기반 IR 얼굴 인식을 위해서는 반드시 에미터를 비활성화(0.0)해야 합니다.
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
            logger.info("IR 레이저 에미터 비활성화 완료 (도트 패턴으로 인한 얼굴 인식 방해 차단).")
        else:
            logger.warning("이 장치는 emitter_enabled 옵션을 지원하지 않습니다.")

        logger.info(
            f"RealSense IR 파이프라인 시작: "
            f"{self._cam_cfg.width}×{self._cam_cfg.height} @ {self._cam_cfg.fps}fps "
            f"(channel={self._cam_cfg.stream_index})"
        )
        return pipeline, profile

    # ------------------------------------------------------------------
    # Private: 메인 루프
    # ------------------------------------------------------------------
    # 일시적 장치 오류 후 재연결 대기 시간 (지수 백오프 상한 포함)
    _RECONNECT_BACKOFF_SEC = 2.0
    _RECONNECT_BACKOFF_MAX_SEC = 10.0

    def _run(self) -> None:
        """
        장치 캡처를 감싸는 최상위 감독(supervisor) 루프.

        RealSense 프레임 타임아웃·USB 일시 단절 등으로 캡처 루프가 예외로
        빠져나오더라도 스레드를 종료하지 않고, 파이프라인을 정리한 뒤
        지수 백오프로 재연결을 시도한다. stop()이 호출된 경우에만 종료한다.
        이로써 단일 일시 오류가 카메라 노드를 영구 정지시키는 것을 차단한다.
        """
        detector = DriverMonitorCV(self._drw_cfg, self._cam_cfg)
        backoff = self._RECONNECT_BACKOFF_SEC

        while not self._stop_event.is_set():
            try:
                if self._cam_cfg.use_webcam:
                    self._run_webcam_session(detector)
                else:
                    self._run_realsense_session(detector)
                # 세션이 정상적으로 반환되면 stop() 요청에 의한 것이다.
                backoff = self._RECONNECT_BACKOFF_SEC
            except Exception:
                logger.exception(
                    "카메라 캡처 세션 오류 — 스레드 유지 후 재연결을 시도합니다."
                )
                # 카메라 비활성 상태를 상태 객체에 반영 (대시보드가 인지 가능)
                self._safe_mark_inactive()
                if self._stop_event.is_set():
                    break
                logger.warning("%.1f초 후 카메라 재연결 시도...", backoff)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, self._RECONNECT_BACKOFF_MAX_SEC)

        logger.info("VisionProcessingLoop 감독 루프 종료.")

    def _safe_mark_inactive(self) -> None:
        """상태 객체에 카메라 비활성 표시 (인터페이스가 있을 때만)."""
        mark = getattr(self._state, "mark_inactive", None)
        if callable(mark):
            try:
                mark()
            except Exception:
                logger.debug("mark_inactive 호출 실패 (무시).", exc_info=True)

    # 전체 메쉬(TESSELATION) 드로잉은 프레임당 ~2600개 선분을 그려 CPU 비용이
    # 크다. 기본값은 가벼운 윤곽선(CONTOURS) 드로잉으로 지연을 줄이고,
    # DRAW_FULL_MESH=1 환경변수로 전체 메쉬 시각화를 활성화할 수 있다.
    _DRAW_FULL_MESH = os.environ.get("DRAW_FULL_MESH", "0") == "1"

    @classmethod
    def _build_drawing_context(cls):
        """매 프레임 재생성을 피하기 위한 드로잉 유틸리티 묶음."""
        styles = mp.solutions.drawing_styles
        return {
            "mp_drawing": mp.solutions.drawing_utils,
            "mp_fm": mp.solutions.face_mesh,
            "full_mesh": cls._DRAW_FULL_MESH,
            "tesselation_style": (
                styles.get_default_face_mesh_tesselation_style()
            ),
            "contour_style": (
                styles.get_default_face_mesh_contours_style()
            ),
        }

    def _process_and_publish(self, ir_rgb, detector, draw_ctx, encode_params) -> None:
        """단일 프레임 처리 → 랜드마크 드로잉 → JPEG 인코딩 → 상태 갱신."""
        snapshot, face_landmarks = detector.process_frame(ir_rgb)

        if face_landmarks:
            if draw_ctx["full_mesh"]:
                draw_ctx["mp_drawing"].draw_landmarks(
                    ir_rgb,
                    face_landmarks[0],
                    draw_ctx["mp_fm"].FACEMESH_TESSELATION,
                    None,
                    draw_ctx["tesselation_style"],
                )
            else:
                draw_ctx["mp_drawing"].draw_landmarks(
                    ir_rgb,
                    face_landmarks[0],
                    draw_ctx["mp_fm"].FACEMESH_CONTOURS,
                    None,
                    draw_ctx["contour_style"],
                )

        ir_bgr_for_display = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", ir_bgr_for_display, encode_params)
        if ok:
            self._state.update(buf.tobytes(), snapshot)

    def _run_webcam_session(self, detector: DriverMonitorCV) -> None:
        """일반 웹캠(cv2.VideoCapture) 캡처 세션."""
        logger.info("일반 웹캠(cv2.VideoCapture) 모드로 시작합니다.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("웹캠(cv2.VideoCapture(0))을 열 수 없습니다.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cam_cfg.fps)

        draw_ctx = self._build_drawing_context()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._cam_cfg.jpeg_quality]
        try:
            while not self._stop_event.is_set():
                if self._state.check_and_clear_reset_calibration():
                    detector.reset_calibration()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("웹캠 프레임 누락 — 건너뜀.")
                    cv2.waitKey(10)
                    continue

                frame = cv2.resize(frame, (self._cam_cfg.width, self._cam_cfg.height))
                ir_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._process_and_publish(ir_rgb, detector, draw_ctx, encode_params)
        finally:
            cap.release()
            logger.info("웹캠 세션 종료.")

    def _run_realsense_session(self, detector: DriverMonitorCV) -> None:
        """RealSense IR 캡처 세션. 프레임 타임아웃 시 예외로 상위에 위임한다."""
        pipeline, profile = self._start_pipeline()

        # 실제 카메라 intrinsics 추출 및 적용
        intrinsics = (
            profile
            .get_stream(rs.stream.infrared, self._cam_cfg.stream_index)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        detector.head_pose.update_from_realsense(intrinsics)

        # CLAHE 객체는 1회만 생성 (매 프레임 재생성은 불필요한 CPU 낭비)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        draw_ctx = self._build_drawing_context()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._cam_cfg.jpeg_quality]

        consecutive_empty = 0
        try:
            while not self._stop_event.is_set():
                if self._state.check_and_clear_reset_calibration():
                    detector.reset_calibration()

                # 프레임 타임아웃 시 RuntimeError가 발생하며, 이는 상위
                # 감독 루프가 재연결을 처리하도록 의도적으로 전파한다.
                frames = pipeline.wait_for_frames(timeout_ms=2000)

                # [Latency Fix] RealSense는 캡처 fps로 프레임을 큐에 쌓지만,
                # CPU 바운드 비전 처리는 그보다 느리다. wait_for_frames는 큐의
                # '가장 오래된' 프레임을 반환하므로, 처리가 캡처를 못 따라가면
                # 지연이 무한 누적된다. poll_for_frames로 큐에 남은 프레임을 모두
                # 비워 항상 '가장 최신' 프레임만 처리함으로써 실시간 지연을 제거한다.
                dropped = 0
                while True:
                    newer = pipeline.poll_for_frames()
                    if not newer:
                        break
                    frames = newer
                    dropped += 1
                if dropped:
                    logger.debug("지연 누적 방지: 오래된 프레임 %d개 폐기.", dropped)

                ir_frame = frames.get_infrared_frame(self._cam_cfg.stream_index)
                if not ir_frame:
                    consecutive_empty += 1
                    logger.warning("IR 프레임 누락 — 건너뜀. (연속 %d회)", consecutive_empty)
                    # 빈 프레임이 반복되면 파이프라인 재연결을 유도
                    if consecutive_empty >= 30:
                        raise RuntimeError("IR 프레임이 연속 누락되어 재연결합니다.")
                    continue
                consecutive_empty = 0

                img = np.asanyarray(ir_frame.get_data())
                assert img.shape == (self._cam_cfg.height, self._cam_cfg.width), (
                    f"예상치 못한 프레임 형상: {img.shape}"
                )

                # [Medical & CV Precision] IR 영상은 가시광 대비 동적 범위가 좁아 MediaPipe의 인식률을 급감시킵니다.
                # 조직(Tissue) 윤곽 검출 시 널리 쓰이는 CLAHE를 적용하여 국소 명암비를 극대화합니다.
                img_enhanced = clahe.apply(img)

                # GRAY(1ch) → RGB(3ch): MediaPipe는 RGB 3채널 입력 요구
                ir_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
                assert ir_rgb.dtype == np.uint8 and ir_rgb.shape[2] == 3, (
                    f"ir_rgb 타입/형상 오류: dtype={ir_rgb.dtype}, shape={ir_rgb.shape}"
                )

                self._process_and_publish(ir_rgb, detector, draw_ctx, encode_params)
        finally:
            try:
                pipeline.stop()
                logger.info("RealSense 파이프라인 정상 종료.")
            except Exception:
                logger.debug("파이프라인 정지 중 예외 (무시).", exc_info=True)
