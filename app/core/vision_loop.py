"""
core/vision_loop.py
===================
VisionProcessingLoop: RealSense IR 카메라 캡처 + DriverMonitorCV 구동.

백그라운드 daemon 스레드로 실행되며, stop() 호출 시 graceful 종료.
처리 결과는 SharedCameraState를 통해 API 레이어로 전달된다.
"""

import logging
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
    def _run(self) -> None:
        pipeline: Optional[rs.pipeline] = None
        cap: Optional[cv2.VideoCapture] = None
        try:
            detector = DriverMonitorCV(self._drw_cfg, self._cam_cfg)

            if self._cam_cfg.use_webcam:
                logger.info("일반 웹캠(cv2.VideoCapture) 모드로 시작합니다.")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cam_cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_cfg.height)
                cap.set(cv2.CAP_PROP_FPS, self._cam_cfg.fps)
            else:
                pipeline, profile = self._start_pipeline()
                
                # 실제 카메라 intrinsics 추출 및 적용
                intrinsics = (
                    profile
                    .get_stream(rs.stream.infrared, self._cam_cfg.stream_index)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
                detector.head_pose.update_from_realsense(intrinsics)

            # 드로잉 유틸리티 (매 루프 재생성 방지)
            mp_drawing = mp.solutions.drawing_utils
            mp_fm = mp.solutions.face_mesh
            tesselation_style = (
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._cam_cfg.jpeg_quality]

            while not self._stop_event.is_set():
                if self._state.check_and_clear_reset_calibration():
                    detector.reset_calibration()

                if self._cam_cfg.use_webcam:
                    assert cap is not None
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("웹캠 프레임 누락 — 건너뜀.")
                        cv2.waitKey(10)
                        continue
                    
                    # 웹캠은 가시광(RGB/BGR)이므로 바로 크기 맞추고 BGR -> RGB 변환
                    frame = cv2.resize(frame, (self._cam_cfg.width, self._cam_cfg.height))
                    ir_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    ir_frame = frames.get_infrared_frame(self._cam_cfg.stream_index)
                    if not ir_frame:
                        logger.warning("IR 프레임 누락 — 건너뜀.")
                        continue

                    img = np.asanyarray(ir_frame.get_data())
                    assert img.shape == (self._cam_cfg.height, self._cam_cfg.width), (
                        f"예상치 못한 프레임 형상: {img.shape}"
                    )

                    # [Medical & CV Precision] IR 영상은 가시광 대비 동적 범위가 좁아 MediaPipe의 인식률을 급감시킵니다.
                    # 조직(Tissue) 윤곽 검출 시 널리 쓰이는 CLAHE를 적용하여 국소 명암비를 극대화합니다.
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_enhanced = clahe.apply(img)

                    # GRAY(1ch) → RGB(3ch): MediaPipe는 RGB 3채널 입력 요구
                    ir_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
                    assert ir_rgb.dtype == np.uint8 and ir_rgb.shape[2] == 3, (
                        f"ir_rgb 타입/형상 오류: dtype={ir_rgb.dtype}, shape={ir_rgb.shape}"
                    )

                snapshot, face_landmarks = detector.process_frame(ir_rgb)

                if face_landmarks:
                    mp_drawing.draw_landmarks(
                        ir_rgb,
                        face_landmarks[0],
                        mp_fm.FACEMESH_TESSELATION,
                        None,
                        tesselation_style,
                    )

                # MediaPipe 출력을 위해 RGB 사용 (흑백이라 배경은 동일하나, Landmark 색상이 올바르게 표시되려면 BGR 인코딩 필수)
                ir_bgr_for_display = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2BGR)
                ret, buf = cv2.imencode(".jpg", ir_bgr_for_display, encode_params)
                if ret:
                    self._state.update(buf.tobytes(), snapshot)

        except Exception:
            logger.exception("VisionProcessingLoop 치명적 오류 발생.")
        finally:
            if pipeline is not None:
                pipeline.stop()
                logger.info("RealSense 파이프라인 정상 종료.")
            if cap is not None:
                cap.release()
                logger.info("웹캠 정상 종료.")
