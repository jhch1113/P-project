"""
main.py
=======
서비스 모드에 따라 역할을 분리할 수 있는 애플리케이션 진입점.

SERVICE_MODE:
    - monolith     : 카메라 + EEG + 융합 + 대시보드 (기본)
    - orchestrator : 융합 + 대시보드만 실행, 다른 서비스에서 메트릭 인입
    - camera       : 카메라 서비스 (IR 영상 + 카메라 메트릭), 선택적으로 인입 전송
    - eeg          : EEG 서비스 (/eeg 포함), 선택적으로 인입 전송

EEG_MODE:
    - stub  : EEG 미연결 → 카메라 단독 판정
    - muse2 : 실제 Muse2 LSL 스트림 사용
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.fusion import DrowsinessFusion
from app.api.routes import create_router
from app.core.config import CameraConfig, DrowsinessConfig, EEGConfig, FusionConfig
from app.core.metrics_publisher import MetricsPublisher
from app.core.state import SharedCameraState, SharedEEGState

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


class _SuppressPollingPaths(logging.Filter):
    """
    대시보드가 50ms 주기로 호출하는 폴링 엔드포인트의 access 로그를 억제한다.
    실제 오류(4xx/5xx)는 이 필터를 통과하지 않으므로 놓치지 않는다.
    억제 대상: /metrics/all, /metrics/camera, /metrics/eeg, /metrics/fusion, /video_feed
    """
    _SUPPRESSED = frozenset([
        "/metrics/all",
        "/metrics/camera",
        "/metrics/eeg",
        "/metrics/fusion",
        "/video_feed",
    ])

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SUPPRESSED)


# uvicorn access 로거에 필터 적용 (basicConfig 이후에 설정해야 적용됨)
logging.getLogger("uvicorn.access").addFilter(_SuppressPollingPaths())

def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("잘못된 환경 변수 %s=%s, 기본값 %.2f 사용", name, raw, default)
        return default


def _normalize_service_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    allowed = {"monolith", "orchestrator", "camera", "eeg"}
    if mode not in allowed:
        logger.warning("알 수 없는 SERVICE_MODE=%s, monolith로 대체", value)
        return "monolith"
    return mode


def create_app() -> FastAPI:
    service_mode = _normalize_service_mode(os.getenv("SERVICE_MODE", "monolith"))
    eeg_mode = (os.getenv("EEG_MODE", "stub") or "stub").strip().lower()
    aggregator_url = os.getenv("AGGREGATOR_URL", "").rstrip("/")
    video_feed_url = os.getenv("VIDEO_FEED_URL", "/video_feed")
    publish_interval = _get_float_env("PUBLISH_INTERVAL_SEC", 0.5)

    logger.info("SERVICE_MODE=%s", service_mode)
    logger.info("EEG_MODE=%s", eeg_mode)

    cam_config = CameraConfig()
    drw_config = DrowsinessConfig()
    eeg_config = EEGConfig()
    fusion_config = FusionConfig()

    camera_state = SharedCameraState()
    eeg_state = SharedEEGState()

    vision_loop = None
    eeg_processor = None
    camera_publisher = None
    eeg_publisher = None
    muse_inference_api = None

    def _camera_payload() -> dict:
        data = camera_state.get_metrics().to_dict()
        data["timestamp"] = time.time()
        return data

    def _eeg_payload() -> dict:
        metrics = eeg_state.get_metrics()
        data = metrics.to_dict()
        data["timestamp"] = metrics.timestamp
        return data

    if service_mode in {"monolith", "camera"}:
        from app.core.vision_loop import VisionProcessingLoop

        vision_loop = VisionProcessingLoop(camera_state, cam_config, drw_config)

        if service_mode == "camera" and aggregator_url and publish_interval > 0:
            camera_publisher = MetricsPublisher(
                name="camera",
                url=f"{aggregator_url}/ingest/camera",
                interval_sec=publish_interval,
                payload_fn=_camera_payload,
            )

    if service_mode in {"monolith", "eeg"}:
        from app.api.eeg_processor import Muse2LSLProcessor, StubEEGProcessor
        from pp_nrsc import muse_inference_api as _muse_inference_api

        if eeg_mode == "muse2":
            eeg_processor = Muse2LSLProcessor(eeg_state, eeg_config)
            logger.info("EEG 모드: Muse2 LSL (실제 장치)")
        else:
            eeg_processor = StubEEGProcessor(eeg_state, eeg_config)
            logger.info("EEG 모드: Stub 시뮬레이터 (테스트용)")

        muse_inference_api = _muse_inference_api

        if service_mode == "eeg" and aggregator_url and publish_interval > 0:
            eeg_publisher = MetricsPublisher(
                name="eeg",
                url=f"{aggregator_url}/ingest/eeg",
                interval_sec=publish_interval,
                payload_fn=_eeg_payload,
            )

    fusion_engine = DrowsinessFusion(fusion_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("═══ 애플리케이션 시작 (mode=%s) ═══", service_mode)
        if muse_inference_api is not None:
            muse_inference_api.get_model()
        if vision_loop is not None:
            vision_loop.start()
        if eeg_processor is not None:
            eeg_processor.start()
        if camera_publisher is not None:
            camera_publisher.start()
        if eeg_publisher is not None:
            eeg_publisher.start()
        yield
        if camera_publisher is not None:
            camera_publisher.stop()
        if eeg_publisher is not None:
            eeg_publisher.stop()
        if vision_loop is not None:
            vision_loop.stop()
        if eeg_processor is not None:
            eeg_processor.stop()
        logger.info("═══ 애플리케이션 종료 완료 ═══")

    app = FastAPI(
        title="Drowsiness Monitor",
        description="카메라(RealSense IR) + EEG(Muse2) 다중모달 실시간 졸음 감지 API",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    router = create_router(
        camera_state=camera_state,
        eeg_state=eeg_state,
        fusion=fusion_engine,
        video_feed_url=video_feed_url,
        enable_ingest=service_mode == "orchestrator",
    )
    app.include_router(router)

    if muse_inference_api is not None:
        app.mount("/eeg", muse_inference_api.app)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# 직접 실행
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
