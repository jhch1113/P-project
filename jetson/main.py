"""
main.py
=======
애플리케이션 진입점.

의존성 조립(Dependency Wiring):
  1. 설정 객체 생성 (CameraConfig, DrowsinessConfig, EEGConfig, FusionConfig)
  2. 공유 상태 컨테이너 생성 (SharedCameraState, SharedEEGState)
  3. 백그라운드 루프 생성 (VisionProcessingLoop, EEGProcessor)
  4. 융합 엔진 생성 (DrowsinessFusion)
  5. FastAPI 앱에 라우터 등록
  6. uvicorn 실행

EEG 모드 전환 (StubEEGProcessor → Muse2LSLProcessor):
  아래 EEG_MODE 상수를 "muse2"로 변경.
  실제 Muse2 연동 전 `pip install muselsl pylsl scipy` 필수.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.eeg_processor import Muse2LSLProcessor, StubEEGProcessor
from api.fusion import DrowsinessFusion
from api.routes import create_router
from core.config import CameraConfig, DrowsinessConfig, EEGConfig, FusionConfig
from core.state import SharedCameraState, SharedEEGState
from core.vision_loop import VisionProcessingLoop

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

# ---------------------------------------------------------------------------
# EEG 모드 선택
# ---------------------------------------------------------------------------
# "stub"  : EEG 미연결 → 카메라 점수만으로 최종 졸음 판정 (기본값)
#           StubEEGProcessor는 is_connected=False를 전송하므로
#           Fusion 엔진이 EEG 가중치를 완전히 무시한다.
#           대시보드 EEG 카드는 표시되지만 최종 점수에 영향 없음.
#
# "muse2" : 실제 Muse2 LSL 스트림 연결 → 카메라(55%) + EEG(45%) 융합 판정
#           사전 준비: pip install muselsl pylsl scipy
#                      muselsl stream  (터미널에서 Muse2 BT 스트리밍 시작)
EEG_MODE: str = "stub"

# ---------------------------------------------------------------------------
# 설정 객체 (전체 시스템 설정을 한 곳에서 관리)
# ---------------------------------------------------------------------------
cam_config = CameraConfig()
drw_config = DrowsinessConfig()
eeg_config = EEGConfig()
fusion_config = FusionConfig()

# ---------------------------------------------------------------------------
# 공유 상태 컨테이너
# ---------------------------------------------------------------------------
camera_state = SharedCameraState()
eeg_state = SharedEEGState()

# ---------------------------------------------------------------------------
# 백그라운드 루프 및 융합 엔진 생성
# ---------------------------------------------------------------------------
vision_loop = VisionProcessingLoop(camera_state, cam_config, drw_config)

if EEG_MODE == "muse2":
    eeg_processor = Muse2LSLProcessor(eeg_state, eeg_config)
    logger.info("EEG 모드: Muse2 LSL (실제 장치)")
else:
    eeg_processor = StubEEGProcessor(eeg_state, eeg_config)
    logger.info("EEG 모드: Stub 시뮬레이터 (테스트용)")

fusion_engine = DrowsinessFusion(fusion_config)

# ---------------------------------------------------------------------------
# FastAPI lifespan (시작/종료 훅)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══ 애플리케이션 시작 ═══")
    vision_loop.start()
    eeg_processor.start()
    yield
    vision_loop.stop()
    eeg_processor.stop()
    logger.info("═══ 애플리케이션 종료 완료 ═══")


# ---------------------------------------------------------------------------
# FastAPI 앱 조립
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Drowsiness Monitor",
    description="카메라(RealSense IR) + EEG(Muse2) 다중모달 실시간 졸음 감지 API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

router = create_router(
    camera_state=camera_state,
    eeg_state=eeg_state,
    fusion=fusion_engine,
)
app.include_router(router)


# ---------------------------------------------------------------------------
# 직접 실행
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
