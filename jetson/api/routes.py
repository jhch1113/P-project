"""
api/routes.py
=============
FastAPI 라우터 정의.

엔드포인트:
  GET /               : 통합 대시보드 HTML
  GET /video_feed     : MJPEG 실시간 IR 영상 스트림
  GET /metrics/camera : 카메라 기반 지표 (EAR, MAR, PERCLOS, Pitch)
  GET /metrics/eeg    : EEG 기반 지표 (Alpha/Beta, Theta, Blink rate)
  GET /metrics/fusion : 최종 융합 졸음 판정 결과
  GET /metrics/all    : 세 지표를 한 번에 반환 (대시보드 단일 요청 최적화)
"""

import logging

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from api.dashboard import build_dashboard_html
from api.fusion import DrowsinessFusion
from core.state import SharedCameraState, SharedEEGState

logger = logging.getLogger(__name__)


def create_router(
    camera_state: SharedCameraState,
    eeg_state: SharedEEGState,
    fusion: DrowsinessFusion,
) -> APIRouter:
    """
    의존성을 주입받아 라우터를 생성한다.
    전역 변수 없이 클로저 기반으로 상태를 참조하므로 테스트 대체가 용이하다.
    """
    router = APIRouter()

    # ------------------------------------------------------------------
    # MJPEG 스트림 생성기
    # ------------------------------------------------------------------
    def _generate_mjpeg():
        """
        새 프레임 도착 시까지 블로킹 대기 후 yield.
        threading.Event 기반이므로 CPU 소비 없음.
        """
        while True:
            frame = camera_state.get_frame(timeout=2.0)
            if frame is None:
                logger.debug("MJPEG: 프레임 타임아웃, 재시도 중...")
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @router.get("/", response_class=HTMLResponse, summary="통합 졸음 감지 대시보드")
    def index():
        return HTMLResponse(content=build_dashboard_html())

    @router.get("/video_feed", summary="MJPEG 실시간 IR 영상 스트림")
    def video_feed():
        return StreamingResponse(
            _generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @router.get("/metrics/camera", summary="카메라 기반 졸음 지표")
    def get_camera_metrics():
        return camera_state.get_metrics().to_dict()

    @router.get("/metrics/eeg", summary="EEG 기반 졸음 지표")
    def get_eeg_metrics():
        return eeg_state.get_metrics().to_dict()

    @router.get("/metrics/fusion", summary="다중모달 최종 졸음 판정")
    def get_fusion_result():
        cam = camera_state.get_metrics()
        eeg = eeg_state.get_metrics()
        return fusion.fuse(cam, eeg).to_dict()

    @router.get("/metrics/all", summary="카메라 + EEG + 융합 결과 통합 응답")
    def get_all_metrics():
        """
        대시보드가 세 번 요청하는 대신 한 번에 수신하여 네트워크 오버헤드를 줄인다.
        """
        cam = camera_state.get_metrics()
        eeg = eeg_state.get_metrics()
        fusion_result = fusion.fuse(cam, eeg)
        return {
            "camera": cam.to_dict(),
            "eeg": eeg.to_dict(),
            "fusion": fusion_result.to_dict(),
            # 카메라 활성 여부를 함께 전달 → 대시보드가 "카메라 연결 끊김"을 표시 가능
            "camera_active": camera_state.is_active(),
        }

    @router.post("/metrics/camera/reset_calibration", summary="카메라 캘리브레이션 재시작")
    def reset_camera_calibration():
        """대시보드에서 캘리브레이션 초기화를 요청할 때 호출되는 엔드포인트."""
        camera_state.request_calibration_reset()
        return {"status": "ok", "message": "Calibration reset requested"}

    @router.get("/debug/raw", summary="[진단] 원시 상태 덤프 — 문제 발생 시 확인")
    def debug_raw():
        """
        카메라·EEG의 원시 상태값과 계산된 서브-점수를 모두 반환한다.
        점수가 예상과 다를 때 이 엔드포인트로 실제 수신값을 확인할 것.

        주요 확인 항목:
          camera.is_calibrated  : False면 캘리브레이션 미완료(5초 소요) → ear/perclos 점수 강제 0
          camera.status         : "No Face"면 FaceMesh 검출 실패 → 모든 값 0
          camera_active         : False면 RealSense 파이프라인에서 프레임 미수신
          camera_score          : 졸음 징후 점수 (0 = 정상/졸음 없음, 1 = 최대 졸음)
          score_note            : 각 점수가 0인 이유 요약
        """
        cam = camera_state.get_metrics()
        eeg = eeg_state.get_metrics()
        fusion_result = fusion.fuse(cam, eeg)
        cs = fusion_result.camera_score

        # 점수가 0인 이유 자동 진단
        notes = []
        if not camera_state.is_active():
            notes.append("❌ 카메라 비활성: RealSense에서 프레임이 도착하지 않음")
        elif cam.status == "Waiting...":
            notes.append("⏳ 카메라 시작 대기 중")
        elif cam.status == "No Face":
            notes.append("🔍 얼굴 미검출: FaceMesh가 얼굴을 찾지 못함 → 모든 값 0")
        elif not cam.is_calibrated:
            notes.append(f"🔧 캘리브레이션 진행 중: {cam.status} — 완료까지 ear/perclos 점수 = 0")
        else:
            if cs.ear == 0.0:
                notes.append(f"👁 EAR({cam.ear:.3f}) > threshold({cam.threshold:.3f}) → 눈 열림 = 정상 = 점수 0")
            if cs.perclos == 0.0:
                notes.append(f"📊 PERCLOS({cam.perclos:.1f}%) = 0 → 눈 감김 없음 = 점수 0")
            if cs.mar == 0.0:
                notes.append(f"👄 MAR({cam.mar:.3f}) ≤ 0.6 → 하품 없음 = 점수 0")
            if cs.pitch == 0.0:
                notes.append(f"🎯 Pitch({cam.pitch:.1f}°) ≤ 25° → 머리 정상 = 점수 0")
            if not notes:
                notes.append("✅ 모든 카메라 지표 정상 반영 중")

        return {
            "camera_active": camera_state.is_active(),
            "seconds_since_last_frame": round(camera_state.seconds_since_last_frame(), 2),
            "camera_raw": cam.to_dict(),
            "eeg_raw": eeg.to_dict(),
            "camera_score_breakdown": {
                "ear_score": round(cs.ear, 4),
                "perclos_score": round(cs.perclos, 4),
                "mar_score": round(cs.mar, 4),
                "pitch_score": round(cs.pitch, 4),
                "total": round(cs.total, 4),
            },
            "fusion_final_score": round(fusion_result.final_score, 4),
            "fusion_level": fusion_result.level.value,
            "eeg_available": fusion_result.eeg_available,
            "score_note": notes,
        }

    return router
