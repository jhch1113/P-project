"""
실시간 졸음운전 감지 (Jetson Orin Nano + OpenCV + MUSE2)
==========================================================
Jetson에서 실행:
1. MUSE2에서 1초씩 EEG 청크 수신
2. FastAPI 서버로 전송 (실시간 변환)
3. 점수화 로직으로 위험도 계산
4. OpenCV로 화면에 표시 + 경고음/진동 발생

구성:
- muse2_reader.py: MUSE2 데이터 수신 (별도 스레드)
- drowsiness_scorer.py: 위험도 점수 계산
- 이 파일: OpenCV UI + 통합 제어
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import requests

from drowsiness_scorer import DrowsinessScorer, DrowsinessScore
from eeg_data_source import EEGReader, EEGChunk, create_reader
from alert_handlers import create_alert_callback


@dataclass
class RealtimeDrowsinessDetector:
    """
    Jetson Orin Nano에서 실시간 졸음운전 감지를 수행합니다.
    """
    
    api_url: str = "http://localhost:8000"  # FastAPI 서버 주소
    alert_callback: Optional[Callable] = None  # 경고 콜백 (진동/음성/LED 등)
    
    def __post_init__(self):
        self.session_id: Optional[str] = None
        self.scorer = DrowsinessScorer()
        self.running = False
        self.latest_score: Optional[DrowsinessScore] = None
        self.lock = threading.Lock()
        
        # 통계
        self.frame_count = 0
        self.alert_count = 0
        self.start_time = time.time()
    
    def start_session(self):
        """FastAPI 세션 시작"""
        try:
            resp = requests.post(f"{self.api_url}/session/start", timeout=5)
            self.session_id = resp.json()["session_id"]
            print(f"✅ 세션 시작: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ 세션 시작 실패: {e}")
            return False
    
    def end_session(self):
        """FastAPI 세션 종료"""
        if self.session_id:
            try:
                requests.post(f"{self.api_url}/session/{self.session_id}/end", timeout=5)
                print(f"✅ 세션 종료: {self.session_id}")
            except Exception as e:
                print(f"⚠️ 세션 종료 실패: {e}")
    
    def process_eeg_chunk(self, eeg_chunk: EEGChunk) -> bool:
        """
        표준화된 EEGChunk를 처리합니다.
        
        Args:
            eeg_chunk: EEGChunk 객체
        
        Returns:
            True if 새 윈도우 결과가 있음
        """
        if not self.session_id:
            return False
        
        try:
            # API에 전송
            payload = {
                "af7": eeg_chunk.af7,
                "af8": eeg_chunk.af8,
                "apply_minmax": False,
            }
            resp = requests.post(
                f"{self.api_url}/session/{self.session_id}/append",
                json=payload,
                timeout=5
            )
            api_response = resp.json()
            
            # 새 윈도우 결과 처리
            if api_response.get("new_results"):
                result = api_response["new_results"][-1]  # 최신 윈도우만
                
                # 점수화
                score = self.scorer.score(
                    prob_raw=result["prob_raw"],
                    prob_scaled=result.get("prob_scaled"),
                    state=result["state"],
                )
                
                with self.lock:
                    self.latest_score = score
                    self.frame_count += 1
                    
                    if score.should_alert:
                        self.alert_count += 1
                        if self.alert_callback:
                            self.alert_callback(score)
                
                return True
            
            return False
        
        except Exception as e:
            print(f"❌ API 호출 실패: {e}")
            return False
    
    def get_latest_score(self) -> Optional[DrowsinessScore]:
        """최신 점수 조회 (스레드 안전)"""
        with self.lock:
            return self.latest_score
    
    def get_stats(self) -> dict:
        """통계 조회"""
        elapsed = time.time() - self.start_time
        return {
            "frame_count": self.frame_count,
            "alert_count": self.alert_count,
            "elapsed_sec": elapsed,
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
        }


class OpenCVDisplay:
    """OpenCV를 사용한 UI 렌더링"""
    
    @staticmethod
    def create_dashboard(score: Optional[DrowsinessScore], stats: dict) -> np.ndarray:
        """
        졸음운전 대시보드 이미지 생성
        
        Returns:
            (1080, 1920, 3) BGR 이미지
        """
        h, w = 1080, 1920
        img = np.ones((h, w, 3), dtype=np.uint8) * 240  # 밝은 배경
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # ===== 제목 =====
        cv2.putText(img, "Real-time Drowsiness Detection", (50, 80),
                    font, 2, (0, 0, 0), 3)
        
        if score is None:
            cv2.putText(img, "Waiting for data...", (50, 200),
                        font, 1.5, (100, 100, 100), 2)
            return img
        
        # ===== 중앙: 위험도 게이지 (원형) =====
        cx, cy = 300, 400
        radius = 150
        
        # 배경 원
        cv2.circle(img, (cx, cy), radius, (200, 200, 200), -1)
        
        # 위험도에 따른 색상
        if score.risk_level == "위험":
            color = (0, 0, 255)  # 빨강
        elif score.risk_level == "주의":
            color = (0, 165, 255)  # 주황
        else:
            color = (0, 255, 0)  # 녹색
        
        # 위험도 게이지 (호)
        angle = int(score.smoothed_score * 1.8)  # 0~100 -> 0~180도
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, 0, angle, color, 15)
        
        # 중앙 점수 표시
        cv2.circle(img, (cx, cy), 80, (255, 255, 255), -1)
        cv2.putText(img, f"{score.smoothed_score:.0f}", (cx-40, cy+20),
                    font, 2.5, color, 3)
        
        # ===== 우측: 상태 표시 =====
        text_x = 600
        y_offset = 200
        
        cv2.putText(img, f"Status: {score.risk_level}", (text_x, y_offset),
                    font, 1.8, color, 2)
        
        y_offset += 80
        cv2.putText(img, f"Instant: {score.instant_score:.1f}", (text_x, y_offset),
                    font, 1.2, (0, 0, 0), 2)
        
        y_offset += 60
        cv2.putText(img, f"Smoothed: {score.smoothed_score:.1f}", (text_x, y_offset),
                    font, 1.2, (0, 0, 0), 2)
        
        y_offset += 60
        cv2.putText(img, f"Drowsy Time: {score.accumulated_drowsy_time:.1f}s", 
                    (text_x, y_offset), font, 1.2, (0, 0, 0), 2)
        
        # ===== 하단: 경고 메시지 =====
        if score.should_alert:
            cv2.rectangle(img, (50, h-150), (w-50, h-50), (0, 0, 255), -1)
            cv2.putText(img, "⚠️ DROWSINESS DETECTED - STAY ALERT ⚠️", 
                        (100, h-80), font, 2, (255, 255, 255), 3)
        
        # ===== 좌측 하단: 통계 =====
        stat_y = h - 250
        cv2.putText(img, f"Frames: {stats['frame_count']}", (50, stat_y),
                    font, 1, (0, 0, 0), 1)
        cv2.putText(img, f"Alerts: {stats['alert_count']}", (50, stat_y + 40),
                    font, 1, (0, 0, 0), 1)
        cv2.putText(img, f"FPS: {stats['fps']:.1f}", (50, stat_y + 80),
                    font, 1, (0, 0, 0), 1)
        
        return img


    def process_eeg_chunk(self, chunk: EEGChunk) -> bool:
        """
        표준화된 EEGChunk를 처리합니다.
        
        Args:
            chunk: EEGChunk 객체
        
        Returns:
            True if 새 윈도우 결과가 있음
        """
        return self.process_eeg_chunk(chunk.af7, chunk.af8)


def main():
    """메인 루프: 실시간 졸음운전 감지"""
    
    print("🚀 졸음운전 감지 시스템 시작")
    
    # ===== 설정 로드 =====
    try:
        from config import load_config
        config = load_config("config.yaml")
    except FileNotFoundError:
        print("⚠️ config.yaml 없음. 기본 설정 사용")
        from config import get_default_config
        config = get_default_config()
    
    print(f"📋 설정: EEG 소스={config.eeg_source.type}, "
          f"API={config.api.url}")
    
    # ===== 감지기 초기화 =====
    alert_cb = create_alert_callback(config.alert)
    detector = RealtimeDrowsinessDetector(
        api_url=config.api.url,
        alert_callback=alert_cb,
    )
    
    if not detector.start_session():
        print("❌ 시작 실패")
        return
    
    # ===== EEG 리더 생성 및 시작 =====
    try:
        eeg_reader = create_reader(
            config.eeg_source.type,
            **config.eeg_source.kwargs
        )
    except Exception as e:
        print(f"❌ EEG 리더 생성 실패: {e}")
        return
    
    def on_eeg_chunk(chunk: EEGChunk):
        """EEG 청크 도착 콜백"""
        detector.process_eeg_chunk(chunk)
    
    try:
        thread = eeg_reader.start_stream(on_eeg_chunk, daemon=False)
    except Exception as e:
        print(f"❌ EEG 스트림 시작 실패: {e}")
        return
    
    # ===== OpenCV 디스플레이 루프 =====
    try:
        print(f"✅ EEG 스트림 시작됨 ({config.eeg_source.type})")
        print("   'q' 키로 종료")
        
        while eeg_reader.running:
            score = detector.get_latest_score()
            stats = detector.get_stats()
            stats.update(eeg_reader.get_stats())
            
            # 디스플레이 유형별 처리
            if config.ui.display_type == "opencv":
                img = OpenCVDisplay.create_dashboard(score, stats)
                cv2.imshow("Drowsiness Detection", img)
                
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            
            elif config.ui.display_type == "headless":
                # 헤드리스: 콘솔에만 출력
                if score and detector.frame_count % 10 == 0:
                    print(f"[t={detector.frame_count}s] "
                          f"{score.risk_level:4s} | "
                          f"score={score.smoothed_score:.1f}")
                time.sleep(0.1)
            
            else:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️ 종료 중...")
    
    finally:
        eeg_reader.stop_stream()
        detector.end_session()
        cv2.destroyAllWindows()
        print("✅ 종료 완료")


if __name__ == "__main__":
    main()
