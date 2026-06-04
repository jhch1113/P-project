"""
졸음운전 점수화 로직
==================
FastAPI 서버에서 받은 window별 확률/상태를 기반으로
실시간 졸음운전 위험도 점수를 계산.

사용처: Jetson Orin Nano + OpenCV에서 1초 단위로 호출
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class DrowsinessScore:
    """한 프레임의 점수 결과"""
    timestamp: float              # 측정 시간
    prob_raw: float              # API 응답: 원본 확률값
    prob_scaled: Optional[float]  # API 응답: 스케일된 확률값
    state: int                   # API 응답: 0(sleep) or 1(awake)
    
    # === 점수화 결과 ===
    instant_score: float         # 0~100: 현재 윈도우 위험도 (높을수록 졸음)
    smoothed_score: float        # 0~100: 최근 N윈도우 평균 (안정적)
    accumulated_drowsy_time: float  # 초: 최근 30초 동안 졸음 누적 시간
    
    risk_level: str              # "안전", "주의", "위험"
    should_alert: bool           # True = 즉시 경고 필요


class DrowsinessScorer:
    """
    실시간 졸음운전 점수 계산기.
    - 1초마다 API 응답 1개 (= 5초 윈도우 1개) 받음
    - 최근 데이터로 위험도 누적 계산
    """

    def __init__(
        self,
        window_size: int = 30,           # 평활화에 사용할 최근 윈도우 개수
        drowsy_threshold: float = 0.55,  # 이 이상이면 졸음 판정
        accumulated_time_limit: float = 20.0,  # 30초 중 20초 이상 졸음 = 위험
        instant_alert_threshold: float = 0.80,  # 순간 확률이 이 이상 = 즉시 경고
        prob_is_awake: bool = True,      # True면 입력 확률을 "각성 확률"로 간주
    ):
        self.window_size = window_size
        self.drowsy_threshold = drowsy_threshold
        self.accumulated_time_limit = accumulated_time_limit
        self.instant_alert_threshold = instant_alert_threshold
        self.prob_is_awake = prob_is_awake
        
        # 최근 window_size개 윈도우의 확률값 저장 (평활화용)
        self.prob_history = deque(maxlen=window_size)
        
        # 최근 30초 동안의 state 기록 (1초 단위 = 30개)
        self.state_history = deque(maxlen=30)
        self.state_timestamps = deque(maxlen=30)
        
        self.last_call_time = None

    def score(self, prob_raw: float, prob_scaled: Optional[float] = None, state: int = 0) -> DrowsinessScore:
        """
        API 응답 1개를 받아 점수를 계산합니다.
        
        Args:
            prob_raw: API의 prob_raw (0~1)
            prob_scaled: API의 prob_scaled (0~1) - None이면 prob_raw 사용
            state: API의 state (0=sleep, 1=awake) - 참고용
        
        Returns:
            DrowsinessScore 객체 (여러 점수 지표 포함)
        """
        current_time = time.time()
        
        # === 1. 즉각 점수 (현재 윈도우만 기반) ===
        # API 기본 출력은 각성 확률로 가정하고, 점수화는 졸림 확률로 계산한다.
        prob_value = prob_scaled if prob_scaled is not None else prob_raw
        drowsy_prob = (1.0 - prob_value) if self.prob_is_awake else prob_value
        drowsy_prob = float(max(0.0, min(1.0, drowsy_prob)))
        instant_score = drowsy_prob * 100  # 0~100 스케일
        
        # === 2. 확률값 히스토리 업데이트 ===
        self.prob_history.append(drowsy_prob)
        
        # === 3. 평활화 점수 (최근 window_size개 평균) ===
        if len(self.prob_history) > 0:
            smoothed_score = sum(self.prob_history) / len(self.prob_history) * 100
        else:
            smoothed_score = instant_score
        
        # === 4. 상태 히스토리 업데이트 ===
        # 졸림 판정은 "졸림 확률" 기준
        is_drowsy_now = drowsy_prob > self.drowsy_threshold
        self.state_history.append(1 if is_drowsy_now else 0)
        self.state_timestamps.append(current_time)
        
        # === 5. 누적 졸음 시간 (최근 30초) ===
        # state_history가 최대 30개 = 최대 30초
        drowsy_count = sum(self.state_history)
        drowsy_time_sec = drowsy_count * 1.0  # 1초씩 * 개수
        
        # 만약 실제 시간 간격이 1초가 아니면 보정
        if len(self.state_timestamps) >= 2:
            first_ts = self.state_timestamps[0]
            last_ts = self.state_timestamps[-1]
            time_span = last_ts - first_ts
            if time_span > 0:
                # 비율 계산으로 보정
                drowsy_time_sec = (drowsy_count / len(self.state_history)) * time_span
        
        accumulated_drowsy_time = drowsy_time_sec
        
        # === 6. 위험도 레벨 판정 ===
        if smoothed_score >= 80:
            risk_level = "위험"
        elif smoothed_score >= 60:
            risk_level = "주의"
        else:
            risk_level = "안전"
        
        # === 7. 즉시 경고 필요 여부 ===
        should_alert = (
            instant_score >= self.instant_alert_threshold * 100 or  # 순간 확률 매우 높음
            accumulated_drowsy_time >= self.accumulated_time_limit   # 최근 30초 중 대부분 졸음
        )
        
        self.last_call_time = current_time
        
        return DrowsinessScore(
            timestamp=current_time,
            prob_raw=prob_raw,
            prob_scaled=prob_scaled,
            state=state,
            instant_score=instant_score,
            smoothed_score=smoothed_score,
            accumulated_drowsy_time=accumulated_drowsy_time,
            risk_level=risk_level,
            should_alert=should_alert,
        )

    def reset(self):
        """점수 계산기 초기화"""
        self.prob_history.clear()
        self.state_history.clear()
        self.state_timestamps.clear()
        self.last_call_time = None


# ============================================================
# 사용 예제
# ============================================================
if __name__ == "__main__":
    import requests
    
    # 1. 점수화 객체 생성
    scorer = DrowsinessScorer(
        window_size=30,
        drowsy_threshold=0.55,
        accumulated_time_limit=20.0,
        instant_alert_threshold=0.80,
    )
    
    # 2. FastAPI 서버에서 세션 생성
    print("🔄 FastAPI 서버 연결 중...")
    BASE_URL = "http://localhost:8000"
    
    # 세션 시작
    resp = requests.post(f"{BASE_URL}/session/start")
    session_id = resp.json()["session_id"]
    print(f"✅ 세션 생성: {session_id}")
    
    # 3. 시뮬레이션: 1초씩 7번 EEG 데이터 전송
    print("\n📊 실시간 점수 계산 시작...")
    print("-" * 70)
    
    for sec in range(1, 8):
        # ⚠️ 실제에서는 OpenCV/MUSE2에서 1초 청크를 받음
        import numpy as np
        np.random.seed(100 + sec)
        n_samples = 256  # 1초 = 256 샘플 (FS=256)
        
        # 인위적 EEG 신호 생성 (실제에서는 MUSE2 데이터)
        af7 = (15 * np.sin(2*np.pi*10*np.arange(n_samples)/256) + 
               np.random.normal(0, 8, n_samples)).astype(float).tolist()
        af8 = (12 * np.sin(2*np.pi*10*np.arange(n_samples)/256 + 0.5) + 
               np.random.normal(0, 8, n_samples)).astype(float).tolist()
        
        # API에 전송
        payload = {
            "af7": af7,
            "af8": af8,
            "apply_minmax": False,  # 스트리밍에서는 False 권장
        }
        resp = requests.post(f"{BASE_URL}/session/{session_id}/append", json=payload)
        api_response = resp.json()
        
        # API 응답에서 최신 윈도우 데이터 추출
        if api_response["new_results"]:
            for result in api_response["new_results"]:
                # ✨ 점수화!
                score_obj = scorer.score(
                    prob_raw=result["prob_raw"],
                    prob_scaled=result.get("prob_scaled"),
                    state=result["state"],
                )
                
                # 출력
                print(f"[t={sec}s] {score_obj.risk_level:4s} | "
                      f"instant={score_obj.instant_score:5.1f} "
                      f"smoothed={score_obj.smoothed_score:5.1f} "
                      f"accum={score_obj.accumulated_drowsy_time:5.1f}s "
                      f"{'⚠️ ALERT' if score_obj.should_alert else ''}")
    
    # 4. 세션 종료
    requests.post(f"{BASE_URL}/session/{session_id}/end")
    print("-" * 70)
    print("✅ 분석 완료")
