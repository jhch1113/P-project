"""
로컬 실시간 테스트 스크립트
---------------------------------
LSL(Muse) 또는 시뮬레이션 모드에서 `EEGChunk`를 읽어
간단한 대체 확률(prob_raw)을 계산한 뒤 `DrowsinessScorer`로
점수화하여 경고를 출력한다.

사용법:
  python3 local_realtime.py --duration 30
"""
import time
import argparse
import numpy as np

from eeg_data_source import create_reader, EEGChunk
from drowsiness_scorer import DrowsinessScorer
from advanced_preprocessing import AdvancedPreprocessor


def estimate_prob_from_chunk(chunk: EEGChunk, preprocessor: AdvancedPreprocessor) -> float:
    """
    향상된 특징 기반 확률 추정
    (실제 모델을 대체하는 용도이며, 테스트/디버깅 전용)
    """
    af7 = np.asarray(chunk.af7)
    af8 = np.asarray(chunk.af8)
    
    # 고급 전처리 (아티팩트 제거)
    af7_proc = preprocessor.process(af7, remove_artifacts=True)
    af8_proc = preprocessor.process(af8, remove_artifacts=True)
    
    # 신호 품질 점수 기반 신뢰도
    quality = preprocessor.get_quality_score()
    
    # 평균 신호 사용
    x = (af7_proc + af8_proc) * 0.5
    
    # FFT 기반 주파수 분석
    N = len(x)
    freqs = np.fft.rfftfreq(N, d=1/256)
    P = np.abs(np.fft.rfft(x))**2
    
    # 밴드 파워 계산
    def band_power(p, f, lo, hi):
        mask = (f >= lo) & (f <= hi)
        return p[mask].sum()
    
    alpha = band_power(P, freqs, 8, 13)      # 알파: 졸음 지표
    beta = band_power(P, freqs, 13, 30)      # 베타: 각성 지표
    theta = band_power(P, freqs, 4, 8)       # 세타: 졸음 지표
    total = P.sum() + 1e-9
    
    # 특징: 알파/베타 비율 (높을수록 졸음)
    alpha_beta_ratio = alpha / (beta + 1e-9)
    theta_alpha_ratio = theta / (alpha + 1e-9)
    
    # 간단한 점수화
    # 알파 비율이 높거나 시그마 활동이 많으면 졸음
    score = (alpha_beta_ratio * 0.6 + theta_alpha_ratio * 0.2) / (1.0 + alpha_beta_ratio + theta_alpha_ratio)
    
    # 신호 품질로 신뢰도 조정
    prob = float(np.clip(score * quality, 0.0, 1.0))
    
    return prob


def main(duration: int = 30, source: str = "muse2"):
    print(f"로컬 실시간 테스트 시작 (source={source}, duration={duration}s)")
    reader = create_reader(source)
    if not reader.connect():
        print("리더 연결 실패")
        return

    # 전처리기 초기화
    preprocessor = AdvancedPreprocessor(verbose=False)

    # 보수적 임계값: 시뮬레이션 및 실환경에서 과민하게 경고가 뜨는 것을 방지
    scorer = DrowsinessScorer(
        window_size=60,
        drowsy_threshold=0.7,
        accumulated_time_limit=25.0,
        instant_alert_threshold=0.95,
    )
    start = time.time()
    last_seq = -1
    seq_counter = 0
    from collections import deque
    prob_history = deque(maxlen=5)

    try:
        while time.time() - start < duration:
            chunk = reader.read_chunk(timeout=2.0)
            if chunk is None:
                print("타임아웃: 청크 수신 없음")
                continue

            prob = estimate_prob_from_chunk(chunk, preprocessor)
            
            # 간단한 평활화: 최근 N 프레임 평균을 사용
            prob_history.append(prob)
            smoothed_prob = float(sum(prob_history) / len(prob_history))

            # sequence id 보정 (read_chunk가 직접 호출될 때 증가)
            chunk.sequence_id = seq_counter
            seq_counter += 1

            score = scorer.score(prob_raw=smoothed_prob, prob_scaled=None, state=0)

            ts = time.strftime('%H:%M:%S')
            alert = "ALERT" if score.should_alert else ""
            quality = f"Q:{preprocessor.get_quality_score():.2f}"
            print(f"[{ts}] seq={chunk.sequence_id} prob={prob:.3f} smoothed={score.smoothed_score:.1f} {score.risk_level} {quality} {alert}")

            last_seq = chunk.sequence_id
    except KeyboardInterrupt:
        print("중단됨")
    finally:
        reader.disconnect()
        print("종료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--source', choices=['muse2','file','tcp'], default='muse2')
    args = parser.parse_args()
    main(duration=args.duration, source=args.source)
