"""
고급 EEG 전처리 모듈 - Muse2 노이즈 대응
==========================================

개선 사항:
1. 60Hz 노치 필터 (전력 주파수 제거)
2. EOG 아티팩트 감지 및 제거 (눈 깜빡임)
3. 동작 아티팩트 감지 및 제거 (기기 움직임, 머리카락)
4. 신호 품질 지표
5. 적응형 필터링

사용:
  from advanced_preprocessing import AdvancedPreprocessor
  
  preprocessor = AdvancedPreprocessor()
  cleaned_signal = preprocessor.process(raw_signal)
  quality_score = preprocessor.get_quality_score()
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
from scipy.ndimage import uniform_filter1d
from typing import Tuple


class AdvancedPreprocessor:
    FS = 256  # 샘플레이트
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # 전역 필터 계수들 (성능 최적화)
        nyq = 0.5 * self.FS
        
        # 메인 밴드패스 필터 (0.5-40Hz)
        self.bp_b, self.bp_a = butter(4, [0.5/nyq, 40/nyq], btype='band')
        
        # 60Hz 노치 필터 (전력 주파수)
        self.notch_b, self.notch_a = iirnotch(60, 30, self.FS)
        
        # 고역 필터 (0.5Hz 이상) - DC 드리프트 제거용
        self.high_b, self.high_a = butter(2, 0.5/nyq, btype='high')
        
        # 통계 저장
        self.last_quality_score = 1.0
        self.last_artifact_ratio = 0.0
    
    def _detect_eog_artifacts(self, signal: np.ndarray, threshold: float = 2.5) -> np.ndarray:
        """
        EOG (눈 깜빡임) 아티팩트 감지
        
        특징:
        - 매우 빠른 고진폭 변화
        - 일반적으로 0.5-1초 지속
        - AF7/AF8에서 특히 두드러짐
        """
        # 미분으로 빠른 변화 감지
        diff = np.abs(np.diff(signal, prepend=signal[0]))
        
        # 롤링 표준편차 (1초 윈도우)
        window = int(self.FS * 1.0)
        rolling_std = uniform_filter1d(np.abs(diff), size=window, mode='nearest')
        rolling_mean = uniform_filter1d(np.abs(diff), size=window, mode='nearest')
        
        # 변화율이 높은 구간 감지
        threshold_val = np.median(diff) + threshold * np.std(diff)
        rapid_change = diff > threshold_val
        
        # 아티팩트 마스킹 (팽창 처리)
        artifact_mask = np.zeros_like(signal, dtype=bool)
        dilation_size = int(self.FS * 0.5)  # 500ms 팽창
        
        for i in range(len(rapid_change)):
            if rapid_change[i]:
                start = max(0, i - dilation_size)
                end = min(len(artifact_mask), i + dilation_size)
                artifact_mask[start:end] = True
        
        return artifact_mask
    
    def _detect_motion_artifacts(self, signal: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        동작 아티팩트 감지 (기기 움직임, 머리카락)
        
        특징:
        - 넓은 주파수 범위의 고진폭 신호
        - 불규칙한 패턴
        - 수십 밀리초에서 수 초의 지속시간
        """
        artifact_mask = np.zeros_like(signal, dtype=bool)
        
        # 1초 윈도우별로 RMS 계산
        window = int(self.FS * 1.0)
        
        for i in range(0, len(signal) - window, window // 2):
            window_data = signal[i:i+window]
            rms = np.sqrt(np.mean(window_data ** 2))
            
            # 동적 임계값 (전체 신호의 통계 기반)
            global_rms = np.sqrt(np.mean(signal ** 2))
            rms_threshold = global_rms * threshold
            
            if rms > rms_threshold:
                artifact_mask[i:min(i+window, len(signal))] = True
        
        return artifact_mask
    
    def _detect_drift(self, signal: np.ndarray, window_sec: float = 5.0) -> np.ndarray:
        """
        DC 드리프트 감지
        
        길이 방향 트렌드 변화 감지
        """
        window_size = int(self.FS * window_sec)
        
        if len(signal) < window_size:
            return np.zeros_like(signal, dtype=bool)
        
        # Savitzky-Golay 필터로 트렌드 추출
        try:
            trend = savgol_filter(signal, window_size // 2 * 2 + 1, 3)
        except:
            trend = uniform_filter1d(signal, size=window_size, mode='nearest')
        
        # 트렌드 변화율이 큰 구간 감지
        trend_diff = np.abs(np.diff(trend, prepend=trend[0]))
        drift_threshold = np.std(trend_diff) * 3
        drift_mask = trend_diff > drift_threshold
        
        return drift_mask
    
    def _interpolate_artifacts(self, signal: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """
        아티팩트 구간을 선형 보간으로 대체
        """
        if not np.any(artifact_mask):
            return signal.copy()
        
        signal_clean = signal.copy()
        valid_idx = np.where(~artifact_mask)[0]
        
        if len(valid_idx) < 2:
            return signal  # 유효한 데이터가 너무 적으면 원본 반환
        
        # 선형 보간
        bad_idx = np.where(artifact_mask)[0]
        signal_clean[bad_idx] = np.interp(bad_idx, valid_idx, signal[valid_idx])
        
        return signal_clean
    
    def _calculate_quality_score(self, signal: np.ndarray, 
                                 artifact_ratio: float) -> float:
        """
        신호 품질 점수 계산 (0~1)
        
        고려 사항:
        - 아티팩트 비율
        - 신호 변동성
        - 신호 범위
        """
        # 아티팩트 비율 기반
        quality_from_artifacts = 1.0 - min(artifact_ratio * 2, 1.0)
        
        # 신호 통계 기반
        signal_std = np.std(signal)
        signal_mean = np.abs(np.mean(signal))
        
        # 너무 약한 신호 또는 너무 강한 신호 페널티
        if signal_std < 5:
            quality_from_stability = 0.3
        elif signal_std > 200:
            quality_from_stability = 0.5
        else:
            quality_from_stability = 1.0
        
        # 신호 범위 확인
        signal_range = np.max(np.abs(signal))
        if signal_range > 500:
            quality_from_range = 0.6
        else:
            quality_from_range = 1.0
        
        quality = (quality_from_artifacts * 0.6 + 
                  quality_from_stability * 0.2 +
                  quality_from_range * 0.2)
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def process(self, signal: np.ndarray, 
               remove_artifacts: bool = True,
               aggressive: bool = False) -> np.ndarray:
        """
        신호 전체 처리
        
        Args:
            signal: 1D 입력 신호
            remove_artifacts: 아티팩트 제거 활성화
            aggressive: 공격적 필터링 (더 많은 아티팩트 제거, 신호 손상 가능성)
        
        Returns:
            전처리된 신호
        """
        signal = np.asarray(signal, dtype=np.float32)
        
        if signal.ndim != 1:
            raise ValueError(f"1D 신호 필요. 받은 shape: {signal.shape}")
        
        if len(signal) < 33:
            raise ValueError(f"신호가 너무 짧음. 최소 33 샘플 필요.")
        
        original_signal = signal.copy()
        
        # 1단계: 60Hz 노치 필터 (전력 주파수 제거)
        signal = filtfilt(self.notch_b, self.notch_a, signal)
        
        # 2단계: 메인 밴드패스 필터 (0.5-40Hz)
        signal = filtfilt(self.bp_b, self.bp_a, signal)
        
        # 3단계: 아티팩트 감지 및 제거
        artifact_ratio = 0.0
        if remove_artifacts and len(signal) > self.FS * 2:
            eog_mask = self._detect_eog_artifacts(signal, 
                                                  threshold=3.0 if aggressive else 2.5)
            motion_mask = self._detect_motion_artifacts(signal,
                                                       threshold=2.5 if aggressive else 3.0)
            
            artifact_mask = eog_mask | motion_mask
            artifact_ratio = float(np.sum(artifact_mask)) / len(signal)
            
            # 너무 많은 아티팩트면 경고
            if artifact_ratio > 0.5:
                if self.verbose:
                    print(f"⚠️  높은 아티팩트 비율: {artifact_ratio*100:.1f}%")
            
            # 아티팩트 제거
            if np.any(artifact_mask):
                signal = self._interpolate_artifacts(signal, artifact_mask)
        
        # 4단계: DC 드리프트 제거 (고역 필터)
        signal = filtfilt(self.high_b, self.high_a, signal)
        
        # 5단계: 중앙값 제거
        signal = signal - np.median(signal)
        
        # 6단계: 클리핑 및 정규화
        signal = np.clip(signal, -100, 100)
        
        signal_mean = np.mean(signal)
        signal_std = np.std(signal) + 1e-6
        signal = (signal - signal_mean) / signal_std
        
        # 품질 점수 계산
        self.last_artifact_ratio = artifact_ratio
        self.last_quality_score = self._calculate_quality_score(signal, artifact_ratio)
        
        return signal.astype(np.float32)
    
    def get_quality_score(self) -> float:
        """마지막 처리된 신호의 품질 점수"""
        return self.last_quality_score
    
    def get_artifact_ratio(self) -> float:
        """마지막 처리된 신호의 아티팩트 비율"""
        return self.last_artifact_ratio


# ============================================================
# 편의 함수들
# ============================================================

_default_preprocessor: AdvancedPreprocessor = None

def get_default_preprocessor(verbose: bool = False) -> AdvancedPreprocessor:
    """기본 전처리기 싱글턴"""
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = AdvancedPreprocessor(verbose=verbose)
    return _default_preprocessor


def preprocess_signal(signal: np.ndarray, 
                     remove_artifacts: bool = True,
                     aggressive: bool = False) -> Tuple[np.ndarray, float, float]:
    """
    편의 함수: 신호 처리 + 품질 메트릭
    
    Returns:
        (processed_signal, quality_score, artifact_ratio)
    """
    preprocessor = get_default_preprocessor()
    processed = preprocessor.process(signal, remove_artifacts, aggressive)
    return processed, preprocessor.get_quality_score(), preprocessor.get_artifact_ratio()
