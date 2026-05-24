"""
고급 후처리 모듈 - 상태 결정 개선
===================================

개선 사항:
1. 향상된 히스테리시스 필터
2. 시간 기반 상태 안정화
3. 노이즈 기반 신뢰도 가중치
4. 다중 스케일 평활화

사용:
  from advanced_postprocessing import AdvancedPostprocessor
  
  postprocessor = AdvancedPostprocessor()
  states, confidences = postprocessor.process(probabilities, quality_scores)
"""

import numpy as np
from typing import Tuple, List
from collections import deque


class AdvancedPostprocessor:
    """확률값을 견고한 상태로 변환하는 후처리 엔진"""
    
    def __init__(self, 
                 history_size: int = 10,
                 confidence_threshold: float = 0.5,
                 verbose: bool = False):
        """
        Args:
            history_size: 상태 결정에 사용할 최근 윈도우 개수
            confidence_threshold: 상태 변경의 최소 신뢰도
            verbose: 디버그 출력
        """
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # 상태 히스토리
        self.prob_history = deque(maxlen=history_size)
        self.quality_history = deque(maxlen=history_size)
        self.current_state = 0  # 0: awake, 1: sleep
        self.state_confidence = 1.0
    
    def _weighted_hysteresis(self, prob: float, 
                            quality: float = 1.0,
                            high: float = 0.6,
                            low: float = 0.4) -> Tuple[int, float]:
        """
        가중치 기반 히스테리시스
        
        Args:
            prob: 원본 확률 (0~1)
            quality: 신호 품질 (0~1) - 낮으면 변경 어려움
            high: 확률이 이보다 높으면 sleep 상태로 변경
            low: 확률이 이보다 낮으면 awake 상태로 변경
        
        Returns:
            (state, confidence)
        """
        # 품질이 낮으면 상태 변경을 꺼려함
        effective_high = high + (1 - quality) * 0.15
        effective_low = low - (1 - quality) * 0.15
        
        if prob > effective_high:
            new_state = 1  # sleep
            # 확률이 높을수록 높은 신뢰도
            confidence = (prob - effective_high) / (1 - effective_high)
        elif prob < effective_low:
            new_state = 0  # awake
            # 확률이 낮을수록 높은 신뢰도
            confidence = (effective_low - prob) / effective_low
        else:
            # 하이스테리시스 구간: 상태 유지
            new_state = self.current_state
            # 중간 구간이므로 신뢰도 낮음
            confidence = 0.5
        
        return new_state, float(np.clip(confidence, 0.0, 1.0))
    
    def _majority_vote(self, probs: List[float], 
                      qualities: List[float],
                      high: float = 0.6,
                      low: float = 0.4) -> Tuple[int, float]:
        """
        최근 윈도우들의 다수결 투표
        
        Returns:
            (majority_state, consensus_confidence)
        """
        if len(probs) == 0:
            return self.current_state, self.state_confidence
        
        states = []
        confidences = []
        
        for prob, quality in zip(probs, qualities):
            state, conf = self._weighted_hysteresis(prob, quality, high, low)
            states.append(state)
            confidences.append(conf)
        
        states = np.array(states)
        confidences = np.array(confidences)
        
        # 가중 다수결 (신뢰도로 가중)
        sleep_votes = np.sum(confidences[states == 1])
        awake_votes = np.sum(confidences[states == 0])
        
        if sleep_votes > awake_votes:
            majority = 1
            consensus = sleep_votes / (sleep_votes + awake_votes + 1e-6)
        else:
            majority = 0
            consensus = awake_votes / (sleep_votes + awake_votes + 1e-6)
        
        return int(majority), float(consensus)
    
    def _exponential_smoothing(self, probs: List[float], 
                              alpha: float = 0.3) -> float:
        """
        지수 평활화 (최근 데이터에 높은 가중치)
        """
        if len(probs) == 0:
            return 0.5
        
        smoothed = probs[0]
        for prob in probs[1:]:
            smoothed = alpha * prob + (1 - alpha) * smoothed
        
        return smoothed
    
    def process(self, prob: float, 
               quality: float = 1.0,
               high: float = 0.6,
               low: float = 0.4) -> Tuple[int, float, dict]:
        """
        새 윈도우의 확률 처리
        
        Args:
            prob: 현재 윈도우의 확률 (0~1)
            quality: 신호 품질 (0~1)
            high: 히스테리시스 상한 (sleep 판정 임계값)
            low: 히스테리시스 하한 (awake 판정 임계값)
        
        Returns:
            (state, confidence, metadata)
            - state: 0 (awake) or 1 (sleep)
            - confidence: 상태에 대한 신뢰도 (0~1)
            - metadata: 추가 정보 dict
        """
        # 히스토리 추가
        self.prob_history.append(prob)
        self.quality_history.append(quality)
        
        # 현재 윈도우의 가중치 히스테리시스
        weighted_state, weighted_conf = self._weighted_hysteresis(
            prob, quality, high, low
        )
        
        # 히스토리가 충분하면 다수결
        if len(self.prob_history) >= 3:
            majority_state, consensus = self._majority_vote(
                list(self.prob_history), 
                list(self.quality_history),
                high, low
            )
        else:
            majority_state = weighted_state
            consensus = weighted_conf
        
        # 상태 전환 규칙
        state_changed = (majority_state != self.current_state)
        
        if state_changed:
            # 상태 변경 시 높은 신뢰도 필요
            if consensus > self.confidence_threshold:
                self.current_state = majority_state
                self.state_confidence = consensus
            # else: 신뢰도 부족하면 상태 유지
        else:
            self.current_state = majority_state
            self.state_confidence = consensus
        
        metadata = {
            'state_changed': state_changed,
            'history_size': len(self.prob_history),
            'prob_mean': float(np.mean(list(self.prob_history))),
            'prob_std': float(np.std(list(self.prob_history))),
            'quality_mean': float(np.mean(list(self.quality_history))),
            'consensus': consensus,
            'weighted_conf': weighted_conf,
        }
        
        if self.verbose:
            print(f"  State: {self.current_state} (conf={self.state_confidence:.2f}), "
                  f"History: {len(self.prob_history)}, Changed: {state_changed}")
        
        return self.current_state, self.state_confidence, metadata
    
    def get_state(self) -> Tuple[int, float]:
        """현재 상태와 신뢰도"""
        return self.current_state, self.state_confidence
    
    def reset(self):
        """상태 초기화"""
        self.prob_history.clear()
        self.quality_history.clear()
        self.current_state = 0
        self.state_confidence = 1.0


# ============================================================
# 편의 함수들
# ============================================================

_default_postprocessor: AdvancedPostprocessor = None

def get_default_postprocessor(history_size: int = 10,
                             confidence_threshold: float = 0.5,
                             verbose: bool = False) -> AdvancedPostprocessor:
    """기본 후처리기 싱글턴"""
    global _default_postprocessor
    if _default_postprocessor is None:
        _default_postprocessor = AdvancedPostprocessor(
            history_size=history_size,
            confidence_threshold=confidence_threshold,
            verbose=verbose
        )
    return _default_postprocessor


class HysteresisFilter:
    """간단한 히스테리시스 필터 (기존 코드 호환성)"""
    
    def __init__(self, high: float = 0.6, low: float = 0.4):
        self.high = high
        self.low = low
        self.state = 0
    
    def update(self, prob: float) -> int:
        """확률 기반 상태 업데이트"""
        if prob > self.high:
            self.state = 1
        elif prob < self.low:
            self.state = 0
        return self.state
    
    def get_state(self) -> int:
        """현재 상태"""
        return self.state
