"""
api/fusion.py
=============
다중모달(카메라 + EEG) 졸음 융합 엔진.

설계 목표:
  1. 카메라 지표(EAR, PERCLOS, MAR, Pitch)와 EEG 지표(Alpha/Beta, Theta, Blink)를
     각각 0-1 점수로 정규화한 후 가중 합산하여 최종 졸음 점수를 산출한다.
  2. EEG 미연결 시: 카메라 점수만으로 동작 (단일 모달 모드)
  3. 향후 확장: ML 기반 후처리(예: XGBoost, SVM 분류기) 플러그인 가능하도록
     FusionResult에 raw 점수를 보존한다.

졸음 판정 기준 (FusionConfig.threshold_* 조정 가능):
  NORMAL  : score < 0.25
  CAUTION : 0.25 ≤ score < 0.45
  WARNING : 0.45 ≤ score < 0.65
  DROWSY  : score ≥ 0.65
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from core.config import FusionConfig
from core.state import CameraMetrics, EEGMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drowsiness Level Enum
# ---------------------------------------------------------------------------
class DrowsinessLevel(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DROWSY = "DROWSY"

    @property
    def is_alert_needed(self) -> bool:
        return self in (DrowsinessLevel.WARNING, DrowsinessLevel.DROWSY)

    @property
    def color_hex(self) -> str:
        return {
            DrowsinessLevel.NORMAL: "#4caf50",
            DrowsinessLevel.CAUTION: "#ff9800",
            DrowsinessLevel.WARNING: "#ff5722",
            DrowsinessLevel.DROWSY: "#ff4d4d",
        }[self]


# ---------------------------------------------------------------------------
# Sub-score containers
# ---------------------------------------------------------------------------
@dataclass
class CameraScore:
    """카메라 기반 서브-점수 분해."""
    ear: float       # 눈 감김 정도 (0=완전 열림, 1=완전 감김)
    perclos: float   # PERCLOS 기반 (0=0%, 1=100%)
    mar: float       # 하품 정도 (0=정상, 1=최대 하품)
    pitch: float     # 머리 숙임 정도 (0=정상, 1=심각)
    total: float     # 가중 합산 최종 점수 (0–1)


@dataclass
class EEGScore:
    """EEG 기반 서브-점수 분해."""
    alpha_beta: float   # alpha/beta 비율 기반 (높을수록 졸음)
    rel_theta: float    # 상대 theta 파워 기반
    blink_rate: float   # 눈 깜빡임 빈도 기반
    total: float        # 가중 합산 최종 점수 (0–1)


@dataclass
class FusionResult:
    """최종 융합 결과."""
    level: DrowsinessLevel
    final_score: float          # 0–1 (1에 가까울수록 졸음)
    camera_score: CameraScore
    eeg_score: Optional[EEGScore]
    eeg_available: bool
    confidence: float           # 0–1 (데이터 품질 기반 신뢰도)

    # 원시 카메라 상태 요약 (대시보드 표시용)
    camera_status: str = "NORMAL"

    def to_dict(self) -> dict:
        result = {
            "level": self.level.value,
            "final_score": round(self.final_score, 4),
            "confidence": round(self.confidence, 3),
            "eeg_available": self.eeg_available,
            "camera_status": self.camera_status,
            "camera_score": {
                "ear": round(self.camera_score.ear, 4),
                "perclos": round(self.camera_score.perclos, 4),
                "mar": round(self.camera_score.mar, 4),
                "pitch": round(self.camera_score.pitch, 4),
                "total": round(self.camera_score.total, 4),
            },
        }
        if self.eeg_score is not None:
            result["eeg_score"] = {
                "alpha_beta": round(self.eeg_score.alpha_beta, 4),
                "rel_theta": round(self.eeg_score.rel_theta, 4),
                "blink_rate": round(self.eeg_score.blink_rate, 4),
                "total": round(self.eeg_score.total, 4),
            }
        else:
            result["eeg_score"] = None
        return result


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------
class DrowsinessFusion:
    """
    카메라 점수와 EEG 점수를 가중 합산하여 최종 졸음 판정을 수행한다.

    확장 포인트:
      - _compute_camera_score() : 카메라 정규화 로직 조정
      - _compute_eeg_score()    : EEG 정규화 로직 조정
      - fuse()                  : 가중치 기반 융합 → ML 분류기로 교체 가능
    """

    def __init__(self, cfg: FusionConfig) -> None:
        self._cfg = cfg
        logger.info(
            f"DrowsinessFusion 초기화: "
            f"camera_weight={cfg.camera_weight}, eeg_weight={cfg.eeg_weight}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fuse(
        self, camera: CameraMetrics, eeg: EEGMetrics
    ) -> FusionResult:
        """
        Parameters
        ----------
        camera : 최신 카메라 메트릭
        eeg    : 최신 EEG 메트릭

        Returns
        -------
        FusionResult : 최종 졸음 판정 결과
        """
        cam_score = self._compute_camera_score(camera)

        eeg_available = eeg.is_connected and eeg.signal_quality >= 0.3
        eeg_score: Optional[EEGScore] = None

        if eeg_available:
            eeg_score = self._compute_eeg_score(eeg)
            final_score = (
                self._cfg.camera_weight * cam_score.total
                + self._cfg.eeg_weight * eeg_score.total
            )
            # 신뢰도: EEG 신호 품질과 카메라 캘리브레이션 여부 반영
            confidence = float(
                np.clip(eeg.signal_quality * (0.7 + 0.3 * camera.is_calibrated), 0.0, 1.0)
            )
        else:
            final_score = cam_score.total
            # 카메라만 사용 시 신뢰도 하향 (단일 모달)
            confidence = 0.55 if camera.is_calibrated else 0.30

        final_score = float(np.clip(final_score, 0.0, 1.0))
        level = self._score_to_level(final_score)

        if level.is_alert_needed:
            logger.warning(
                f"졸음 경보: level={level.value}, score={final_score:.3f}, "
                f"confidence={confidence:.2f}, eeg={eeg_available}"
            )

        return FusionResult(
            level=level,
            final_score=final_score,
            camera_score=cam_score,
            eeg_score=eeg_score,
            eeg_available=eeg_available,
            confidence=confidence,
            camera_status=camera.status,
        )

    # ------------------------------------------------------------------
    # Private: 카메라 정규화
    # ------------------------------------------------------------------
    def _compute_camera_score(self, m: CameraMetrics) -> CameraScore:
        """
        각 카메라 지표를 0–1 범위로 정규화.
        캘리브레이션 전에는 EAR 점수를 0으로 처리하여 오경보를 방지한다.
        """
        # EAR: 임계값 미만인 경우에만 점수 발생
        if not m.is_calibrated or m.threshold < 1e-6:
            ear_score = 0.0
        else:
            deficit = m.threshold - m.ear
            # deficit이 0보다 작으면(눈을 충분히 뜨고 있으면) 점수는 0
            # 눈을 완전히 감았을 때(ear=0) 점수가 1이 되도록 (deficit / threshold) 사용
            ear_score = float(np.clip(deficit / max(m.threshold, 1e-6), 0.0, 1.0))

        # PERCLOS: 0–100% → 0–1
        perclos_score = float(np.clip(m.perclos / 100.0, 0.0, 1.0))

        # MAR: 임계값 초과분을 정규화
        cfg = self._cfg
        if m.mar <= 0.6:  # mar_threshold 하드코딩 회피 — 여기서는 설정 직접 참조
            mar_score = 0.0
        else:
            mar_score = float(np.clip((m.mar - 0.6) / (1.0 - 0.6), 0.0, 1.0))

        # Pitch: 임계값(pitch_threshold) 초과분을 정규화
        # 기존에는 최대 임계값의 2배(50도)에서 포화되었으나,
        # pitch_danger_threshold(40도)를 도입하여, 임계값을 넘어서도 점진적으로 점수가 오르도록 완화함
        pitch_thr = cfg.pitch_threshold
        pitch_danger = cfg.pitch_danger_threshold
        if m.pitch <= pitch_thr:
            pitch_score = 0.0
        else:
            # (현재 각도 - 시작 임계각) / (위험 각도 - 시작 임계각)
            pitch_score = float(np.clip((m.pitch - pitch_thr) / max(pitch_danger - pitch_thr, 1e-6), 0.0, 1.0))

        total = (
            cfg.ear_weight * ear_score
            + cfg.perclos_weight * perclos_score
            + cfg.mar_weight * mar_score
            + cfg.pitch_weight * pitch_score
        )
        return CameraScore(
            ear=ear_score,
            perclos=perclos_score,
            mar=mar_score,
            pitch=pitch_score,
            total=float(np.clip(total, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Private: EEG 정규화
    # ------------------------------------------------------------------
    def _compute_eeg_score(self, m: EEGMetrics) -> EEGScore:
        """
        EEG 특징을 0–1 졸음 점수로 정규화.

        Alpha/Beta 비율:
          정상 기준선(baseline)에서 위험값(danger)까지 선형 정규화.
          baseline=1.5, danger=4.0 이면 비율 4.0 이상 → score = 1.0

        Relative Theta:
          정상 0.10–0.15, 졸음 0.35+ → 0.35 이상에서 포화

        Blink rate:
          극단값(매우 낮거나 매우 높음) 모두 이상으로 간주
        """
        cfg = self._cfg

        # Alpha/Beta 비율 점수
        baseline = cfg.eeg_alpha_beta_baseline
        danger = cfg.eeg_alpha_beta_danger
        alpha_beta_score = float(
            np.clip((m.alpha_beta_ratio - baseline) / max(danger - baseline, 1e-6), 0.0, 1.0)
        )

        # Relative Theta 점수 (0.12가 정상, 0.40이 위험)
        rel_theta_score = float(np.clip((m.relative_theta - 0.12) / 0.28, 0.0, 1.0))

        # Blink Rate 점수: 정상(15/min) 대비 이탈 정도
        # 매우 낮음(졸음성 서행) 또는 매우 높음(과각성) 모두 이상
        normal_blink = cfg.normal_blink_rate
        blink_deviation = abs(m.blink_rate - normal_blink) / max(normal_blink, 1.0)
        blink_score = float(np.clip(blink_deviation / 1.5, 0.0, 1.0))

        total = (
            cfg.alpha_beta_weight * alpha_beta_score
            + cfg.rel_theta_weight * rel_theta_score
            + cfg.blink_rate_weight * blink_score
        )
        return EEGScore(
            alpha_beta=alpha_beta_score,
            rel_theta=rel_theta_score,
            blink_rate=blink_score,
            total=float(np.clip(total, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Private: 점수 → 레벨
    # ------------------------------------------------------------------
    def _score_to_level(self, score: float) -> DrowsinessLevel:
        cfg = self._cfg
        if score >= cfg.threshold_drowsy:
            return DrowsinessLevel.DROWSY
        if score >= cfg.threshold_warning:
            return DrowsinessLevel.WARNING
        if score >= cfg.threshold_caution:
            return DrowsinessLevel.CAUTION
        return DrowsinessLevel.NORMAL
