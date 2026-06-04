# Deprecated (미사용·레거시)

실시간 졸음 파이프라인(`app/` + `pp_nrsc/muse_inference_api.py`)에서 **더 이상 쓰지 않는** 스크립트·실험 코드·중복 문서를 보관합니다.

## 현재 프로덕션에서 쓰는 설정 파일

하이퍼파라미터(가중치·threshold)는 **`app/core/config.py`** 한 곳이 기준입니다.

| 클래스 | 용도 |
|--------|------|
| `FusionConfig` | 카메라/EEG 융합 가중치, NORMAL/CAUTION/WARNING/DROWSY 임계값, 모델 baseline |
| `DrowsinessConfig` | EAR/MAR/Pitch/PERCLOS, 캘리브레이션 |
| `EEGConfig` | LSL, 밴드패스·노치, DSP 특징 주기 |
| `CameraConfig` | 해상도·FPS |

환경변수로 덮어쓰기: `docker-compose.yml` → `FUSION_THRESHOLD_*`, `EEG_MODEL_AWAKE_BASELINE` 등.

점수 **산출 로직**(정규화·합산)은 `app/api/fusion.py`, 카메라는 `app/core/calculators.py` / `monitor.py`.

모델 전처리 상수(FS, SEQ_LEN, 히스테리시스)만 `pp_nrsc/muse_inference_api.py` 상단에 있습니다(융합 threshold와 별개).

## 이 폴더 구성

- `pp_nrsc/` — 구형 단독 스코어러·학습/검증 스크립트·샘플 CSV
- `scripts/` — Muse 스캔·수동 EEG 테스트
- `docker/` — CPU 전용 구형 EEG Dockerfile
- 루트 — `test_simulation.py`, Notion용 중복 README/가이드

`muse_inference_api`의 `/ws/live-muse` 데모만 `deprecated/pp_nrsc/eeg_data_source.py`, `drowsiness_scorer.py`를 참조합니다.

**주의:** `README_notion_import.md`, `developer_guide_notion_import.md` 등은 과거 스냅샷이며, 최신 정보는 저장소 루트 `README.md`·`developer_guide.md`·`docs/`를 따릅니다.
