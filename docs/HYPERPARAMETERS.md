# 하이퍼파라미터 위치

## 1. 메인 설정: `app/core/config.py`

실시간 점수·단계 판정에 쓰는 값은 여기 dataclass에 정의합니다.

### `FusionConfig` — 최종 졸음 점수·단계

- `camera_weight` / `eeg_weight` (기본 0.55 / 0.45)
- 카메라 서브: `ear_weight`, `perclos_weight`, `mar_weight`, `pitch_weight`
- EEG DSP 폴백 서브: `alpha_beta_weight`, `rel_theta_weight`, `blink_rate_weight`
- 단계 threshold: `threshold_caution`, `threshold_warning`, `threshold_drowsy`
- 모델 baseline: `model_awake_baseline_default`, `model_baseline_calib_sec`
- EEG DSP 정규화: `eeg_alpha_beta_baseline`, `normal_blink_rate` 등

환경변수: `FUSION_THRESHOLD_CAUTION`, `FUSION_THRESHOLD_WARNING`, `FUSION_THRESHOLD_DROWSY`, `EEG_MODEL_AWAKE_BASELINE`, `EEG_MODEL_BASELINE_CALIB_SEC`

### `DrowsinessConfig` — 카메라 비전

- EAR 캘리브: `ear_threshold_factor`, `ear_threshold_std_k`, …
- `mar_threshold`, `pitch_threshold`, `perclos_danger_pct`, 버퍼·EMA

### `EEGConfig` — Muse DSP

- 샘플레이트, 밴드 경계, `bandpass_low/high`, `notch_freq`, `feature_update_interval`

### `CameraConfig`

- `CAM_WIDTH`, `CAM_HEIGHT`, `CAM_FPS` (환경변수, FPS 기본 **15** — 비전 처리·큐 지연과 trade-off)
- `CAM_JPEG_QUALITY` — `/video_feed` MJPEG 품질
- `USE_WEBCAM` — RealSense 대신 웹캠 (`1`/`0`)

**비전 런타임(코드/환경):** `DRAW_FULL_MESH`(가벼운 CONTOURS vs 전체 메쉬). 지연 제거는 `app/core/vision_loop.py`의 **프레임 drain** + 적정 `CAM_FPS`.

## 2. 산출 로직 (값이 아닌 공식)

| 파일 | 역할 |
|------|------|
| `app/api/fusion.py` | 서브점수 0–1 정규화, 가중 합산, 단계 매핑 |
| `app/api/eeg_processor.py` | 모델 baseline 수집, `adjust_model_drowsy_prob` 적용 |
| `app/core/calculators.py` | EAR/PERCLOS/MAR/Pitch 계산 |
| `app/core/monitor.py` | 비전 파이프라인 오케스트레이션 |

## 3. AI 모델 전용 (융합 threshold와 분리)

`pp_nrsc/muse_inference_api.py` 상단:

- `FS`, `SEQ_LEN`, `STRIDE`, `HYS_HIGH_DEFAULT`, `HYS_LOW_DEFAULT`
- `MUSE_MODEL_PATH` 환경변수

## 4. 배포 시 오버라이드 (`docker-compose.yml`)

| 환경변수 | 적용 서비스 | 설명 |
|----------|-------------|------|
| `FUSION_THRESHOLD_*` | orchestrator | 최종 점수 단계 threshold |
| `EEG_MODEL_AWAKE_BASELINE`, `EEG_MODEL_BASELINE_CALIB_SEC` | eeg | 모델 awake baseline |
| `MUSE_MODEL_PATH`, `MUSE_ADDRESS` | eeg | 모델 파일·BLE MAC |
| `EEG_LSL_*` | eeg (코드 기본) | LSL 재연결 타이밍 |
| `CAM_FPS` 등 | camera (이미지 빌드 시 env) | 캡처 FPS |

## 5. 레거시 (사용 안 함)

구형 스코어·학습 설정: `deprecated/pp_nrsc/config.py`, `config.yaml`, `drowsiness_scorer.py`
