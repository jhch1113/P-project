# 개발자 코드 구조 설계서 (Developer Guide)

본 문서는 프로젝트 유지보수 및 추가 개발을 담당하는 코딩 인원을 위한 상세 구조 설계서입니다. 전체 시스템의 통합 코드 흐름(Data Flow), 각 파일의 역할 및 동작 원리, Input/Output 구조를 명세합니다.

---

## 1. 통합 코드 흐름 (Integrated Data Flow)

시스템은 비동기 파이프라인 구조를 가집니다. 센서 데이터를 수집하는 엣지 환경의 특성상 메인 스레드가 블로킹되지 않도록 **백그라운드 데몬 스레드(Producer)와 FastAPI 라우터(Consumer)**가 스레드 안전 상태 객체를 공유하며 통신합니다.

### 1.1. Producer 1: 비전 처리 루프 (Vision Loop)
1. `pyrealsense2` IR 파이프라인(640×480, 기본 **15fps**, `CAM_FPS`)에서 프레임을 수집합니다. 처리 속도가 캡처보다 느릴 때를 대비해 **`poll_for_frames()` drain**으로 큐에 쌓인 오래된 프레임을 버리고 **항상 최신 프레임만** MediaPipe에 넣습니다(실시간 지연 누적 방지).
2. **감독(supervisor) 루프:** USB 단절·프레임 타임아웃 등으로 세션이 끊겨도 스레드를 종료하지 않고, 파이프라인을 정리한 뒤 **지수 백오프(2→10초)** 로 RealSense를 재연결합니다.
3. IR 레이저 에미터 **OFF**(도트 패턴이 FaceMesh 검출을 망가뜨림) → CLAHE → Gray→RGB → `DriverMonitorCV` → 랜드마크는 기본 **CONTOURS**만 그림(`DRAW_FULL_MESH=1` 시 전체 메쉬).
4. `CameraMetrics` + JPEG → `SharedCameraState`. Orchestrator의 재보정 POST는 camera:8001로 **포워딩**됩니다.
5. (분산) `MetricsPublisher`가 0.5초 주기로 `POST /ingest/camera` 전송.

### 1.2. Producer 2: 생체 신호 처리 루프 (EEG Loop)
1. 컨테이너 `entrypoint-eeg.sh`가 **`muselsl stream` 감독 루프**(BLE FD 누수 시 프로세스 재기동)를 띄운 뒤, 앱의 **`Muse2LSLProcessor` 감독 루프**가 LSL inlet을 유지합니다.
2. `pylsl`로 4채널(TP9, AF7, AF8, TP10)을 논블로킹 수집 → **6초** 링 버퍼(모델 5초 + filtfilt 여유). **5초 이상 무표본**이면 stale로 보고 inlet **재-resolve**(`EEG_LSL_STALE_TIMEOUT_SEC` 등).
3. 1.0초마다 **detrend→bandpass(1–40Hz)→notch(60Hz)** 후 Welch PSD(DSP) + **`MUSE_activity_model` P(졸음)**. 연결 후 **30초** median → `model_awake_baseline` → `model_drowsy_prob_adj`.
4. `EEGMetrics` → `SharedEEGState` → (분산) 1.0초 주기 `POST /ingest/eeg`.

### 1.3. Consumer: FastAPI 라우터 및 융합 엔진 (API & Fusion)
1. 클라이언트(대시보드 브라우저)가 `GET /metrics/all` 엔드포인트를 50ms 주기로 폴링합니다.
2. 라우터는 `SharedCameraState`와 `SharedEEGState`에서 최신 Metrics를 읽어옵니다.
3. 읽어온 두 Metrics를 `DrowsinessFusion.fuse()` 메서드에 주입하여 최종 졸음 판정 객체 `FusionResult`를 생성하고 이를 JSON으로 직렬화하여 응답합니다.

---

## 2. 핵심 모듈 상세 분석 (File-by-File Breakdown)

### 2.1. 진입점 및 환경 설정

#### `app/main.py` — 애플리케이션 진입점
* **역할:** FastAPI 애플리케이션 객체를 생성하고, `@asynccontextmanager` 기반 Lifespan으로 전체 생명주기를 관리합니다.
* **Input:** 환경 변수 `SERVICE_MODE` (monolith / camera / eeg / orchestrator), `EEG_MODE` (stub / muse2).
* **Output:** FastAPI `app` 인스턴스.
* **핵심 동작:** `SERVICE_MODE` 값에 따라 `VisionProcessingLoop`, `EEGProcessor`, `MetricsPublisher` 등 백그라운드 스레드를 조건부로 생성 및 시작합니다. CORS 미들웨어 설정, 라우터 마운트, PP-NRSC EEG 모듈(`/eeg` 경로) 마운트를 수행합니다.

#### `app/core/config.py` — 시스템 설정값 정의
* **역할:** 시스템 전체의 불변 설정값을 `@dataclass(frozen=True)` 기반으로 일원 관리합니다. 하드코딩된 매직 넘버를 완전히 제거하여 재현성과 실험 용이성을 보장합니다.
* **정의 클래스:**
  * `CameraConfig` — 해상도(640×480), FPS(기본 **15**, `CAM_FPS`), IR 채널 인덱스, JPEG 품질.
  * `DrowsinessConfig` — 캘리브레이션 시간(5초), EAR/MAR/Pitch 임계값, 이동 평균 버퍼 크기, EMA 가중치.
  * `EEGConfig` — 샘플링 레이트(256Hz), 주파수 대역 경계, Welch PSD 파라미터(nperseg=256, noverlap=128), 프론탈 채널 인덱스(AF7=1, AF8=2).
  * `FusionConfig` — 카메라/EEG 가중치(0.55:0.45), 서브-점수 내부 가중치, 졸음 판정 임계값(기본 CAUTION=0.32, WARNING=0.52, DROWSY=0.72, `FUSION_THRESHOLD_*`로 조정). 전체 표: `docs/HYPERPARAMETERS.md`.

---

### 2.2. 상태 관리 및 DTO (Data Transfer Object)

#### `app/core/state.py` — 스레드 간 공유 상태 컨테이너
* **역할:** 백그라운드 스레드(Producer)와 API 라우터(Consumer) 사이에서 안전한 데이터 교환을 보장합니다.
* **핵심 구조:**
  * `CameraMetrics` — 카메라 프레임 1장의 처리 결과를 담는 데이터 전달 객체(DTO).
  * `EEGMetrics` — EEG 특징 추출 결과를 담는 데이터 전달 객체(DTO).
  * `SharedCameraState` — `threading.Lock` + `threading.Event` 기반. MJPEG 스트리밍 시 `Event.wait()`를 통해 CPU 스핀 없이 새 프레임을 블로킹 대기합니다. 캘리브레이션 재설정 이벤트도 관리합니다.
  * `SharedEEGState` — `threading.Lock` 기반의 단순 읽기/쓰기 컨테이너.

---

### 2.3. 비전(Vision) 처리 파이프라인

#### `app/core/vision_loop.py` — RealSense 하드웨어 루프
* **역할:** RealSense(또는 `USE_WEBCAM=1` 웹캠) 캡처·처리·게시. **최상위 `_run()` 감독 루프**가 세션 단위로 `_run_realsense_session()`을 반복 호출하며, 예외 시 재연결합니다.
* **Input:** RealSense IR (Y8, 640×480, `CameraConfig.fps`).
* **Output:** JPEG + `CameraMetrics` → `SharedCameraState`.
* **지연·안정성 (필수 이해):**
  1. **프레임 drain:** `wait_for_frames()` 직후 `poll_for_frames()` 루프로 큐를 비워 **최신 IR만** 처리. MediaPipe가 15fps 캡처를 따라가지 못할 때 0.4s+ 지연 누적을 막습니다.
  2. **캡처 FPS:** 기본 15fps(`CAM_FPS`). 30fps는 CPU 큐 적체를 유발하기 쉬워 drain과 함께 보수적 기본값을 씁니다.
  3. **재연결:** 프레임 타임아웃(2s)·IR 연속 30회 누락 시 예외 → 파이프라인 stop → 백오프 후 재시작.
  4. **시각화 부하:** FaceMesh **CONTOURS** 기본(`DRAW_FULL_MESH=0`). TESSELATION 전체 메쉬는 CPU 부담이 큼.
  5. 에미터 OFF, RealSense intrinsics → `HeadPoseEstimator`, CLAHE, Gray→RGB, `DriverMonitorCV`, JPEG(`CAM_JPEG_QUALITY`).

#### `app/core/monitor.py` — 프레임 단위 졸음 지표 계산
* **역할:** 단일 RGB 프레임을 입력받아 모든 카메라 기반 졸음 지표를 산출하는 핵심 분석 모듈입니다.
* **Input:** `np.ndarray` (H, W, 3) uint8 형태의 RGB 이미지.
* **Output:** `CameraMetrics` 객체 + MediaPipe Landmark 리스트(시각화용).
* **핵심 동작:**
  1. MediaPipe FaceMesh 추론 (IR 최적화: confidence=0.3).
  2. `EARCalculator`, `MARCalculator`, `HeadPoseEstimator`를 호출하여 원시 수치 계산.
  3. Moving Average(이동 평균) 및 EMA(지수 이동 평균) 필터로 프레임 간 노이즈 제거.
  4. `CalibrationManager`를 통해 최초 5초간 개인화 EAR 임계값 및 Pitch 기준선을 자동 보정.
  5. 보정 완료 후 `PERCLOSCalculator`로 60초 창 기반 눈 감김 비율 계산.

#### `app/core/calculators.py` — 단일 책임 수치 계산기 모음
* **역할:** 단일 책임 원칙(SRP)에 따라 분리된 독립 계산기 클래스 모음입니다.
* **포함 클래스:**
  * `EARCalculator` — Eye Aspect Ratio (6점 랜드마크 기반 눈 개방비).
  * `MARCalculator` — Mouth Aspect Ratio (입 수직/수평 비율 기반 하품 감지).
  * `HeadPoseEstimator` — OpenCV `solvePnP` + `RQDecomp3x3` 기반 머리 Pitch/Yaw/Roll 추정. RealSense 실측 Intrinsics 적용.
  * `CalibrationManager` — 5초간 개인 EAR 임계값: `min(mean−k·σ, mean×factor)` 후 **`min(·, mean×0.75)`** 로 상한(뜬 눈보다 높은 threshold로 PERCLOS가 상시 포화되던 오류 방지). `ear_threshold_factor` 기본 **0.62**.
  * `PERCLOSCalculator` — 60초 슬라이딩 창, `smoothed_ear < ear_threshold` 비율(%). 캘리브 전에는 융합에서 EAR/PERCLOS 점수 0 처리.

---

### 2.4. 생체 신호(EEG) 처리 파이프라인

#### `app/api/eeg_processor.py` — Muse2 EEG 수신 및 특징 추출
* **역할:** LSL 네트워크 스트림 데이터를 수신하고 주파수 분석을 통해 졸음 관련 뇌파 특징을 추출합니다.
* **Input:** `pylsl.StreamInlet`으로부터 수신한 4채널 원시 전압값 (μV 단위, 256Hz 샘플링).
* **Output:** `EEGMetrics` 객체 → `SharedEEGState` 갱신.
* **포함 클래스:**
  * `Muse2LSLProcessor` — LSL 수신·6초 버퍼·1초 주기 특징 갱신. DSP(Welch 밴드파워, blink) + **`pp_nrsc.muse_inference_api`** 모델 추론·baseline 보정. LSL stale/유령 스트림 시 **자동 재-resolve** 감독 루프.
  * `StubEEGProcessor` — Muse2 미연결 시 사용되는 테스트용 더미 프로세서. 모든 EEG 값을 0으로 고정하고 `is_connected=False`를 유지합니다.
  * `extract_eeg_features()` — 다채널 EEG 원시 데이터에서 대역별 파워를 계산하는 유틸리티 함수.

---

### 2.5. 다중모달 융합 엔진 (Fusion Engine)

#### `app/api/fusion.py` — 최종 졸음 상태 판정
* **역할:** 분리되어 산출된 카메라 지표와 뇌파 지표를 통합하여 최종 졸음 점수를 산출하고 상태 레벨을 분류합니다.
* **Input:** `CameraMetrics` 객체, `EEGMetrics` 객체.
* **Output:** `FusionResult` 객체 (level, final_score, confidence, 서브-점수 분해 포함).
* **핵심 동작:**
  1. `_compute_camera_score()` — EAR, PERCLOS, MAR, Pitch 각각을 0.0~1.0 범위로 정규화한 뒤 가중 합산 (기본: EAR=0.30, PERCLOS=0.40, MAR=0.15, Pitch=0.15).
  2. `_compute_eeg_score()` — DSP 폴백 시 α/β·θ·blink 가중 합산(0.50/0.40/0.10). **`model_available`이면 `eeg_score.total = model_drowsy_prob_adj`**, `source="model"`.
  3. 최종 융합: `final = 0.55 × cam_score + 0.45 × eeg_score`. EEG 미연결 또는 `signal_quality < 0.3` 시 카메라 단독.
  4. 4단계: NORMAL < **0.32**, CAUTION **0.32~0.52**, WARNING **0.52~0.72**, DROWSY ≥ **0.72** (기본값, `FusionConfig`·`FUSION_THRESHOLD_*`).

---

### 2.6. API 라우터 및 대시보드

#### `app/api/routes.py` — REST API 엔드포인트 정의
* **역할:** 외부 클라이언트와의 통신 및 내부 노드 간 데이터 수집을 위한 REST API를 정의합니다.
* **주요 엔드포인트:**
  * `GET /` — 통합 웹 대시보드 HTML 응답.
  * `GET /metrics/all` — 카메라 + EEG + 융합 결과 통합 JSON 반환.
  * `GET /metrics/fusion` — 최종 융합 점수만 반환.
  * `GET /metrics/camera` — 카메라 원시 지표 반환.
  * `GET /metrics/eeg` — EEG 원시 지표 반환.
  * `GET /video_feed` — MJPEG 실시간 영상 스트리밍 (`yield` 제너레이터 기반).
  * `GET /debug/raw` — 진단용 원시 상태 덤프 (점수가 0인 이유를 자동 진단).
  * `POST /ingest/camera` — (Orchestrator 모드 전용) 카메라 메트릭 수신.
  * `POST /ingest/eeg` — (Orchestrator 모드 전용) EEG 메트릭 수신.
  * `POST /metrics/camera/reset_calibration` — 캘리브레이션 재시작 요청.

#### `app/api/dashboard.py` — 통합 웹 대시보드 UI
* **역할:** HTML/CSS/JavaScript로 구성된 단일 파일 대시보드를 생성합니다.
* **핵심 동작:** `/metrics/all`을 50ms 폴링. 카메라 지표·**AI 모델 P(졸음) 보정/원시/baseline**·DSP 진단 지표·최종 융합 판정 보드. 재보정 버튼은 Orchestrator에서 camera 서비스로 포워딩.

---

### 2.7. 분산 통신 모듈

#### `app/core/metrics_publisher.py` — 주기적 메트릭 전송기
* **역할:** 엣지 노드(camera/eeg 컨테이너)에서 산출된 메트릭을 오케스트레이터로 주기적 HTTP POST 전송하는 백그라운드 데몬 스레드입니다.
* **Input:** `payload_fn` 콜백 함수로부터 반환된 `dict` 데이터.
* **Output:** `urllib.request`를 통한 JSON POST 요청.
* **핵심 동작:** `threading.Event.wait(timeout=interval)`으로 주기를 제어하며, 네트워크 오류 시 예외를 잡아 로그만 남기고 다음 주기에 재시도합니다 (장애 전파 방지).

---

### 2.8. 운영·튜닝 환경변수 (요약)

| 변수 | 서비스 | 효과 |
|------|--------|------|
| `CAM_FPS`, `CAM_WIDTH`, `CAM_HEIGHT` | camera | 캡처·처리 부하·지연 |
| `DRAW_FULL_MESH` | camera | 1이면 무거운 FaceMesh 메쉬 |
| `USE_WEBCAM` | camera | RealSense 대신 웹캠 |
| `EEG_LSL_*` | eeg | LSL stale·재-resolve·프로브 |
| `MUSE_MODEL_PATH`, `EEG_MODEL_*` | eeg | 모델·baseline |
| `FUSION_THRESHOLD_*` | orchestrator | NORMAL/CAUTION/… 단계 |

전체: `docs/HYPERPARAMETERS.md`, `docker-compose.yml`.

---

### 2.9. PP-NRSC 모듈 (EEG 확장)

#### `pp_nrsc/` — 실시간 AI 추론 (프로덕션)
* **역할:** `Muse2LSLProcessor`가 호출하는 Keras 추론·전처리. FastAPI 서브앱(`/eeg`)으로도 마운트 가능.
* **활성 파일:** `muse_inference_api.py`, `advanced_preprocessing.py`, `advanced_postprocessing.py`
* **레거시:** 학습·구형 스코어러·CSV는 `deprecated/pp_nrsc/` (졸음 **단계** 판정은 `app/api/fusion.py` + `FusionConfig`가 담당).

---

## 3. 핵심 입출력(I/O) 데이터 명세서

개발 시 JSON 형태로 직렬화되어 오가는 최상위 규격입니다. 각 객체가 `to_dict()` 메서드를 호출하여 반환합니다.

### 3.1. CameraMetrics (카메라 출력 규격)
| 필드 | 타입 | 설명 | 예시값 |
|------|------|------|--------|
| ear | float | 눈 개방비 (높을수록 크게 뜸) | 0.284 |
| mar | float | 입 개방비 (0.6 초과 시 하품 판정) | 0.312 |
| perclos | float | 60초 창 내 눈 감김 비율 (%) | 15.4 |
| pitch | float | 고개 숙임 각도 (보정된 상대각, 25° 초과 위험) | 5.2 |
| status | string | 원시 판정 상태 | "NORMAL" |
| threshold | float | 개인화된 EAR 기준선 (캘리브레이션 결과) | 0.250 |
| is_calibrated | bool | 캘리브레이션 완료 여부 | true |

### 3.2. EEGMetrics (생체 신호 출력 규격)
| 필드 | 타입 | 설명 | 예시값 |
|------|------|------|--------|
| alpha_power | float | Alpha 대역 절대 전력 (μV²/Hz) | 12.4 |
| theta_power | float | Theta 대역 절대 전력 (μV²/Hz) | 8.1 |
| beta_power | float | Beta 대역 절대 전력 (μV²/Hz) | 5.2 |
| alpha_beta_ratio | float | Alpha/Beta 전력 비율 (정상≈1.5, 졸음≥4.0) | 2.38 |
| relative_theta | float | 전체 파워 대비 Theta 비율 (0~1) | 0.18 |
| blink_rate | float | 눈 깜빡임 빈도 (회/분, 정상 12~20) | 15.4 |
| model_drowsy_prob | float | AI P(졸음) 원시 0–1 | 0.52 |
| model_drowsy_prob_adj | float | baseline 보정 후(융합에 사용) | 0.10 |
| model_awake_baseline | float | awake baseline(0=미확정) | 0.50 |
| model_available | bool | 모델 추론 유효 여부 | true |
| is_connected | bool | Muse2 장치 연결 상태 | true |
| signal_quality | float | 신호 무결성 (0.0=불량, 1.0=양호) | 0.85 |

### 3.3. FusionResult (최종 융합 응답 규격)
| 필드 | 타입 | 설명 | 예시값 |
|------|------|------|--------|
| level | string | 졸음 상태 분류 (NORMAL/CAUTION/WARNING/DROWSY) | "CAUTION" |
| final_score | float | 통합 졸음 점수 (0.0~1.0) | 0.38 |
| confidence | float | 판정 신뢰도 (EEG 신호 품질 반영) | 0.85 |
| eeg_available | bool | EEG 데이터 반영 여부 | true |
| camera_status | string | 카메라 원시 상태 문자열 | "NORMAL" |
| camera_score | object | 카메라 서브-점수 분해 (ear, perclos, mar, pitch, total) | { "total": 0.15 } |
| eeg_score | object/null | 서브-점수 + `source`(`model`/`dsp`), `model_prob`, `model_prob_adj` | { "total": 0.10, "source": "model" } |
