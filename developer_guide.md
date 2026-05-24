# 개발자 코드 구조 설계서 (Developer Guide)

본 문서는 프로젝트 유지보수 및 추가 개발을 담당하는 코딩 인원을 위한 상세 구조 설계서입니다. 전체 시스템의 통합 코드 흐름(Data Flow), 각 파일의 역할 및 동작 원리, Input/Output 구조를 명세합니다.

---

## 1. 통합 코드 흐름 (Integrated Data Flow)

시스템은 비동기 파이프라인 구조를 가집니다. 센서 데이터를 수집하는 엣지 환경의 특성상 메인 스레드가 블로킹되지 않도록 **백그라운드 데몬 스레드(Producer)와 FastAPI 라우터(Consumer)**가 스레드 안전 상태 객체를 공유하며 통신합니다.

### 1.1. Producer 1: 비전 처리 루프 (Vision Loop)
1. `pyrealsense2` 파이프라인에서 적외선(IR) 영상 프레임을 수집합니다 (640x480, 30fps).
2. IR 이미지에 CLAHE(국소 히스토그램 평활화)를 적용하여 명암비를 극대화한 뒤, 1채널(Grayscale)을 3채널(RGB)로 변환합니다.
3. 변환된 프레임을 `DriverMonitorCV` 모듈에 전달하여 MediaPipe FaceMesh 기반의 랜드마크 추출 및 EAR/MAR/Pitch 수치를 계산합니다.
4. 산출된 `CameraMetrics` 객체와 JPEG 인코딩된 프레임을 스레드 안전 컨테이너 `SharedCameraState`에 업데이트합니다.
5. (분산 환경 시) `MetricsPublisher`가 0.5초 주기로 Orchestrator의 API(`POST /ingest/camera`)로 데이터를 전송합니다.

### 1.2. Producer 2: 생체 신호 처리 루프 (EEG Loop)
1. `pylsl` 라이브러리를 통해 Muse2 블루투스 LSL 스트림에서 4채널(TP9, AF7, AF8, TP10) 원시 전압 데이터를 논블로킹으로 수집합니다.
2. 수집된 데이터를 2.0초 크기의 링 버퍼(Ring Buffer)에 누적합니다.
3. 설정된 업데이트 주기(기본 1.0초)마다 Scipy의 Welch 방법을 사용하여 전두엽 채널(AF7, AF8)의 Alpha/Beta/Theta 대역 파워 스펙트럼을 계산합니다.
4. 산출된 `EEGMetrics` 객체를 스레드 안전 컨테이너 `SharedEEGState`에 업데이트합니다.
5. (분산 환경 시) `MetricsPublisher`가 1.0초 주기로 Orchestrator의 API(`POST /ingest/eeg`)로 데이터를 전송합니다.

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
  * `CameraConfig` — 해상도(640x480), FPS(30), IR 채널 인덱스, JPEG 품질.
  * `DrowsinessConfig` — 캘리브레이션 시간(5초), EAR/MAR/Pitch 임계값, 이동 평균 버퍼 크기, EMA 가중치.
  * `EEGConfig` — 샘플링 레이트(256Hz), 주파수 대역 경계, Welch PSD 파라미터(nperseg=256, noverlap=128), 프론탈 채널 인덱스(AF7=1, AF8=2).
  * `FusionConfig` — 카메라/EEG 가중치(0.55:0.45), 서브-점수 내부 가중치, 졸음 판정 임계값(CAUTION=0.25, WARNING=0.45, DROWSY=0.65).

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
* **역할:** RealSense 카메라 파이프라인의 초기화, 프레임 수집, 종료를 관리하는 무한 루프 스레드입니다.
* **Input:** RealSense IR 프레임 (Y8 포맷, 640x480).
* **Output:** JPEG 인코딩된 시각화 프레임 + `CameraMetrics` → `SharedCameraState` 갱신.
* **핵심 동작:**
  1. RealSense 파이프라인 시작 시 **IR 레이저 에미터를 비활성화**합니다 (Structured Light 도트 패턴이 얼굴에 맺히면 MediaPipe 검출 실패를 유발하므로).
  2. 실제 카메라 내부 파라미터(Intrinsics)를 RealSense SDK로부터 추출하여 HeadPoseEstimator에 적용합니다.
  3. CLAHE → Grayscale→RGB 변환 → `DriverMonitorCV.process_frame()` 호출 → MediaPipe Landmark 오버레이 → JPEG 인코딩 순서로 수행합니다.

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
  * `CalibrationManager` — 최초 5초간 EAR 평균·표준편차 기반 개인화 임계값 산출 (mean - k*σ 방식).
  * `PERCLOSCalculator` — NHTSA 기준 Sliding Window (60초) 기반 눈 감김 비율(%) 실시간 계산.

---

### 2.4. 생체 신호(EEG) 처리 파이프라인

#### `app/api/eeg_processor.py` — Muse2 EEG 수신 및 특징 추출
* **역할:** LSL 네트워크 스트림 데이터를 수신하고 주파수 분석을 통해 졸음 관련 뇌파 특징을 추출합니다.
* **Input:** `pylsl.StreamInlet`으로부터 수신한 4채널 원시 전압값 (μV 단위, 256Hz 샘플링).
* **Output:** `EEGMetrics` 객체 → `SharedEEGState` 갱신.
* **포함 클래스:**
  * `Muse2LSLProcessor` — 실제 Muse2 연동 구현체. 논블로킹 `pull_chunk()`로 데이터를 수집하여 링 버퍼에 쌓고, Welch 방법으로 Alpha(8-12Hz)/Beta(13-30Hz)/Theta(4-8Hz) 대역 파워를 계산합니다. 눈 깜빡임 빈도는 전두엽 채널의 피크 진폭 기반으로 추정합니다.
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
  2. `_compute_eeg_score()` — Alpha/Beta 비율, Relative Theta, Blink Rate를 0.0~1.0 범위로 정규화한 뒤 가중 합산 (기본: α/β=0.50, θ=0.40, Blink=0.10).
  3. 최종 융합: `final = 0.55 × cam_score + 0.45 × eeg_score`. EEG 미연결 또는 신호 품질 0.3 미만 시 카메라 단독 모드로 자동 전환.
  4. 졸음 상태 4단계 분류: NORMAL(<0.25), CAUTION(0.25~0.45), WARNING(0.45~0.65), DROWSY(≥0.65).

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
* **핵심 동작:** JavaScript `setInterval`로 `/metrics/all`을 50ms 주기로 폴링하여, 카메라 4개 지표 카드 + EEG 4개 지표 카드 + 최종 판정 보드를 실시간 갱신합니다. 캘리브레이션 진행 바, 재보정 버튼, 연결 상태 배지 등의 인터랙션을 포함합니다.

---

### 2.7. 분산 통신 모듈

#### `app/core/metrics_publisher.py` — 주기적 메트릭 전송기
* **역할:** 엣지 노드(camera/eeg 컨테이너)에서 산출된 메트릭을 오케스트레이터로 주기적 HTTP POST 전송하는 백그라운드 데몬 스레드입니다.
* **Input:** `payload_fn` 콜백 함수로부터 반환된 `dict` 데이터.
* **Output:** `urllib.request`를 통한 JSON POST 요청.
* **핵심 동작:** `threading.Event.wait(timeout=interval)`으로 주기를 제어하며, 네트워크 오류 시 예외를 잡아 로그만 남기고 다음 주기에 재시도합니다 (장애 전파 방지).

---

### 2.8. PP-NRSC 모듈 (EEG 확장)

#### `pp_nrsc/` — EEG 추론 및 기계학습 모델 실험 모듈
* **역할:** Keras 기반 사전 학습 모델을 활용한 뇌파 졸음 분류 실험 및 검증 모듈입니다. FastAPI 서브앱으로 `/eeg` 경로에 마운트됩니다.
* **주요 파일:**
  * `muse_inference_api.py` — `/eeg/health`, `/eeg/monitor`, `/eeg/session/*` 등 EEG 세션 관리 API.
  * `drowsiness_scorer.py` — EEG 기반 졸음 점수 산출 로직.
  * `train_muse_model_v2.py` — 모델 학습 스크립트.
  * `*.csv` — 학습/검증에 사용된 실험 데이터셋.

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
| eeg_score | object/null | EEG 서브-점수 분해 (alpha_beta, rel_theta, blink_rate, total) | { "total": 0.61 } |
