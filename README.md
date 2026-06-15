# 다중모달 기반 실시간 졸음 감지 시스템 (Multi-modal Drowsiness Detection System)

```bash
# 환경이 구축된 현 Jetson Orin Nano에서 다음과 같이 실행
cd ~/drowsiness_project
docker compose --env-file .env up -d

# 코드/이미지를 바꾼 뒤 첫 기동
# docker compose --env-file .env up --build -d

# 실행 후 확인
docker compose --env-file .env ps
docker compose --env-file .env logs -f eeg 

# Muse가 안 붙을 때
# (이미 했으면 생략 가능)
# bluetoothctl (이미 했으면 생략 가능)
# trust 00:55:DA:B8:28:A3
# exit
docker compose --env-file .env restart eeg

# Muse 없이 카메라만
# .env에 EEG_MODE=stub으로 설정 -> Muse 없이도 up 가능
```
로 실행

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 적외선 카메라(Intel RealSense D435i)를 통한 시각적 특징 추출과 뇌파 측정 장비(Muse2)를 통한 생체 신호 분석을 결합한 다중모달(Multi-modal) 기반의 실시간 졸음 감지 시스템입니다. 연산 부하가 높은 영상 처리 및 신호 분석 로직을 분산 처리하기 위해, Jetson Orin Nano와 같은 엣지 디바이스 컴퓨팅 환경에 최적화된 마이크로서비스(Docker 컨테이너) 아키텍처를 설계 및 구현하였습니다.

## 2. 시스템 아키텍처 (System Architecture)
실시간 처리 성능의 보장 및 시스템의 모듈화(안정성)를 확보하기 위해, 시스템은 기능별로 독립된 3개의 도커(Docker) 컨테이너 노드로 분리되어 구동됩니다.

### 2.1. 비전 처리 노드 (Camera Container / Edge Node, 포트 8001)
* **주요 역할:** RealSense IR → MediaPipe FaceMesh → EAR/MAR/PERCLOS/Pitch 산출. CPU 부하를 줄이기 위해 기본 **15fps**(`CAM_FPS`)·가벼운 랜드마크 드로잉을 사용합니다.
* **지연 방지:** 처리가 캡처보다 느릴 때 큐에 쌓인 프레임을 **`poll_for_frames` drain**으로 버리고 최신 프레임만 처리합니다. USB 단절 시 **자동 재연결**(지수 백오프).
* **추출 지표:** EAR, MAR, PERCLOS(개인 EAR 임계값 캘리브 5초), Head Pitch 등.
* **통신:** 0.5초 주기 `POST /ingest/camera`. MJPEG는 오케스트레이터가 `http://camera:8001/video_feed`를 프록시합니다.

### 2.2. 생체 신호 처리 노드 (EEG Container / Edge Node, 호스트 네트워크 · 포트 8002)
* **주요 역할:** Muse2 → LSL 수신 후 **밴드패스·노치 전처리 → Welch PSD**로 DSP 특징을 추출하고, **Keras 모델(`MUSE_activity_model`)**로 AF7/AF8 5초 윈도우 기반 **P(졸음)** 을 실시간 추론합니다.
* **추출 지표:** `model_drowsy_prob`(원시), `model_drowsy_prob_adj`(awake baseline 보정), Alpha/Beta·relative theta·blink rate(DSP, **진단용**), 신호 품질 등.
* **통신 프로토콜:** 산출된 지표는 1.0초 주기로 오케스트레이터 `POST /ingest/eeg`로 전송합니다. LSL 단절 시 자동 재-resolve(`EEG_LSL_*` 환경변수).

### 2.3. 통합 제어 및 분석 노드 (Orchestrator Container / Central Node)
* **주요 역할:** 비전 처리 노드와 생체 신호 처리 노드로부터 비동기적으로 수집된 데이터를 통합하여 최종 졸음 상태를 판정하고, 시각화된 모니터링 환경을 제공하는 중앙 집중형 서버입니다.
* **점수 산출 알고리즘 (Score Calculation Logic):**
  1. **카메라:** EAR·PERCLOS·MAR·Pitch를 0–1로 정규화 후 가중 합산(`FusionConfig`: 예) EAR 0.30, PERCLOS 0.40 …).
  2. **EEG:** 모델 추론이 유효하면 **`model_drowsy_prob_adj`가 EEG 점수 전체**를 대체합니다. 없으면 DSP(α/β, θ, blink) 가중 합산으로 폴백합니다.
  3. **융합:** `final = 0.55 × camera + 0.45 × eeg`(EEG 품질 ≥ 0.3·연결 시). 미연결 시 카메라만 사용.
  4. **4단계:** 기본 threshold — CAUTION ≥ **0.32**, WARNING ≥ **0.52**, DROWSY ≥ **0.72** (`FUSION_THRESHOLD_*`로 조정, `docs/HYPERPARAMETERS.md` 참고).
* **서비스 제공:** 최종 융합 결과 및 실시간 모니터링 UI를 웹 대시보드 형태로 제공하며, 외부 시스템과의 연동을 위한 RESTful API를 노출합니다.

---

## 3. 실행 및 검증 방법 (Execution & Verification)
본 시스템은 Jetson Orin Nano와 같은 엣지 컴퓨팅 환경에서 하드웨어 장치를 물리적으로 연결한 후 아래의 절차에 따라 구동됩니다.

### 3.1. 하드웨어 연결 및 환경 준비
1. Intel RealSense D435i 카메라를 타겟 보드(Jetson Orin Nano)의 USB 3.0 포트에 연결합니다.
2. Muse2 뇌파 측정 기기의 전원을 인가하여 블루투스 페어링 대기 상태를 유지합니다.
3. 동일 네트워크 대역 내에 위치한 호스트 PC를 활용하여 타겟 보드의 터미널(SSH 등)에 접속합니다.

### 3.2. Muse2 블루투스 페어링 (호스트, 최초 1회)
LSL 스트리밍 자체는 EEG 컨테이너 내부에서 수행되지만, 블루투스 장치의 신뢰(trust) 등록은 호스트의 BlueZ 데몬에서 1회 수행해야 합니다. 컨테이너는 호스트의 D-Bus 소켓을 공유하여 이 페어링 정보를 그대로 사용합니다.
```bash
bluetoothctl
# (bluetoothctl 내부)
scan on                          # "Muse-XXXX" 장치의 MAC 주소 확인
trust  00:55:DA:XX:XX:XX         # 예시 MAC — 실제 장치 주소로 대체
exit
```
*(참고: Muse2는 BLE 기기로 전통적 PIN 페어링(`pair`)이 `AuthenticationFailed`로 끝나는 것이 정상이며, `trust` 등록만으로 충분합니다.)*

### 3.3. 분산 시스템 컨테이너 구동 (Docker 배포)

**전체 시스템은 이미 3개 컨테이너(orchestrator / camera / eeg)로 패키징되어 있습니다.** 다른 환경에서 재현하려면 모델 파일·`.env`만 맞춘 뒤 Compose를 실행하면 됩니다. 상세: **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)**.

```bash
cd drowsiness_project
cp .env.example .env          # MUSE_ADDRESS, 모델 경로 수정
mkdir -p models && cp /path/to/MUSE_activity_model.keras models/

# Jetson (GPU EEG)
docker compose --env-file .env up --build -d

# x86 Linux PC (CPU EEG)
docker compose -f docker-compose.cpu.yml --env-file .env up --build -d
```

EEG 컨테이너는 기동 시 **`muselsl stream`을 자동 실행**(`docker/entrypoint-eeg.sh`)하고, 동일 컨테이너의 `pylsl`이 LSL을 수신합니다.
*(참고 1: EEG 컨테이너는 LSL 멀티캐스트 수신 및 호스트 BlueZ 접근을 위해 호스트 네트워크 모드(`network_mode: "host"`)와 D-Bus 소켓 공유로 구성됩니다.)*
*(참고 2: 기본 모델 경로는 컨테이너 내 `/models/MUSE_activity_model.keras`(호스트 `./models` 마운트)입니다. AF7/AF8 5초(1280×2) → P(졸음), EEG 졸음 판정 주도(DSP는 진단용). `MUSE_MODEL_PATH`는 **컨테이너 내부 경로** 기준.)*
*(참고 3: Muse2 MAC은 **`.env`의 `MUSE_ADDRESS`에 필수**입니다(`cp .env.example .env` 후 `bluetoothctl`로 확인한 주소 입력).)*
*(참고 4: **Jetson GPU 추론** — EEG 서비스는 `docker/Dockerfile.eeg.gpu`(L4T JetPack + NVIDIA TensorFlow GPU 휠)로 빌드되며 `runtime: nvidia`가 필요합니다. 로그에 `GPU devices=[PhysicalDevice(name='/physical_device:GPU:0'...)]`가 보이면 CUDA 가속이 활성화된 것입니다. x86/CPU 전용 환경에서는 `deprecated/docker/Dockerfile.eeg`를 `docker/`로 복사해 사용하십시오.)*
*(참고 5: **LSL 자동 재연결** — `muselsl`이 재시작되어도 EEG 앱이 stale inlet에 묶이지 않도록, 표본 단절(기본 5초) 또는 유령 스트림(프로브 실패) 시 LSL을 자동 재-resolve합니다. 조정: `EEG_LSL_STALE_TIMEOUT_SEC`, `EEG_LSL_PROBE_TIMEOUT_SEC`, `EEG_LSL_RESOLVE_RETRY_SEC`.)*

EEG 컨테이너의 Muse2 연결 상태는 다음으로 실시간 확인합니다.
```bash
docker compose logs -f eeg
```

### 3.4. 통합 모니터링 시스템 접속
모든 컨테이너 노드가 정상적으로 구동된 후, 동일 네트워크 내 클라이언트 기기(PC/모바일 등)의 웹 브라우저를 통해 통합 대시보드에 접근하여 결과를 검증합니다.
* **접속 주소:** `http://<Target-IP-Address>:8000` (예: `http://192.168.0.15:8000`)
* **출력 정보:** 실시간 적외선(IR) 영상, 개별 원시 지표 수치 통계, 그리고 융합된 최종 졸음 점수 범례 및 시각적 경고 상태.

---

## 4. REST API 명세 (API Specifications)
오케스트레이터 서버(Port: 8000)는 외부 시스템 연동 및 내부 노드 간 데이터 파이프라인 구성을 위해 다음의 API를 제공합니다.

### 4.1. 외부 시스템 연동 및 데이터 조회 (GET)
* `/` : 실시간 모니터링을 위한 통합 웹 대시보드 HTML 응답
* `/metrics/all` : 카메라 지표, EEG 지표, 최종 융합 점수를 포괄하는 통합 JSON 페이로드 반환
* `/metrics/fusion` : 최종 산출된 융합 점수(Final Score) 및 현재 위협 수준(Level) 단일 반환
* `/metrics/camera` : 비전 노드로부터 갱신된 최근 원시 지표 반환
* `/metrics/eeg` : 생체 신호 노드로부터 갱신된 최근 원시 뇌파 지표 반환
* `/video_feed` : MJPEG 규격에 기반한 실시간 적외선(IR) 카메라 영상 스트림 제공

### 4.2. 내부·제어 (POST)
* `/ingest/camera` : (Orchestrator 모드) 카메라 메트릭 수집
* `/ingest/eeg` : (Orchestrator 모드) EEG 메트릭 수집
* `/metrics/camera/reset_calibration` : 캘리브레이션 재시작(Orchestrator는 camera:8001로 포워딩)

### 4.3. 진단 (GET)
* `/debug/raw` : 카메라·EEG·융합 서브점수 원시 덤프

### 4.4. EEG 실험 API (선택, `SERVICE_MODE=eeg` 또는 `monolith`일 때만)
오케스트레이터(8000)에는 마운트되지 않습니다. EEG 컨테이너 **8002**에서 `/eeg/health`, `/eeg/session/*` 등(`pp_nrsc/muse_inference_api`). 상세: `docs/COMMANDS.md`.

---

## 5. 프로젝트 디렉토리 구조 (Directory Structure)
```text
app/          # 통합 융합 엔진, REST API, 대시보드 (하이퍼파라미터: app/core/config.py)
pp_nrsc/      # 실시간 AI 추론(muse_inference_api) + 전처리 모듈만 유지
docker/       # 마이크로서비스 Dockerfile (EEG GPU: Dockerfile.eeg.gpu)
models/       # (선택) 호스트 마운트 ./models → 컨테이너 /models. 기본 모델 경로는 pp_nrsc/ 또는 MUSE_MODEL_PATH
docs/         # DEPLOYMENT.md(배포), COMMANDS.md, HYPERPARAMETERS.md
developer_guide.md  # 개발자용 모듈·데이터 흐름 상세
deprecated/   # 미사용 학습/검증 스크립트, 구형 스코어러, 샘플 CSV, 레거시 문서
```

**튜닝:** `app/core/config.py` + `docker-compose.yml` 환경변수. 표·설명: [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md). 실행 커맨드: [`docs/COMMANDS.md`](docs/COMMANDS.md).