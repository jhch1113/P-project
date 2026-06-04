# 배포 가이드 (Docker)

이 프로젝트는 **이미 전체가 Docker Compose로 패키징**되어 있습니다.  
다른 PC·Jetson에서 **같은 방식으로 재현**하려면: 저장소 복제 → 모델 파일 배치 → `.env` 작성 → `docker compose up`.

---

## 1. 배포 형태 이해

```text
┌─────────────────┐     ingest      ┌──────────────────┐
│ camera :8001    │ ──────────────► │ orchestrator     │
│ RealSense IR    │   0.5s POST     │ :8000            │
└─────────────────┘                 │ 대시보드·융합    │
┌─────────────────┐     ingest      │                  │
│ eeg (host net)  │ ──────────────► └────────▲─────────┘
│ Muse2 LSL+AI    │   1.0s POST              │
└─────────────────┘                        브라우저 :8000
```

| 컨테이너 | 역할 | 호스트 포트 |
|----------|------|-------------|
| **orchestrator** | 융합·웹 UI·`/ingest/*` | **8000** |
| **camera** | RealSense·비전 메트릭 | 8001 |
| **eeg** | Muse2·모델 추론 | 8002 (내부), LSL은 **host network** |

단일 프로세스로도 가능: `SERVICE_MODE=monolith` (개발용, 아래 §5).

---

## 2. 사전 요구사항

### 공통
- Docker Engine 24+ 및 **Docker Compose v2** (`docker compose`)
- Git으로 프로젝트 클론
- **Keras 모델 파일** `MUSE_activity_model.keras` (저장소에 없을 수 있음 → 별도 복사)

### Jetson Orin (권장, GPU EEG)
- JetPack 6.x (L4T r36.x)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — `runtime: nvidia`
- RealSense D435i (USB3)
- Muse2 + 호스트 BlueZ (`bluetooth` 서비스 실행)

### x86 / amd64 Linux (CPU EEG)
- Ubuntu 22.04+ 등 (glibc 호환)
- 동일 하드웨어; EEG 추론은 **CPU TensorFlow** (느릴 수 있음)
- Compose 파일: **`docker-compose.cpu.yml`**

### Windows / macOS
- RealSense·Muse BLE·privileged `/dev` 제약으로 **공식 타깃이 아님**. Linux 호스트(Jetson/PC) 권장.

---

## 3. 재현 가능 배포 절차 (표준)

### 3.1 저장소·모델

```bash
git clone https://github.com/jhch1113/P-project.git drowsiness_project
cd drowsiness_project
git checkout Concatenated-Jetson   # 또는 사용 중인 배포 브랜치

mkdir -p models
cp /path/to/MUSE_activity_model.keras models/   # Git에 없음 — 팀 공유 드라이브/Release에서 복사
```

### 3.2 환경 변수

```bash
cp .env.example .env
# MUSE_ADDRESS, MODELS_DIR, MUSE_MODEL_PATH 등 수정
nano .env
```

### 3.3 Muse2 블루투스 (호스트, 최초 1회)

```bash
bluetoothctl
# scan on → Muse MAC 확인
trust XX:XX:XX:XX:XX:XX
exit
```

`.env`의 `MUSE_ADDRESS`에 동일 MAC을 넣습니다.

### 3.4 빌드·기동

**Jetson (GPU):**

```bash
docker compose --env-file .env up --build -d
```

기본 `docker-compose.yml`은 EEG 이미지가 `Dockerfile.eeg.gpu`(L4T + TF GPU)입니다.  
모델 경로 기본값: `/app/pp_nrsc/MUSE_activity_model.keras` — 이미지에 모델을 bake-in 했다면 그대로, 아니면:

```bash
# compose에 이미 ./models:/models 마운트 있음 → .env 예:
# MUSE_MODEL_PATH=/models/MUSE_activity_model.keras
```

**x86 / CPU Linux:**

```bash
docker compose -f docker-compose.cpu.yml --env-file .env up --build -d
```

### 3.5 확인

```bash
docker compose ps
docker compose logs -f eeg    # Muse LSL · 모델 로드
curl -s http://localhost:8000/metrics/all | head -c 500
```

브라우저: `http://<호스트-IP>:8000`

---

## 4. Compose 파일 선택

| 파일 | 대상 | EEG 이미지 |
|------|------|------------|
| `docker-compose.yml` | **Jetson**, CUDA | `docker/Dockerfile.eeg.gpu` |
| `docker-compose.cpu.yml` | **PC / 서버**, CPU | `docker/Dockerfile.eeg.cpu` |

두 파일 모두 **3서비스 분리** 구조는 동일합니다. 재현성은 **같은 compose + 같은 .env + 같은 모델 파일**로 확보합니다.

---

## 5. 단일 컨테이너(monolith) — 개발·소규모

마이크로서비스 분리 없이 한 컨테이너에 넣을 수는 있지만, **Muse BLE(host network)·RealSense(/dev)** 때문에 운영에서는 3분할이 더 단순합니다.

로컬 개발(Compose 없이):

```bash
export SERVICE_MODE=monolith EEG_MODE=muse2
export MUSE_MODEL_PATH=./models/MUSE_activity_model.keras
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 6. 다른 머신으로 옮길 때 체크리스트

| 항목 | 확인 |
|------|------|
| `models/MUSE_activity_model.keras` 복사 | 필수 |
| `.env`의 `MUSE_ADDRESS` | 해당 머신에서 페어링한 MAC |
| Jetson vs PC | 맞는 compose 파일 사용 |
| `docker compose build --no-cache` | 의존성 꼬일 때 |
| RealSense | `lsusb`에 Intel 장치, camera 로그 |
| EEG | `docker compose logs eeg`에 LSL·GPU/CPU 메시지 |
| 방화벽 | LAN에서 **8000** 허용 |

이미지를 레지스트리로 배포하려면:

```bash
docker compose build
docker tag drowsiness_project-orchestrator:latest your-registry/drowsiness-orchestrator:2.0
docker push your-registry/drowsiness-orchestrator:2.0
# camera, eeg 동일 — 대상 머신에서 pull 후 compose의 build: 대신 image: 사용
```

---

## 7. 트러블슈팅

| 증상 | 조치 |
|------|------|
| EEG 연결 안 됨 | 호스트 `bluetoothctl`, `MUSE_ADDRESS`, eeg 로그의 muselsl |
| 모델 로드 실패 | `MUSE_MODEL_PATH`·볼륨 `./models`, keras/TF 버전(Jetson vs CPU 이미지 상이) |
| 카메라 없음 | USB3, `privileged`, `/dev` 마운트 |
| CAUTION 과다 | `docs/HYPERPARAMETERS.md`, baseline 30초 대기 |
| Too many open files (EEG) | compose의 `ulimits`·entrypoint muselsl 감독 확인 |

---

## 8. 관련 문서

- 사용자 개요: [README.md](../README.md)
- 커맨드: [COMMANDS.md](COMMANDS.md)
- 튜닝: [HYPERPARAMETERS.md](HYPERPARAMETERS.md)
- 개발 구조: [developer_guide.md](../developer_guide.md)
