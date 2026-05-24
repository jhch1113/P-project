# 다중모달 기반 실시간 졸음 감지 시스템 (Multi-modal Drowsiness Detection System)

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 적외선 카메라(Intel RealSense D435i)를 통한 시각적 특징 추출과 뇌파 측정 장비(Muse2)를 통한 생체 신호 분석을 결합한 다중모달(Multi-modal) 기반의 실시간 졸음 감지 시스템입니다. 연산 부하가 높은 영상 처리 및 신호 분석 로직을 분산 처리하기 위해, Jetson Orin Nano와 같은 엣지 디바이스 컴퓨팅 환경에 최적화된 마이크로서비스(Docker 컨테이너) 아키텍처를 설계 및 구현하였습니다.

## 2. 시스템 아키텍처 (System Architecture)
실시간 처리 성능의 보장 및 시스템의 모듈화(안정성)를 확보하기 위해, 시스템은 기능별로 독립된 3개의 도커(Docker) 컨테이너 노드로 분리되어 구동됩니다.

### 2.1. 비전 처리 노드 (Camera Container / Edge Node)
* **주요 역할:** RealSense IR 카메라로부터 획득한 실시간 영상 데이터를 기반으로 딥러닝 기반 안면 랜드마크를 추출 및 분석합니다.
* **추출 지표:** 눈 깜빡임(EAR: Eye Aspect Ratio), 하품(MAR: Mouth Aspect Ratio), 눈 감김 빈도(PERCLOS), 머리 기울기(Head Pitch) 등의 시각적 원시 지표(Raw Metrics)를 산출합니다.
* **통신 프로토콜:** 산출된 지표 데이터는 0.5초 주기로 중앙 오케스트레이터(Orchestrator)의 REST API로 전송(Push)됩니다.

### 2.2. 생체 신호 처리 노드 (EEG Container / Edge Node)
* **주요 역할:** Muse2 장치로부터 블루투스 기반 LSL(Lab Streaming Layer) 프로토콜을 통해 뇌파 스트림 데이터를 수신하고, 노이즈 필터링 및 고속 푸리에 변환(FFT)을 통한 주파수 스펙트럼 분석을 수행합니다.
* **추출 지표:** 전두엽 채널 기반의 Alpha/Beta 대역 전력 비율(Power Ratio), 상대적 Theta파 전력, 그리고 안구 전도(EOG) 기반의 눈 깜빡임(Blink) 빈도 등 뇌파 원시 지표를 산출합니다.
* **통신 프로토콜:** 산출된 지표 데이터는 1.0초 주기로 중앙 오케스트레이터(Orchestrator)의 REST API로 전송(Push)됩니다.

### 2.3. 통합 제어 및 분석 노드 (Orchestrator Container / Central Node)
* **주요 역할:** 비전 처리 노드와 생체 신호 처리 노드로부터 비동기적으로 수집된 데이터를 통합하여 최종 졸음 상태를 판정하고, 시각화된 모니터링 환경을 제공하는 중앙 집중형 서버입니다.
* **점수 산출 알고리즘 (Score Calculation Logic):**
  1. **개별 정규화:** 수신된 카메라 지표와 뇌파 지표를 각각 독립적인 수리 모델로 정규화하여, 카메라 기반 졸음 점수(0.0~1.0)와 뇌파 기반 졸음 점수(0.0~1.0)를 산출합니다.
  2. **다중모달 융합 (Multi-modal Fusion):** 각 모달리티(Modality)의 신뢰도 지표 및 사전에 정의된 알고리즘 가중치를 반영하여 합산함으로써 최종 졸음 판정 점수(Final Score)를 산출합니다. 판정 점수에 따라 시스템 상태를 4단계(NORMAL, CAUTION, WARNING, DROWSY)로 분류합니다.
* **서비스 제공:** 최종 융합 결과 및 실시간 모니터링 UI를 웹 대시보드 형태로 제공하며, 외부 시스템과의 연동을 위한 RESTful API를 노출합니다.

---

## 3. 실행 및 검증 방법 (Execution & Verification)
본 시스템은 Jetson Orin Nano와 같은 엣지 컴퓨팅 환경에서 하드웨어 장치를 물리적으로 연결한 후 아래의 절차에 따라 구동됩니다.

### 3.1. 하드웨어 연결 및 환경 준비
1. Intel RealSense D435i 카메라를 타겟 보드(Jetson Orin Nano)의 USB 3.0 포트에 연결합니다.
2. Muse2 뇌파 측정 기기의 전원을 인가하여 블루투스 페어링 대기 상태를 유지합니다.
3. 동일 네트워크 대역 내에 위치한 호스트 PC를 활용하여 타겟 보드의 터미널(SSH 등)에 접속합니다.

### 3.2. 생체 신호 스트리밍 프로세스 가동
타겟 보드(Jetson)의 터미널 환경에서 Muse2 장치와 블루투스 세션을 수립하고, LSL 스트림 브로드캐스팅을 개시합니다.
```bash
cd /home/neuro/drowsiness_project
pip install muselsl
muselsl stream
```
*(주의: 본 스트리밍 프로세스는 데이터 연속성 확보를 위해 백그라운드 환경에서 지속적으로 유지되어야 합니다.)*

### 3.3. 분산 시스템 컨테이너 구동
신규 세션(터미널)을 생성한 후, Docker Compose를 활용하여 분산 컨테이너 아키텍처를 백그라운드로 실행합니다.
```bash
cd /home/neuro/drowsiness_project
docker-compose up --build -d
```
*(참고: 생체 신호 처리(EEG) 컨테이너는 LSL 네트워크 패킷의 원활한 멀티캐스트 수신을 보장하기 위해 호스트 네트워크 모드(`network_mode: "host"`)로 구성되어 있습니다.)*

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

### 4.2. 내부 노드 간 데이터 통신 (POST)
* `/ingest/camera` : 비전 처리 노드가 주기적으로 산출된 카메라 지표를 송신하는 데이터 수집(Ingestion) 엔드포인트
* `/ingest/eeg` : 생체 신호 처리 노드가 주기적으로 산출된 뇌파 지표를 송신하는 데이터 수집(Ingestion) 엔드포인트

---

## 5. 프로젝트 디렉토리 구조 (Directory Structure)
```text
app/          # 통합 융합 엔진, REST API 라우터, 웹 대시보드 서비스 핵심 로직
pp_nrsc/      # 뇌파 추론 및 기계학습 모델 검증/적용 모듈 (확장 및 실험용)
scripts/      # 가상 데이터 주입 시뮬레이션 및 단위 컴포넌트 검증 스크립트
docker/       # 마이크로서비스 구성을 위한 서비스 노드별 Dockerfile 정의
models/       # 뇌파 데이터 분석 및 평가를 위한 사전 학습된 인공지능 모델
docs/         # 프로젝트 운영 지침, 부가 기술 문서 및 커맨드 가이드
```
