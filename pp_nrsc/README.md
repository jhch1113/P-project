# Neuro Sync - AI 기반 졸음운전 감지 시스템

Muse2 EEG 데이터를 실시간으로 분석하여 졸음/각성 상태를 예측하는 프로젝트입니다. AF7/AF8 채널을 전처리하고 TensorFlow 모델로 추론한 뒤, 점수화 및 위험도 판단을 수행합니다.

## 주요 기능

- 실시간 추론: 1초마다 윈도우 예측
- 고급 전처리: 노이즈, 아티팩트 제거
- 모듈형 구조: FastAPI 서버 + 점수화 엔진 + 모니터링
- 입력 지원: Muse2, CSV 파일, TCP
- 데이터 분석 및 검증 도구 포함

## 성능 지표

| 항목 | 값 | 설명 |
|------|-----|------|
| 정확도 | 55% | Awake/Sleep 분류 정확도 |
| 정밀도 | 54% | Sleep 예측 정확도 |
| 재현율 | 67% | Sleep 감지 민감도 |
| 지연 시간 | 약 5초 | 5초 윈도우 기준 |
| 갱신 주기 | 1Hz | 초당 1회 판정 |
| 메모리 | 약 800MB | TensorFlow 모델 포함 |

## 시스템 구조

```
Muse2 헤드셋 (256Hz EEG)
         |
         | AF7, AF8 채널
         v
   고급 전처리
   - 밴드패스 필터 (0.5-40Hz)
   - 60Hz 노치 필터
   - EOG 아티팩트 제거
   - 신호 품질 평가
         |
         v
   TensorFlow 모델
   - CNN-LSTM 기반 추론
   - 확률 값 출력
         |
         v
   점수화 엔진
   - 히스테리시스 적용
   - 평활화 및 누적 졸음 시간 계산
   - 위험도 레벨 산출
         |
         v
   경고 시스템
   - 시각/청각 알림
   - OpenCV 대시보드
   - 원격 연동 선택 가능
```

## 설치 및 실행

### 1. 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 모델 준비

```bash
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
# 테스트용 더미 모델
export MUSE_MODEL_PATH=dummy_model.keras
```

### 3. 서버 실행

```bash
python muse_inference_api.py
curl http://localhost:8000/health
```

### 4. 실시간 감지 실행

```bash
python realtime_detector.py
```

## 테스트 및 검증

```bash
python test_accuracy_improvement.py
python test_api.py
python validate_eeg.py --source file --file awake_study.csv
```

## 사용 데이터

- `awake_study.csv`, `awake_study2.csv`, `awake_study3.csv`, `awake_study4.csv` (깨어있는 상태)
- `bedtime.csv`, `bedtime2.csv` (졸음 상태)

## 현재 진행 상황

- FastAPI 추론 파이프라인 구현: 완료
- 실시간 전처리 및 예측: 완료
- 점수화 로직 연결: 완료
- `awake_study4.csv` 추가 반영: 완료
- 남은 작업: 모델 경로 확인 및 배포 검증

## 프로젝트 구조

```
nrsc/
├── muse_inference_api.py       # FastAPI 추론 서버
├── drowsiness_scorer.py        # 점수화 및 위험도 로직
├── realtime_detector.py        # 실시간 모니터링 클라이언트
├── test_api.py                 # API 스모크 테스트
├── advanced_preprocessing.py   # EEG 전처리
├── advanced_postprocessing.py  # 후처리 및 품질 필터링
├── analyze_improvements.py     # 전처리 개선 분석
├── test_preprocessing_quality.py # 전처리 품질 검증
├── eeg_data_source.py          # EEG 데이터 입출력
├── config.py                   # 설정 관리
├── config.yaml                 # YAML 설정 파일
├── validate_eeg.py             # 데이터 검증 도구
├── train_muse_model_v2.py      # 모델 학습 스크립트
├── dummy_model.keras           # 테스트 모델
├── test_accuracy_improvement.py # 정확도 평가
├── README.md                   # 프로젝트 문서
├── IMPROVEMENTS_V2.md          # 개선 사항 정리
├── PROTOCOL.md                 # API 프로토콜 문서
├── CHECKLIST.md                # 개발 체크리스트
├── INTEGRATION.md              # 통합 가이드
├── alert_handlers.py           # 경고 처리 유틸리티
├── local_realtime.py           # 로컬 테스트 도구
└── requirements.txt            # Python 의존성
```

## API 엔드포인트

### GET /health
서비스 상태와 모델 로드 상태 확인

### POST /predict/batch
JSON 입력으로 배치 추론 수행

예시:

```json
{
  "af7": [12.3, 11.9, ...],
  "af8": [10.1, 10.5, ...],
  "apply_minmax": true
}
```

### POST /session/start → /session/{id}/append → /session/{id}/end
스트리밍 세션 방식으로 실시간 데이터를 전송하여 추론 처리

### WebSocket /ws/stream
저지연 실시간 스트리밍 엔드포인트

## 설정

`config.yaml`에서 시스템 설정을 조정합니다:

```yaml
eeg_source:
  type: muse2
  kwargs:
    device_name: "Muse"

scorer:
  window_size: 30
  drowsy_threshold: 0.55
  instant_alert_threshold: 0.80
  accumulated_time_limit: 20.0

alert:
  enabled: true
  alarm_file: null
  gpio_pin: null
  send_to_server: false
```

## 개선 사항 (v2)

### 고급 전처리 적용 결과
- 정확도 49% → 55% 향상
- Sleep 감지 재현율 56% → 67% 개선
- 60Hz 노치 필터, EOG 아티팩트 제거 추가
- 신호 품질 기반 동적 임계값 적용

### 데이터 확장
- 다중 CSV 파일 통합
- 클래스 균형 조정
- 다양한 조건의 데이터 수집

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

**주의**: 이 시스템은 연구/프로토타입 용도입니다. 실제 운전 환경에서는 전문가 검증이 필요합니다.
