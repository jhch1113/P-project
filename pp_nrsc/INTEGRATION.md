# 졸음운전 감지 시스템 - 최종 구성
## 아키텍처

```
MUSE2 (EEG 센서)
    ↓ (1초씩 256 샘플)
Jetson Orin Nano
    ├─ Step 1: muse2_reader.py → AF7/AF8 청크 수신
    ├─ Step 2: FastAPI 서버 (muse_inference_api.py) → 전처리/변환
    │  ├─ Bandpass filter (0.5~40Hz)
    │  ├─ Median 제거
    │  ├─ Z-score 정규화
    │  문높동 모델 추론 → 확률률값 + 상태(0/1)
    ├─ Step 3: drowsiness_scorer.py → 점수화 로직
    │  ├─ 즘각 점수 (현재 확률률 × 100)
    │  ├─ 평활화 점수 (최근 30개 평균)
    │  ├─ 누적 쉁음 시간 (최근 30초)
    │  문높동 경고 판정 (즘시 또는 누적)
    문늀─ Step 4: realtime_detector.py → OpenCV UI + 경고

    ↓
  화면 표시 + 경고음/진동 발생
```

---

## 설정 및 실행

### 1. FastAPI 서버 시작 (Step 2)

```bash
# 모델 경로 설정
export MUSE_MODEL_PATH="/path/to/MUSE_activity_model.keras"

# 개발 모드
python muse_inference_api.py

# 또는 프로덕션 (Jetson에서)
uvicorn muse_inference_api:app --host 0.0.0.0 --port 8000 --workers 1
```

 확인: `curl http://localhost:8000/health`

### 2. 점수화 로직 테스트 (Step 3)

```bash
python drowsiness_scorer.py
```

 7초의 시뮬레이션 데이터로 동작 확인

### 3. 실시간 감지 시작 (Step 4)

```bash
# Jetson (디스플레이 있는 경우)
python realtime_detector.py

# 또는 헤드리스 (SSH)
python realtime_detector.py --headless
```

---

## 📊 점수화 로직 설명

### 입력: API 응답
```json
{
  "t_start_sec": 5.0,
  "t_end_sec": 10.0,
  "prob_raw": 0.62,           ← 원본 확률 (0~1)
  "prob_scaled": 0.68,        ← 정규화된 확률 (0~1)
  "state": 1                  ← 0=sleep, 1=awake
}
```

### 점수화 단계

| 지표 | 계산 | 의미 |
|------|------|------|
| **Instant Score** | `prob_scaled × 100` | 현재 윈도우 위험도 (0~100) |
| **Smoothed Score** | 최근 30개 평균 | 노이즈 제거된 안정적 점수 |
| **Accumulated Time** | 30초 중 졸음 누적 시간 | 졸음이 얼마나 지속됐나 |
| **Risk Level** | Smoothed ≥80? → "위험" | 사용자 친화적 레벨 |
| **Should Alert** | Instant≥80 OR Accum≥20초 | 즉시 경고 필요? |

### 임계값 커스터마이징

```python
scorer = DrowsinessScorer(
    drowsy_threshold=0.55,           # 이 이상 확률 = 졸음
    instant_alert_threshold=0.80,    # 순간 경고 임계값
    accumulated_time_limit=20.0,     # 30초 중 20초 이상 = 위험
    window_size=30,                  # 평활화 윈도우 개수
)
```

---

##  사용자 커스터마이제이션 포인트

### A. 경고 콜백 (경고음/진동/SMS 등)

**파일**: `realtime_detector.py`

```python
def my_alert_callback(score: DrowsinessScore):
    """경고 트리거 시 호출"""
    if score.risk_level == "위험":
        os.system("aplay /path/to/alarm.wav")  # 경고음
        # 또는 진동 모터
        os.system("echo 1 > /sys/class/gpio/gpio17/value")
        # 또는 원격 서버에 전송
        requests.post("https://alertserver.com/incident", json={
            "severity": "critical",
            "score": score.smoothed_score,
        })

detector = RealtimeDrowsinessDetector(
    alert_callback=my_alert_callback
)
```

### B. 점수 로깅/저장

**파일**: `drowsiness_scorer.py` → 수정

```python
def score(self, ...):
    # ... 기존 코드 ...
    score_obj = DrowsinessScore(...)
    
    # ✨ 로깅 추가
    with open("drowsiness_log.csv", "a") as f:
        f.write(f"{score_obj.timestamp},"
                f"{score_obj.smoothed_score},"
                f"{score_obj.risk_level},"
                f"{score_obj.should_alert}\n")
    
    return score_obj
```

### C. OpenCV UI 커스터마이징

**파일**: `realtime_detector.py` → `OpenCVDisplay.create_dashboard()`

현재: 원형 게이지 + 위험도 표시
추가 가능: 그래프, 시간 추이, 운전자 얼굴 감지 등

---

## 📈 성능 최적화 (Jetson Orin Nano)

### 1. 멀티 워커 불가 (메모리 문제)
```bash
# 불가 불가
uvicorn muse_inference_api:app --workers 4

#  권장
uvicorn muse_inference_api:app --workers 1
```

### 2. TensorFlow GPU 사용
```bash
# Jetson에서 TF GPU 활성화
export CUDA_VISIBLE_DEVICES=0
# muse_inference_api.py가 자동으로 GPU 사용
```

### 3. 배치 크기 조정
```python
# muse_inference_api.py 내부
preds = model.predict(windows, batch_size=64, verbose=0)  # 128 → 64
```

---

## 🧪 테스트 흐름

### 1. API 테스트 (모델 없이)
```bash
python test_api.py
```
결과: ` 전체 통과` (모델 파일 없어도 에러 처리 확인)

### 2. 점수화 로직 테스트
```bash
python drowsiness_scorer.py
```
결과: 7초 시뮬레이션, 최종 3개 윈도우 결과 표시

### 3. 통합 테스트 (전체 파이프라인)
```bash
# Terminal 1: FastAPI 서버 시작
python muse_inference_api.py

# Terminal 2: 실시간 감지 실행
python realtime_detector.py
```

---

## 📝 현재 상황 및 남은 일

###  완료
- [x] FastAPI 서버 (전처리/변환)
- [x] 모델 업로드 (정확도 53%)
- [x] 점수화 로직
- [x] 실시간 UI

### ⏳ 필요한 작업
- [ ] MUSE2 LSL 스트림 통합 (`muse2_reader.py`)
- [ ] 실제 Jetson 환경 테스트
- [ ] 경고 하드웨어 연결 (진동/LED/음성)
- [ ] 정확도 개선 (더 많은 노이즈 데이터로 파인튜닝)

---

##  주의사항

### 1. 정확도 53%의 의미
-  프로토타입 검증용으로 충분
- 주의: 실제 상용화는 더 높은 정확도 필요
-  개선 방법:
  - 더 많은 MUSE2 노이즈 데이터 수집
  - 전이학습 (ImageNet → EEG)
  - 다른 채널 추가 (Tp7, Tp8)

### 2. 1초 Stride의 한계
- 5초 윈도우로 1초씩 슬라이딩 → 중복 있음
- 반응성: ~5초 지연 (모델 지연)
- 개선: 더 작은 윈도우 (2초) 또는 다른 모델 구조

### 3. MUSE2 자체 한계
- 낮은 샘플레이트 (256Hz)
- 높은 노이즈
- 웨어러블 특성상 움직임 아티팩트
- → Bandpass filter로 어느 정도 보정 중

---

## 파일 구조

```
/nrsc/
├── muse_inference_api.py        # FastAPI 서버 (Step 2)
├── test_api.py                  # 테스트 (API 검증)
├── drowsiness_scorer.py         # 점수화 로직 (Step 3) - 새로운 파일
├── realtime_detector.py         # 실시간 감지 + UI (Step 4) - 새로운 파일
├── README.md                    # (원본)
└── INTEGRATION.md               # 현재 파일
```

---

## 핵심 메시지

체크해야 할 일
1. FastAPI 서버에서 받은 `prob_raw` / `prob_scaled` / `state`
2. `DrowsinessScorer`로 점수화
3. 위험도별 액션 (경고/기록/서버 전송)

FastAPI가 이미 수행하는 일:
- 데이터 입력 → 전처리 → 모델 변환 → 확률/상태 반환

이제 점수화와 UI만 신경 쓰면 끝이다.

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| `모델 파일 없음` | `export MUSE_MODEL_PATH=...` |
| `포트 8000 사용 중` | `lsof -i :8000` → `kill` |
| `numpy shape 에러` | AF7/AF8 길이 같은지 확인 |
| `점수가 항상 0` | API가 정상인지 `/health` 확인 |
| `경고가 너무 자주` | `drowsy_threshold` 또는 `accumulated_time_limit` 상향 |

---

**문의**: 각 파일 상단의 주석 참고 또는 코드 내 `TODO` 검색
