# 팀별 작업 체크리스트

## MUSE2 팀 - EEG 데이터 수집 및 전송

### Phase 1: 기본 구현
- [ ] `eeg_data_source.py`의 `MUSE2Reader` 클래스 구현
- [ ] LSL 스트림 연결 (또는 MUSE2 API)
- [ ] AF7, AF8 채널 데이터 추출
- [ ] 샘플 레이트 확인 (256Hz)
- [ ] `sequence_id` 순차 증가 관리
- [ ] 배터리/신호 품질을 `metadata`에 포함

### Phase 2: 테스트 및 검증
```bash
# 1. 인터페이스 구현 확인
python validate_eeg.py --source muse2

# 2. 데이터 품질 확인
# - AF7/AF8 각 256개 샘플? ✓
# - 타임스탐프 정상? ✓
# - Sequence ID 연속? ✓
# - 배터리 충분? ✓
```

### Phase 3: 배포 옵션

**옵션 A: LSL 스트림 (권장 - 실시간)**
- MUSE2 디바이스 → LSL 네트워크 브로드캐스트
- Jetson이 LSL로 수신
- 설정:
  ```yaml
  eeg_source:
    type: muse2
    kwargs:
      device_name: "Muse"
  ```

**옵션 B: TCP 네트워크**
- MUSE2 시스템에서 EEGChunk를 JSON으로 TCP 전송
- Jetson이 TCP로 수신
- 예제: [PROTOCOL.md의 섹션 4B](PROTOCOL.md#b-tcp-네트워크-다른-머신의-muse2)
- 설정:
  ```yaml
  eeg_source:
    type: tcp
    kwargs:
      host: "192.168.1.100"
      port: 5000
  ```

**옵션 C: 파일 기반 (테스트/재현)**
- CSV 파일로 저장된 EEG 데이터 사용
- 테스트/디버깅에만 사용
- 설정:
  ```yaml
  eeg_source:
    type: file
    kwargs:
      filepath: /path/to/data.csv
  ```

### Phase 4: 검증 체크리스트
```python
from eeg_data_source import MUSE2Reader, EEGChunk

reader = MUSE2Reader()
reader.connect()

chunk = reader.read_chunk()

# 반드시 확인할 항목
assert len(chunk.af7) == 256          # ✓
assert len(chunk.af8) == 256          # ✓
assert chunk.sample_rate == 256       # ✓
assert chunk.timestamp > 0            # ✓
assert chunk.sequence_id >= 0         # ✓
print("✅ MUSE2Reader 구현 완료")
```

### 📞 지원
- 문제 발생 시: `validate_eeg.py` 로그 확인
- 데이터 포맷 확인: [PROTOCOL.md](PROTOCOL.md)
- 구현 예제: `eeg_data_source.py` 주석

---

## Jetson 팀 - 실시간 감지 및 UI

### Phase 1: 환경 설정
```bash
# 1. 의존성 설치
pip install fastapi uvicorn tensorflow scipy pandas numpy \
            opencv-python pyyaml websockets

# 2. 모델 파일 준비
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras

# 3. FastAPI 서버 시작
python muse_inference_api.py
# ✓ http://localhost:8000/health 확인
```

### Phase 2: 설정 파일 작성
```bash
# 1. 기본 설정 생성
python config.py config.yaml

# 2. config.yaml 수정
# - EEG 소스 설정 (MUSE2 팀과 협의)
# - API URL
# - 점수화 임계값
# - 경고 설정
```

### Phase 3: 코드 통합
```bash
# 1. 점수화 로직 테스트
python drowsiness_scorer.py
# → 7초 시뮬레이션, 3개 윈도우 결과 확인

# 2. 실시간 감지 테스트 (테스트 데이터 사용)
python realtime_detector.py
# → OpenCV UI에 대시보드 표시

# 3. 실제 MUSE2 데이터로 테스트
# config.yaml의 eeg_source 변경
# python realtime_detector.py
```

### Phase 4: 경고 커스터마이징
`realtime_detector.py`의 `alert_callback` 수정:

```python
def my_alert_callback(score):
    if score.risk_level == "위험":
        # 경고음
        os.system("aplay /path/to/alarm.wav")
        
        # 진동 모터 (GPIO)
        os.system("echo 1 > /sys/class/gpio/gpio17/value")
        
        # 원격 서버 알림
        requests.post("https://alert.server.com/incident", json={
            "level": score.risk_level,
            "score": score.smoothed_score,
        })

detector = RealtimeDrowsinessDetector(
    alert_callback=my_alert_callback
)
```

### Phase 5: 성능 최적화
```bash
# 1. Jetson 리소스 확인
nvidia-smi -l 1

# 2. TensorFlow GPU 활성화
export CUDA_VISIBLE_DEVICES=0

# 3. 배치 크기 조정 (muse_inference_api.py)
preds = model.predict(windows, batch_size=64, verbose=0)  # 메모리 부족 시 감소
```

### Phase 6: 디버깅 및 로깅
```bash
# EEG 데이터 검증
python validate_eeg.py --source file --file test_data.csv

# API 헬스체크
curl http://localhost:8000/health

# 점수화 로직 단위 테스트
python -c "from drowsiness_scorer import *; scorer = DrowsinessScorer(); print(scorer.score(0.8))"
```

### 📊 설정 예제 (config.yaml)

**파일 기반 (초기 테스트)**
```yaml
eeg_source:
  type: file
  kwargs:
    filepath: test_data.csv
    loop: true
```

**TCP 네트워크 (MUSE2와 분리)**
```yaml
eeg_source:
  type: tcp
  kwargs:
    host: "192.168.1.100"    # MUSE2 서버
    port: 5000
```

**LSL 실시간 (MUSE2 직접 연결)**
```yaml
eeg_source:
  type: muse2
  kwargs:
    device_name: "Muse"
```

### 📞 지원
- 문제 발생 시: `config.yaml` 및 로그 확인
- API 문제: `curl http://localhost:8000/health` 확인
- 점수화 문제: `drowsiness_scorer.py` 임계값 조정

---

## 공동 작업 - 통합 테스트

### Week 1: 개별 검증
- MUSE2팀: `validate_eeg.py` → EEG 데이터 OK
- Jetson팀: `test_api.py` → API OK

### Week 2: 로컬 테스트
```bash
# Terminal 1 (어느 팀에서든 실행 가능)
python muse_inference_api.py

# Terminal 2 (Jetson팀)
python realtime_detector.py  # config.yaml eeg_source=file 설정
```

### Week 3: 실제 통합
```bash
# Terminal 1: 어느 팀에서든
python muse_inference_api.py

# Terminal 2: MUSE2팀이 EEG 스트림 시작
# (TCP 또는 LSL로 data 송출)

# Terminal 3: Jetson팀
python realtime_detector.py  # config.yaml eeg_source=tcp/muse2 설정
```

### Week 4: 성능 최적화
- 레이턴시 측정 (`validate_eeg.py` 결과)
- 정확도 평가 (수동 검증)
- 시스템 통합 테스트

---

## 참고 자료

| 문서 | 설명 |
|------|------|
| [PROTOCOL.md](PROTOCOL.md) | 📋 EEG 데이터 포맷 & 통신 규약 (필독!) |
| [INTEGRATION.md](INTEGRATION.md) | 📖 전체 시스템 아키텍처 |
| [eeg_data_source.py](eeg_data_source.py) | 💻 인터페이스 & 구현 예제 |
| [config.yaml](config.yaml) | ⚙️ 설정 파일 예제 |

---

## 트러블슈팅

### MUSE2팀
| 문제 | 해결 |
|------|------|
| LSL 스트림 못 찾음 | `resolve_stream()` 확인, device 켰는지 확인 |
| 데이터 길이 안 맞음 | 256 샘플 = 1초 @ 256Hz 확인 |
| Sequence ID 끊김 | 수신 타임아웃 또는 데이터 손실 (로그 확인) |

### Jetson팀
| 문제 | 해결 |
|------|------|
| API 연결 안 됨 | `http://localhost:8000/health` 확인 |
| 메모리 부족 | `--workers 1` 사용, 배치 크기 감소 |
| 점수가 항상 0 | API 응답 형식 확인, 로그 출력 |
| 경고 안 울림 | `alert.enabled=true` & `alert_callback` 확인 |

---

## 최종 체크리스트

### 배포 전
- [ ] MUSE2팀: `validate_eeg.py` 모든 항목 PASS
- [ ] Jetson팀: `test_api.py` 모든 항목 PASS
- [ ] 통합 테스트: 끝-끝(E2E) 정상 작동
- [ ] 성능 측정: 레이턴시 < 500ms, 정확도 > 50%
- [ ] 문서 최신화: PROTOCOL.md, config.yaml 검토

### 상용화 전
- [ ] MUSE2 정확도 개선 (현재 53% → 목표 >80%)
- [ ] 에러 처리 강화 (연결 끊김, 데이터 손실)
- [ ] UI/UX 개선 (대시보드, 경고 알림)
- [ ] 성능 최적화 (지연시간, 메모리)
- [ ] 보안 검토 (API, 네트워크, 데이터)

---

모든 파일을 충분히 이해한 후, 각 팀이 독립적으로 작업할 수 있다.
