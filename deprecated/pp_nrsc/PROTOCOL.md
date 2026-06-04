# MUSE2 ↔ Jetson 통신 프로토콜 사양서

## 목표
- **MUSE2 팀**: EEG 데이터를 표준 포맷으로 전달
- **Jetson 팀**: 다양한 소스에서 데이터 수신, 처리, 결과 반환
- **목표**: 느슨한 결합, 쉬운 통합, 확장 가능

---

## 1. 데이터 포맷: EEGChunk

모든 EEG 데이터는 다음 형식으로 표준화된다.

```python
class EEGChunk:
    af7: list[float]        # AF7 채널, 256개 샘플 (1초 @ 256Hz)
    af8: list[float]        # AF8 채널, 256개 샘플 (1초 @ 256Hz)
    timestamp: float        # Unix timestamp (수신 시간)
    sample_rate: int        # 256 (고정)
    sequence_id: int        # 순서 번호 (데이터 손실 감지)
    metadata: dict          # 추가 정보 (배터리, 신호 품질 등)
```

### JSON 예제
```json
{
  "af7": [12.3, 11.9, 12.1, ...],      // 256개 요소
  "af8": [10.1, 10.5, 10.2, ...],      // 256개 요소
  "timestamp": 1714818234.567,          // Unix timestamp
  "sample_rate": 256,
  "sequence_id": 1,
  "metadata": {
    "battery": 85,
    "signal_quality": 0.92
  }
}
```

---

## 2. 소스 구현 가이드

### A. LSL 스트림 (권장: 실시간 EEG 시스템)

**MUSE2 팀 구현 예제:**
```python
from eeg_data_source import EEGReader, EEGChunk
from pylsl import resolve_stream, StreamInlet

class MUSE2Reader(EEGReader):
    def connect(self):
        # MUSE2 LSL 스트림 찾기
        streams = resolve_stream('type', 'EEG')
        if not streams:
            return False
        self.inlet = StreamInlet(streams[0])
        self.is_connected = True
        return True
    
    def read_chunk(self, timeout=1.0):
        chunk_af7 = []
        chunk_af8 = []
        
        # 1초분 데이터 수집 (256 샘플)
        for _ in range(256):
            sample, ts = self.inlet.pull_sample(timeout=timeout/256)
            if sample:
                chunk_af7.append(float(sample[0]))  # AF7
                chunk_af8.append(float(sample[1]))  # AF8
        
        if len(chunk_af7) != 256:
            return None  # 데이터 손실
        
        return EEGChunk(
            af7=chunk_af7,
            af8=chunk_af8,
            timestamp=time.time(),
            sequence_id=self.chunk_count,
            metadata={
                "battery": get_battery_level(),
                "signal_quality": calculate_signal_quality(sample)
            }
        )
```

**설정:**
```yaml
eeg_source:
  type: muse2
  kwargs:
    device_name: "Muse"
```

---

### B. TCP 네트워크 (다른 머신의 MUSE2)

**MUSE2 서버 (데이터 송신측):**
```python
import socket
import json

def send_eeg_to_jetson():
    sock = socket.socket()
    sock.connect(("jetson_ip", 5000))
    
    while True:
        chunk = read_from_muse2()  # EEGChunk 수신
        chunk_json = json.dumps(chunk.to_dict())
        sock.send(chunk_json.encode() + b"\n")
```

**Jetson 수신:**
```yaml
eeg_source:
  type: tcp
  kwargs:
    host: "192.168.1.100"    # MUSE2 서버 IP
    port: 5000
```

---

### C. 파일 (테스트/재현용)

**CSV 파일 포맷:**
```csv
AF7,AF8
12.3,10.1
11.9,10.5
12.1,10.2
...
```

**설정:**
```yaml
eeg_source:
  type: file
  kwargs:
    filepath: /path/to/muse_data.csv
    loop: true              # 반복 재생
```

---

## 3. 검증 체크리스트 (MUSE2 팀)

구현 시 다음을 확인해야 한다.

- [ ] **샘플 레이트**: 반드시 256Hz (다르면 변환 필요)
- [ ] **채널 순서**: AF7이 첫 번째, AF8이 두 번째
- [ ] **샘플 개수**: 각 청크는 정확히 256개 (1초분)
- [ ] **타임스탐프**: Unix timestamp (float)
- [ ] **sequence_id**: 증가하는 정수 (데이터 손실 감지용)
- [ ] **에러 처리**: 연결 끊김/노이즈 데이터에 대한 복구
- [ ] **메타데이터**: 배터리, 신호 품질 등 추가 정보

**테스트:**
```python
from eeg_data_source import MUSE2Reader

reader = MUSE2Reader()
if reader.connect():
    chunk = reader.read_chunk(timeout=2.0)
    
    # 검증
    assert len(chunk.af7) == 256
    assert len(chunk.af8) == 256
    assert chunk.sample_rate == 256
    assert chunk.timestamp > 0
    print("MUSE2Reader 검증 통과")
```

---

## 4. 에러 처리 및 복구

### 데이터 손실 감지
```python
# Jetson 측: sequence_id 확인
last_seq_id = 0
chunk = reader.read_chunk()

if chunk.sequence_id != last_seq_id + 1:
    missing = chunk.sequence_id - (last_seq_id + 1)
    print(f"경고: {missing}개 청크 손실됨")

last_seq_id = chunk.sequence_id
```

### 연결 끊김 복구
```python
reader = create_reader("muse2")

while True:
    try:
        if not reader.is_connected:
            reader.connect()
        chunk = reader.read_chunk(timeout=2.0)
        if chunk:
            process(chunk)
    except Exception as e:
        print(f"⚠️ 에러: {e}")
        reader.disconnect()
        time.sleep(1)  # 재연결 대기
```

---

## 5. 성능 요구사항

| 항목 | 요구사항 | 비고 |
|------|---------|------|
| **레이턴시** | < 100ms | 1초 청크 수집 + 전송 |
| **데이터 손실율** | < 1% | 데이터 무결성 |
| **배터리 소비** | ~8시간 | MUSE2 자체 사양 |
| **신호 품질** | > 0.7 | 0~1 범위 |

---

## 6. 모니터링 및 디버깅

### 통계 확인 (Jetson 측)
```python
reader = create_reader("muse2")
reader.start_stream(callback=process)

time.sleep(10)
stats = reader.get_stats()
print(f"청크 수: {stats['chunk_count']}")
print(f"에러 수: {stats['error_count']}")
print(f"연결 상태: {stats['connected']}")
```

### 데이터 로깅
```python
def log_chunk(chunk):
    with open("eeg_debug.jsonl", "a") as f:
        f.write(json.dumps(chunk.to_dict()) + "\n")
```

---

## 7. 마이그레이션 경로

### 단계 1: 파일 기반 테스트 (현재)
```yaml
eeg_source:
  type: file
  kwargs:
    filepath: test_data.csv
```

### 단계 2: TCP 네트워크 테스트
```yaml
eeg_source:
  type: tcp
  kwargs:
    host: localhost
    port: 5000
```

### 단계 3: LSL 실시간 (MUSE2 직접)
```yaml
eeg_source:
  type: muse2
  kwargs:
    device_name: "Muse"
```

---

## 문의 및 지원

- **MUSE2 팀**: eeg_data_source.py의 MUSE2Reader 클래스 구현
- **Jetson 팀**: realtime_detector.py에서 소스 선택 및 처리
- **공동**: config.yaml에서 설정 관리

---

## 핵심 메시지

MUSE2 팀과 Jetson 팀은 **EEGChunk** 포맷으로만 통신한다.

- MUSE2 팀: LSL/TCP/기타 → **EEGChunk** 변환
- Jetson 팀: **EEGChunk** 수신 → 처리 → 점수 계산

느슨한 결합으로 양쪽 독립적 개발이 가능하다.
