"""
EEG 데이터 소스 추상화 - 플러그인 아키텍처
==========================================

MUSE2 작업자와 Jetson 작업자 간의 명확한 인터페이스.
다양한 입력 소스(MUSE2, 파일, TCP, LSL 등)에 대응 가능.

사용처:
- realtime_detector.py에서 소스 선택 시 플러그인처럼 사용
- MUSE2 팀: 자신의 리더만 구현하면 됨 (인터페이스 준수)
- Jetson 팀: 소스 변경 없이 작동
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
from collections import deque
import threading
import json
import time
import numpy as np


@dataclass
class EEGChunk:
    """
    표준 EEG 데이터 포맷
    
    모든 소스(MUSE2, 파일, 네트워크)는 이 형식으로 변환해서 전달
    """
    af7: list[float]        # AF7 채널 (256 샘플 = 1초, FS=256Hz)
    af8: list[float]        # AF8 채널 (256 샘플 = 1초, FS=256Hz)
    timestamp: float        # 수신 타임스탐프 (unix time)
    sample_rate: int = 256  # 고정: MUSE2는 256Hz
    sequence_id: int = 0    # 순서 번호 (데이터 손실 감지용)
    metadata: dict = None   # 추가 정보 (배터리, 신호 품질 등)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # 검증
        if len(self.af7) != 256 or len(self.af8) != 256:
            raise ValueError(f"각 채널은 256 샘플이어야 함. "
                           f"받은 길이: AF7={len(self.af7)}, AF8={len(self.af8)}")
    
    def to_dict(self) -> dict:
        """JSON 직렬화용"""
        return {
            "af7": self.af7,
            "af8": self.af8,
            "timestamp": self.timestamp,
            "sample_rate": self.sample_rate,
            "sequence_id": self.sequence_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EEGChunk":
        """JSON 역직렬화"""
        return cls(
            af7=data["af7"],
            af8=data["af8"],
            timestamp=data["timestamp"],
            sample_rate=data.get("sample_rate", 256),
            sequence_id=data.get("sequence_id", 0),
            metadata=data.get("metadata", {}),
        )


class EEGReader(ABC):
    """
    EEG 데이터 소스의 추상 기본 클래스
    
    모든 리더는 이 인터페이스를 구현해야 함:
    - MUSE2Reader
    - LSLReader (Lab Streaming Layer)
    - TCPReader (네트워크)
    - FileReader (CSV/저장 파일)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.running = False
        self.lock = threading.Lock()
        self.is_connected = False
        self.error_count = 0
        self.chunk_count = 0
    
    @abstractmethod
    def connect(self) -> bool:
        """연결 시작. 성공하면 True 반환."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """연결 종료."""
        pass
    
    @abstractmethod
    def read_chunk(self, timeout: float = 1.0) -> Optional[EEGChunk]:
        """
        1초 분량의 EEG 청크 읽기 (블로킹).
        
        Args:
            timeout: 최대 대기 시간 (초)
        
        Returns:
            EEGChunk 또는 None (타임아웃/에러)
        """
        pass
    
    def start_stream(self, callback: Callable[[EEGChunk], None], daemon: bool = True):
        """
        백그라운드 스트리밍 시작 (스레드).
        
        청크가 들어올 때마다 callback() 호출.
        """
        if not self.connect():
            raise RuntimeError(f"{self.name}: 연결 실패")
        
        self.running = True
        thread = threading.Thread(
            target=self._stream_loop,
            args=(callback,),
            daemon=daemon
        )
        thread.start()
        return thread
    
    def _stream_loop(self, callback: Callable):
        """내부: 스트리밍 루프"""
        while self.running:
            try:
                chunk = self.read_chunk(timeout=2.0)
                if chunk:
                    with self.lock:
                        self.chunk_count += 1
                    callback(chunk)
            except Exception as e:
                with self.lock:
                    self.error_count += 1
                print(f"❌ {self.name} 에러: {e}")
    
    def stop_stream(self):
        """스트리밍 중지."""
        self.running = False
        self.disconnect()
    
    def get_stats(self) -> dict:
        """통계 조회"""
        with self.lock:
            return {
                "name": self.name,
                "connected": self.is_connected,
                "chunk_count": self.chunk_count,
                "error_count": self.error_count,
            }


# ============================================================
# 구현 예제 1: MUSE2 (실제 구현 - MUSE2 팀이 작성)
# ============================================================
class MUSE2Reader(EEGReader):
    """
    MUSE2 헤드셋에서 데이터 읽기 (LSL 또는 직접 연결).
    
    MUSE2 팀이 구현할 부분:
    - LSL 스트림 연결
    - 채널 선택 (AF7, AF8)
    - 샘플 레이트 확인 (256Hz)
    - 배터리/신호 품질 메타데이터
    """
    
    def __init__(self, device_name: str = "Muse"):
        super().__init__("MUSE2Reader")
        self.device_name = device_name
        self.stream = None
        self.sample_rate = 256
        self.inlet = None
        self._simulate = False
        self._seq = 0
    
    def connect(self) -> bool:
        """LSL 스트림 연결"""
        try:
            # 우선 pylsl이 설치되어 있으면 LSL 스트림을 시도한다.
            try:
                from pylsl import resolve_streams, StreamInlet
            except Exception:
                resolve_streams = None
                StreamInlet = None

            if resolve_streams is not None and StreamInlet is not None:
                all_streams = resolve_streams(wait_time=2)
                streams = [s for s in all_streams if s.type().upper() == 'EEG']
                if not streams:
                    # LSL 스트림이 없으면 시뮬레이션 모드로 전환
                    self._simulate = True
                else:
                    self.inlet = StreamInlet(streams[0])
                    self._simulate = False
            else:
                # pylsl 미설치: 시뮬레이션 모드
                self._simulate = True

            self.is_connected = True
            if self._simulate:
                print(f"{self.name}: pylsl 미설치 또는 스트림 미발견 — 시뮬레이션 모드")
            else:
                print(f"{self.name}: LSL 스트림에 연결됨")
            return True
        except Exception as e:
            print(f"{self.name}: 연결 실패 - {e}")
            return False
    
    def disconnect(self):
        """연결 종료"""
        self.is_connected = False
    
    def read_chunk(self, timeout: float = 1.0) -> Optional[EEGChunk]:
        """
        LSL 스트림에서 1초 청크 읽기.
        
        실제 구현 (MUSE2 팀):
            chunk_af7 = []
            chunk_af8 = []
            for i in range(256):
                sample, ts = self.inlet.pull_sample(timeout=0.1)
                chunk_af7.append(sample[0])  # AF7 채널
                chunk_af8.append(sample[1])  # AF8 채널
            
            return EEGChunk(
                af7=chunk_af7,
                af8=chunk_af8,
                timestamp=time.time(),
                metadata={"battery": ..., "quality": ...}
            )
        """
        # 실제 LSL 스트림에서 읽기
        if self._simulate:
            # 시뮬레이션 신호: 알파(10Hz)+베타(20Hz)+노이즈
            n = 256
            t = np.arange(n) / float(self.sample_rate)
            af7 = (15.0 * np.sin(2 * np.pi * 10 * t) + 5.0 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 8, n)).astype(float).tolist()
            af8 = (12.0 * np.sin(2 * np.pi * 10 * t + 0.5) + 4.0 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 8, n)).astype(float).tolist()
            # 실제 장비처럼 1초 간격으로 청크를 생성
            time.sleep(1.0)
            chunk = EEGChunk(af7=af7, af8=af8, timestamp=time.time(), sequence_id=self._seq)
            self._seq += 1
            return chunk

        try:
            # LSL에서 chunk 단위로 받아 256샘플을 누적
            samples_af7 = []
            samples_af8 = []
            deadline = time.time() + timeout

            while len(samples_af7) < self.sample_rate:
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0:
                    return None

                max_need = self.sample_rate - len(samples_af7)
                chunk_samples, _ = self.inlet.pull_chunk(timeout=min(0.2, remaining), max_samples=max_need)
                if not chunk_samples:
                    continue

                for sample in chunk_samples:
                    if len(sample) < 2:
                        continue
                    # AF7, AF8가 첫 두 채널이라고 가정
                    samples_af7.append(float(sample[0]))
                    samples_af8.append(float(sample[1]))
                    if len(samples_af7) >= self.sample_rate:
                        break

            chunk = EEGChunk(af7=samples_af7[:self.sample_rate], af8=samples_af8[:self.sample_rate], timestamp=time.time(), sequence_id=self._seq)
            self._seq += 1
            return chunk
        except Exception as e:
            print(f"{self.name}: read_chunk 실패 - {e}")
            return None


# ============================================================
# 구현 예제 2: 파일 기반 (테스트/디버깅용)
# ============================================================
class FileReader(EEGReader):
    """CSV 파일에서 EEG 데이터 읽기 (테스트/재현용)"""
    
    def __init__(self, filepath: str, loop: bool = True):
        super().__init__("FileReader")
        self.filepath = filepath
        self.loop = loop
        self.df = None
        self.idx = 0
    
    def connect(self) -> bool:
        try:
            import pandas as pd
            self.df = pd.read_csv(self.filepath)
            
            # AF7, AF8 컬럼 확인
            if "AF7" not in self.df.columns or "AF8" not in self.df.columns:
                raise ValueError("CSV에 AF7, AF8 컬럼이 필요")
            
            self.is_connected = True
            print(f"✅ {self.name}: {self.filepath} 로드됨 ({len(self.df)} 샘플)")
            return True
        except Exception as e:
            print(f"❌ {self.name}: 로드 실패 - {e}")
            return False
    
    def disconnect(self):
        self.is_connected = False
        self.df = None
    
    def read_chunk(self, timeout: float = 1.0) -> Optional[EEGChunk]:
        if self.df is None or self.idx >= len(self.df):
            if self.loop:
                self.idx = 0
            else:
                return None
        
        # 256 샘플 (1초) 슬라이스
        end_idx = min(self.idx + 256, len(self.df))
        chunk_af7 = self.df["AF7"].iloc[self.idx:end_idx].tolist()
        chunk_af8 = self.df["AF8"].iloc[self.idx:end_idx].tolist()
        
        self.idx = end_idx
        
        return EEGChunk(
            af7=chunk_af7,
            af8=chunk_af8,
            timestamp=time.time(),
            sequence_id=self.chunk_count,
        )


# ============================================================
# 구현 예제 3: TCP 네트워크 (원격 MUSE2)
# ============================================================
class TCPReader(EEGReader):
    """
    TCP 소켓으로 EEG 데이터 수신.
    
    MUSE2 디바이스가 다른 시스템에 있을 경우, 
    네트워크로 데이터 전송.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        super().__init__("TCPReader")
        self.host = host
        self.port = port
        self.sock = None
    
    def connect(self) -> bool:
        try:
            import socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            print(f"✅ {self.name}: {self.host}:{self.port} 연결됨")
            return True
        except Exception as e:
            print(f"❌ {self.name}: 연결 실패 - {e}")
            return False
    
    def disconnect(self):
        if self.sock:
            self.sock.close()
        self.is_connected = False
    
    def read_chunk(self, timeout: float = 1.0) -> Optional[EEGChunk]:
        try:
            # JSON 형식으로 수신
            data_str = self.sock.recv(4096).decode()
            data = json.loads(data_str)
            return EEGChunk.from_dict(data)
        except Exception as e:
            print(f"⚠️ {self.name}: 수신 실패 - {e}")
            return None


# ============================================================
# 헬퍼: 리더 팩토리
# ============================================================
def create_reader(source: str, **kwargs) -> EEGReader:
    """
    소스 타입에 따라 적절한 리더 생성.
    
    Examples:
        reader = create_reader("muse2")
        reader = create_reader("file", filepath="data.csv")
        reader = create_reader("tcp", host="192.168.1.100", port=5000)
    """
    source = source.lower()
    
    if source == "muse2":
        return MUSE2Reader(**kwargs)
    elif source == "file":
        return FileReader(**kwargs)
    elif source == "tcp":
        return TCPReader(**kwargs)
    else:
        raise ValueError(f"알 수 없는 소스: {source}. "
                        "muse2/file/tcp 중 선택")


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("📋 EEG 데이터 소스 테스트")
    print("=" * 60)
    
    # 테스트: FileReader
    print("\n[1] FileReader 테스트 (CSV 파일)")
    try:
        reader = create_reader("file", filepath="test_data.csv")
        if reader.connect():
            for i in range(3):
                chunk = reader.read_chunk()
                if chunk:
                    print(f"  청크 {i+1}: AF7[0]={chunk.af7[0]:.1f}, "
                          f"AF8[0]={chunk.af8[0]:.1f}")
            reader.disconnect()
    except Exception as e:
        print(f"  ⚠️ 파일 없음 (정상): {e}")
    
    # 테스트: 인터페이스 검증
    print("\n[2] EEGChunk 직렬화 테스트")
    chunk = EEGChunk(
        af7=[1.0] * 256,
        af8=[2.0] * 256,
        timestamp=time.time(),
        metadata={"battery": 85}
    )
    chunk_dict = chunk.to_dict()
    chunk_restored = EEGChunk.from_dict(chunk_dict)
    print(f"  ✅ 직렬화/역직렬화 성공")
    print(f"     메타데이터: {chunk_restored.metadata}")
    
    print("\n✅ 테스트 완료")
