"""
EEG 데이터 검증 유틸리티
========================

MUSE2 팀과 Jetson 팀 간 데이터 무결성 확인.
통신 문제 디버깅에 사용.

사용:
    python validate_eeg.py --source file --file test_data.csv
    python validate_eeg.py --source tcp --host 192.168.1.100 --port 5000
"""

import argparse
import json
import time
from eeg_data_source import create_reader, EEGChunk


def validate_chunk(chunk: EEGChunk, chunk_count: int = 0) -> dict:
    """
    단일 EEGChunk 검증.
    
    Returns:
        검증 결과 dict
    """
    errors = []
    warnings = []
    
    # 1. 길이 확인
    if len(chunk.af7) != 256:
        errors.append(f"AF7 길이: {len(chunk.af7)} (기대: 256)")
    if len(chunk.af8) != 256:
        errors.append(f"AF8 길이: {len(chunk.af8)} (기대: 256)")
    
    # 2. 샘플 레이트 확인
    if chunk.sample_rate != 256:
        warnings.append(f"샘플 레이트: {chunk.sample_rate} (기대: 256)")
    
    # 3. 타임스탐프 확인
    if chunk.timestamp <= 0:
        errors.append(f"타임스탐프 이상: {chunk.timestamp}")
    
    # 4. Sequence ID 확인
    if chunk_count > 0 and chunk.sequence_id != chunk_count + 1:
        missing = chunk.sequence_id - (chunk_count + 1)
        if missing > 0:
            warnings.append(f"데이터 손실: {missing}개 청크 누락됨 "
                          f"({chunk_count} → {chunk.sequence_id})")
        else:
            warnings.append(f"Sequence ID 역순: {chunk_count} → {chunk.sequence_id}")
    
    # 5. 데이터 범위 확인 (-1000~1000 정도 정상)
    import numpy as np
    af7_arr = np.array(chunk.af7)
    af8_arr = np.array(chunk.af8)
    
    if np.any(np.abs(af7_arr) > 10000):
        warnings.append(f"AF7 범위 이상: [{af7_arr.min():.1f}, {af7_arr.max():.1f}]")
    if np.any(np.abs(af8_arr) > 10000):
        warnings.append(f"AF8 범위 이상: [{af8_arr.min():.1f}, {af8_arr.max():.1f}]")
    
    # 6. 메타데이터 확인
    if chunk.metadata:
        if "battery" in chunk.metadata and chunk.metadata["battery"] < 10:
            warnings.append(f"배터리 부족: {chunk.metadata['battery']}%")
        if "signal_quality" in chunk.metadata and chunk.metadata["signal_quality"] < 0.5:
            warnings.append(f"신호 품질 낮음: {chunk.metadata['signal_quality']:.2f}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "af7_range": [float(af7_arr.min()), float(af7_arr.max())],
        "af8_range": [float(af8_arr.min()), float(af8_arr.max())],
        "af7_mean": float(af7_arr.mean()),
        "af8_mean": float(af8_arr.mean()),
    }


def validate_source(source_type: str, duration: int = 30, **kwargs):
    """
    EEG 소스 연속 검증.
    
    Args:
        source_type: muse2, file, tcp
        duration: 검증 시간 (초)
        **kwargs: 소스별 파라미터
    """
    print(f"🔍 EEG 소스 검증 시작: {source_type} ({duration}초)")
    print("=" * 70)
    
    try:
        reader = create_reader(source_type, **kwargs)
    except Exception as e:
        print(f"❌ 리더 생성 실패: {e}")
        return
    
    if not reader.connect():
        print(f"❌ 연결 실패")
        return
    
    print(f"✅ 연결 성공: {reader.name}")
    print("-" * 70)
    
    stats = {
        "total_chunks": 0,
        "valid_chunks": 0,
        "invalid_chunks": 0,
        "errors": [],
        "warnings": [],
        "latencies": [],
    }
    
    start_time = time.time()
    last_seq_id = -1
    
    try:
        while time.time() - start_time < duration:
            recv_time = time.time()
            chunk = reader.read_chunk(timeout=2.0)
            latency_ms = (time.time() - recv_time) * 1000
            
            if chunk is None:
                print("⚠️ 타임아웃: 데이터 수신 없음")
                continue
            
            stats["total_chunks"] += 1
            stats["latencies"].append(latency_ms)
            
            # 검증
            result = validate_chunk(chunk, last_seq_id)
            last_seq_id = chunk.sequence_id
            
            if result["valid"]:
                stats["valid_chunks"] += 1
                status = "✅"
            else:
                stats["invalid_chunks"] += 1
                status = "❌"
                stats["errors"].extend(result["errors"])
            
            # 출력
            print(f"{status} [{stats['total_chunks']:3d}] "
                  f"AF7: [{result['af7_range'][0]:7.1f}, {result['af7_range'][1]:7.1f}] "
                  f"AF8: [{result['af8_range'][0]:7.1f}, {result['af8_range'][1]:7.1f}] "
                  f"latency={latency_ms:.1f}ms")
            
            # 경고 표시
            if result["warnings"]:
                for w in result["warnings"]:
                    print(f"  ⚠️  {w}")
    
    except KeyboardInterrupt:
        print("\n⏹️  중단됨")
    
    finally:
        reader.disconnect()
    
    # 최종 보고서
    print("=" * 70)
    print("📊 검증 결과:")
    print(f"  총 청크: {stats['total_chunks']}")
    print(f"  성공: {stats['valid_chunks']} ✅")
    print(f"  실패: {stats['invalid_chunks']} ❌")
    
    if stats['total_chunks'] > 0:
        success_rate = stats['valid_chunks'] / stats['total_chunks'] * 100
        print(f"  성공률: {success_rate:.1f}%")
        
        if stats['latencies']:
            import numpy as np
            latencies = np.array(stats['latencies'])
            print(f"  레이턴시: avg={latencies.mean():.1f}ms, "
                  f"min={latencies.min():.1f}ms, max={latencies.max():.1f}ms")
    
    if stats['errors']:
        print(f"  에러 목록:")
        for e in set(stats['errors']):
            count = stats['errors'].count(e)
            print(f"    - {e} (×{count})")


def test_roundtrip(filepath: str):
    """
    데이터 직렬화/역직렬화 테스트.
    """
    print("🔄 직렬화/역직렬화 테스트")
    
    # 1. 파일에서 읽기
    reader = create_reader("file", filepath=filepath)
    if not reader.connect():
        print(f"❌ 파일 로드 실패: {filepath}")
        return
    
    chunk = reader.read_chunk()
    if chunk is None:
        print("❌ 청크 읽기 실패")
        return
    
    # 2. JSON 직렬화
    chunk_dict = chunk.to_dict()
    chunk_json = json.dumps(chunk_dict)
    
    # 3. JSON 역직렬화
    chunk_dict2 = json.loads(chunk_json)
    chunk2 = EEGChunk.from_dict(chunk_dict2)
    
    # 4. 비교
    import numpy as np
    af7_eq = np.allclose(chunk.af7, chunk2.af7)
    af8_eq = np.allclose(chunk.af8, chunk2.af8)
    
    print(f"  AF7 일치: {af7_eq} ✅" if af7_eq else f"  AF7 일치: {af7_eq} ❌")
    print(f"  AF8 일치: {af8_eq} ✅" if af8_eq else f"  AF8 일치: {af8_eq} ❌")
    print(f"  타임스탐프: {chunk.timestamp} = {chunk2.timestamp}")
    print(f"  메타데이터: {chunk.metadata} = {chunk2.metadata}")
    
    if af7_eq and af8_eq:
        print("✅ 직렬화 성공")
    else:
        print("❌ 직렬화 실패")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG 데이터 검증")
    parser.add_argument("--source", choices=["file", "tcp", "muse2"],
                       default="file", help="데이터 소스")
    parser.add_argument("--file", help="CSV 파일 경로 (file 소스)")
    parser.add_argument("--host", default="localhost", help="TCP 호스트")
    parser.add_argument("--port", type=int, default=5000, help="TCP 포트")
    parser.add_argument("--duration", type=int, default=30, help="검증 시간 (초)")
    parser.add_argument("--roundtrip", help="직렬화 테스트 (파일 경로)")
    
    args = parser.parse_args()
    
    if args.roundtrip:
        test_roundtrip(args.roundtrip)
    else:
        kwargs = {}
        if args.source == "file":
            kwargs["filepath"] = args.file or "test_data.csv"
        elif args.source == "tcp":
            kwargs["host"] = args.host
            kwargs["port"] = args.port
        
        validate_source(args.source, duration=args.duration, **kwargs)
