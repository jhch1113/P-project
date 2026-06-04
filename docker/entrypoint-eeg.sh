#!/bin/bash
# EEG 컨테이너 엔트리포인트
# 1. muselsl stream 을 백그라운드 '감독 루프'로 기동 → Muse2 BT 연결 + LSL 브로드캐스트
# 2. uvicorn(FastAPI) 기동 — pylsl 이 위 LSL 스트림을 수신

set -e

echo "[entrypoint-eeg] EEG_MODE=${EEG_MODE:-stub}"

# [FD Leak Fix] bleak 백엔드는 BLE 재연결을 반복할 때 파일 디스크립터/소켓을
# 누수시키는 경향이 있다. 누수가 1024(기본 한계)에 도달하면
# "OSError: [Errno 24] Too many open files"로 컨테이너 전체가 마비된다.
# 소프트 한계를 상향하여 1차 방어한다(하드 한계는 compose의 ulimits로 설정).
ulimit -n 65536 2>/dev/null || echo "[entrypoint-eeg] ulimit 상향 실패 (compose ulimits에 의존)."

# muselsl 감독 루프: 스트림이 끊기면 '프로세스를 통째로 재시작'한다.
# 프로세스 단위 재시작은 bleak가 누적한 FD/소켓을 매 사이클 완전히 해제하므로
# (1) FD 누수 누적, (2) <defunct> 좀비 프로세스 누적을 동시에 차단한다.
start_muselsl_supervisor() {
    local addr_args=""
    if [ -n "${MUSE_ADDRESS:-}" ]; then
        addr_args="--address ${MUSE_ADDRESS}"
    fi
    while true; do
        echo "[entrypoint-eeg] muselsl stream 시작 (backend=bleak ${addr_args})..."
        # 포그라운드 실행 → 이 서브셸이 직접 reap 하므로 좀비가 남지 않는다.
        muselsl stream --backend bleak ${addr_args} || true
        echo "[entrypoint-eeg] muselsl stream 종료/단절 — 5초 후 재연결 시도 (FD 정리됨)."
        sleep 5
    done
}

if [ "${EEG_MODE:-stub}" = "muse2" ]; then
    start_muselsl_supervisor &
    SUP_PID=$!
    echo "[entrypoint-eeg] muselsl 감독 프로세스 PID=${SUP_PID}. LSL 스트림 초기화 대기 (8초)..."
    sleep 8
    echo "[entrypoint-eeg] uvicorn 기동..."
else
    echo "[entrypoint-eeg] EEG_MODE=stub — muselsl stream 생략."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8002
