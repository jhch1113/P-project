import requests
import time
import json

BASE_URL = "http://localhost:8000"

def send_data(endpoint, data):
    url = f"{BASE_URL}{endpoint}"
    print(f"\n[POST] {url} 로 데이터 전송 중...")
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("성공적으로 데이터가 전송되었습니다!")
        else:
            print(f"오류 발생: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"서버에 연결할 수 없습니다: {e}")

def get_status():
    print("\n[GET] 현재 서버 최종 판정(Fusion) 결과:")
    try:
        res = requests.get(f"{BASE_URL}/metrics/fusion")
        print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"상태를 가져올 수 없습니다: {e}")

if __name__ == "__main__":
    print("=== Drowsiness Monitor 시뮬레이션 테스트 ===")
    print("먼저 브라우저에서 http://localhost:8000 에 접속해서 대시보드를 띄워주세요!")
    time.sleep(2)

    # 1. 초기 상태 확인
    get_status()
    time.sleep(2)

    # 2. 졸음(Drowsy) 데이터 주입
    print("\n--- 1단계: 카메라에 '졸음(Drowsy)' 상태를 강제 주입합니다 ---")
    camera_drowsy_data = {
        "ear": 0.05, "perclos": 80.0, "mar": 0.8, "pitch": 30.0, 
        "status": "Drowsy", "threshold": 0.25, "is_calibrated": True
    }
    send_data("/ingest/camera", camera_drowsy_data)
    get_status()
    time.sleep(4)

    # 3. 뇌파(EEG) 위험 데이터 주입
    print("\n--- 2단계: 뇌파(EEG)에도 '위험' 수치를 주입하여 다중모달 융합을 테스트합니다 ---")
    eeg_danger_data = {
        "alpha_beta_ratio": 5.0, "relative_theta": 0.5, "blink_rate": 5.0, 
        "is_connected": True, "signal_quality": 1.0
    }
    send_data("/ingest/eeg", eeg_danger_data)
    get_status()
    time.sleep(4)

    # 4. 정상 상태 복구
    print("\n--- 3단계: 다시 정상(Normal) 상태로 복구합니다 ---")
    camera_normal_data = {
        "ear": 0.35, "perclos": 0.0, "mar": 0.0, "pitch": 0.0, 
        "status": "Normal", "threshold": 0.25, "is_calibrated": True
    }
    send_data("/ingest/camera", camera_normal_data)
    get_status()
    
    print("\n테스트가 완료되었습니다. 대시보드 화면이 실시간으로 변하는 것을 확인하셨나요?")
