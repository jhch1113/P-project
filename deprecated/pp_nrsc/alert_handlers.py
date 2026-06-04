import threading
import subprocess
import platform
import time
import requests


def _play_sound(path: str) -> None:
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["afplay", path])
        else:
            import simpleaudio as sa
            wave_obj = sa.WaveObject.from_wave_file(path)
            wave_obj.play()
    except Exception as e:
        print(f"경고음 재생 실패: {e}")


def _send_server_alert(url: str, payload: dict) -> None:
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        print(f"원격 경고 전송 실패: {e}")


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    # support dict-like or object with attributes
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    except Exception:
        return default


def create_alert_callback(alert_config) -> 'callable | None':
    """생성된 콜백은 `score`(DrowsinessScore)를 받아 알림을 처리합니다.

    alert_config은 `config.alert` (dict 또는 객체) 입니다.
    """
    enabled = _get(alert_config, 'enabled', True)
    if not enabled:
        return None

    alarm_file = _get(alert_config, 'alarm_file', None)
    send_to_server = _get(alert_config, 'send_to_server', False)
    server_url = _get(alert_config, 'server_url', None)
    gpio_pin = _get(alert_config, 'gpio_pin', None)

    def _cb(score):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] ⚠️ ALERT: {score.risk_level} score={score.smoothed_score:.1f}")

        if alarm_file:
            threading.Thread(target=_play_sound, args=(alarm_file,), daemon=True).start()

        if send_to_server and server_url:
            payload = {
                'risk_level': score.risk_level,
                'score': float(score.smoothed_score),
                'timestamp': time.time(),
            }
            threading.Thread(target=_send_server_alert, args=(server_url, payload), daemon=True).start()

        if gpio_pin is not None:
            # GPIO 제어는 플랫폼/하드웨어 종속적이므로 여기에 구현하세요.
            print(f"GPIO 알림: 핀={gpio_pin} (구현 필요)")

    return _cb
