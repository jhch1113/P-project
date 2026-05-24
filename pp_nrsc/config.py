"""
설정 파일 (config.yaml)
=======================

실시간 졸음운전 감지 시스템의 모든 설정을 중앙에서 관리.

사용:
    from config import load_config
    config = load_config("config.yaml")
"""

import yaml
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class EEGSourceConfig:
    """EEG 데이터 소스 설정"""
    type: str                    # muse2, file, tcp
    kwargs: Dict[str, Any]       # 리더별 파라미터


@dataclass
class APIConfig:
    """FastAPI 서버 설정"""
    url: str
    timeout: float
    apply_minmax: bool


@dataclass
class ScorerConfig:
    """점수화 로직 설정"""
    window_size: int
    drowsy_threshold: float
    instant_alert_threshold: float
    accumulated_time_limit: float


@dataclass
class AlertConfig:
    """경고 설정"""
    enabled: bool
    alarm_file: Optional[str]
    gpio_pin: Optional[int]
    send_to_server: bool
    server_url: Optional[str]


@dataclass
class UIConfig:
    """UI 설정"""
    display_type: str            # "opencv", "web", "headless"
    fps: int
    show_stats: bool


@dataclass
class SystemConfig:
    """전체 시스템 설정"""
    eeg_source: EEGSourceConfig
    api: APIConfig
    scorer: ScorerConfig
    alert: AlertConfig
    ui: UIConfig
    debug: bool


def load_config(filepath: str) -> SystemConfig:
    """YAML 설정 파일 로드"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return SystemConfig(
        eeg_source=EEGSourceConfig(
            type=data["eeg_source"]["type"],
            kwargs=data["eeg_source"].get("kwargs", {})
        ),
        api=APIConfig(
            url=data["api"]["url"],
            timeout=data["api"]["timeout"],
            apply_minmax=data["api"].get("apply_minmax", False)
        ),
        scorer=ScorerConfig(
            window_size=data["scorer"]["window_size"],
            drowsy_threshold=data["scorer"]["drowsy_threshold"],
            instant_alert_threshold=data["scorer"]["instant_alert_threshold"],
            accumulated_time_limit=data["scorer"]["accumulated_time_limit"]
        ),
        alert=AlertConfig(
            enabled=data["alert"]["enabled"],
            alarm_file=data["alert"].get("alarm_file"),
            gpio_pin=data["alert"].get("gpio_pin"),
            send_to_server=data["alert"].get("send_to_server", False),
            server_url=data["alert"].get("server_url")
        ),
        ui=UIConfig(
            display_type=data["ui"]["display_type"],
            fps=data["ui"]["fps"],
            show_stats=data["ui"].get("show_stats", True)
        ),
        debug=data.get("debug", False)
    )


def get_default_config() -> SystemConfig:
    """기본 설정 반환 (파일 없을 때)"""
    return SystemConfig(
        eeg_source=EEGSourceConfig(
            type="file",
            kwargs={"filepath": "test_data.csv", "loop": True}
        ),
        api=APIConfig(
            url="http://localhost:8000",
            timeout=5.0,
            apply_minmax=False
        ),
        scorer=ScorerConfig(
            window_size=30,
            drowsy_threshold=0.55,
            instant_alert_threshold=0.80,
            accumulated_time_limit=20.0
        ),
        alert=AlertConfig(
            enabled=True,
            alarm_file=None,
            gpio_pin=None,
            send_to_server=False,
            server_url=None
        ),
        ui=UIConfig(
            display_type="opencv",
            fps=10,
            show_stats=True
        ),
        debug=False
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 기본 설정 파일 생성
        filepath = sys.argv[1]
        config_dict = {
            "eeg_source": {
                "type": "file",  # muse2, file, tcp 중 선택
                "kwargs": {
                    "filepath": "test_data.csv",
                    "loop": True
                }
            },
            "api": {
                "url": "http://localhost:8000",
                "timeout": 5.0,
                "apply_minmax": False
            },
            "scorer": {
                "window_size": 30,
                "drowsy_threshold": 0.55,
                "instant_alert_threshold": 0.80,
                "accumulated_time_limit": 20.0
            },
            "alert": {
                "enabled": True,
                "alarm_file": None,
                "gpio_pin": None,
                "send_to_server": False,
                "server_url": None
            },
            "ui": {
                "display_type": "opencv",  # opencv, web, headless
                "fps": 10,
                "show_stats": True
            },
            "debug": False
        }
        
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 설정 파일 생성됨: {filepath}")
    else:
        config = get_default_config()
        print("기본 설정:")
        print(f"  EEG 소스: {config.eeg_source.type}")
        print(f"  API: {config.api.url}")
        print(f"  점수화 임계값: {config.scorer.drowsy_threshold}")
