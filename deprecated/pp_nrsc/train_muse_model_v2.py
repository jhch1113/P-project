"""
MUSE2 졸음운전 감지 모델 V2 - 향상된 학습 파이프라인
====================================================

개선 사항:
1. 강화된 전처리 (60Hz 노치 필터, EOG 아티팩트 감지)
2. 고급 특징 엔지니어링 (주파수 대역별 파워, 통계 특징)
3. 견고한 모델 아키텍처 (CNN-LSTM, Batch Norm, 정규화)
4. 데이터 증강 및 클래스 불균형 처리
5. 교차 검증 및 조기 종료

사용법:
  python train_muse_model_v2.py \
    --awake awake_study.csv awake_study2.csv awake_study3.csv awake_study4.csv \
    --sleep bedtime.csv bedtime2.csv \
    --output better_model.keras \
    --epochs 100
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, iirnotch, spectrogram, welch
from scipy.ndimage import uniform_filter1d
import argparse
from pathlib import Path
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 향상된 전처리 함수들
# ============================================================

FS = 256  # 샘플 레이트

# 글로벌 필터 계수 (성능 최적화)
_NYQ = 0.5 * FS
_BUTTER_B, _BUTTER_A = butter(4, [0.5 / _NYQ, 40 / _NYQ], btype="band")
_NOTCH_B, _NOTCH_A = iirnotch(60, 30, FS)  # 60Hz 노치 필터


def detect_eog_artifact(signal: np.ndarray, fs: int = FS, threshold: float = 2.5) -> np.ndarray:
    """
    EOG (눈 깜빡임) 아티팩트 감지
    
    눈 깜빡임은:
    - 매우 빠른 고진폭 변화
    - 일반적으로 1-5초의 짧은 지속시간
    
    Returns:
        Boolean 배열: True = 아티팩트 구간
    """
    # 미분으로 빠른 변화 감지
    diff = np.abs(np.diff(signal, prepend=signal[0]))
    
    # 롤링 표준편차 계산 (1초 윈도우)
    window = int(fs * 1.0)
    rolling_std = uniform_filter1d(diff, size=window, mode='nearest')
    
    # threshold 이상의 변화를 아티팩트로 표시
    artifact_mask = diff > (np.median(diff) + threshold * np.std(diff))
    
    # 아티팩트 근처 구간도 마스킹 (전파 효과)
    dilated_mask = np.zeros_like(artifact_mask, dtype=bool)
    dilation_size = int(fs * 0.5)  # 500ms 팽창
    for i in range(len(artifact_mask)):
        if artifact_mask[i]:
            dilated_mask[max(0, i-dilation_size):min(len(dilated_mask), i+dilation_size)] = True
    
    return dilated_mask


def detect_motion_artifact(signal: np.ndarray, fs: int = FS, threshold: float = 3.0) -> np.ndarray:
    """
    동작 아티팩트 (기기 움직임, 머리카락 접촉) 감지
    
    Returns:
        Boolean 배열: True = 아티팩트 구간
    """
    # 1초 윈도우별 RMS 계산
    window = int(fs * 1.0)
    rms_values = []
    for i in range(0, len(signal) - window, window // 2):
        window_signal = signal[i:i+window]
        rms = np.sqrt(np.mean(window_signal ** 2))
        rms_values.append(rms)
    
    rms_values = np.array(rms_values)
    rms_threshold = np.median(rms_values) + threshold * np.std(rms_values)
    
    # RMS가 높은 구간을 아티팩트로 표시
    artifact_mask = np.zeros_like(signal, dtype=bool)
    for i, rms_val in enumerate(rms_values):
        if rms_val > rms_threshold:
            start = i * window // 2
            end = min(start + window, len(signal))
            artifact_mask[start:end] = True
    
    return artifact_mask


def advanced_preprocess(x: np.ndarray, remove_artifacts: bool = True) -> np.ndarray:
    """
    향상된 전처리:
    1. 60Hz 노치 필터 (전력 주파수)
    2. 0.5-40Hz 밴드패스 필터
    3. 아티팩트 감지 및 제거
    4. 중앙값 제거 및 클리핑
    5. 정규화
    """
    x = np.asarray(x, dtype=np.float32)
    
    if x.ndim != 1:
        raise ValueError(f"1D 신호 필요. 받은 shape: {x.shape}")
    
    if len(x) < 33:
        raise ValueError(f"신호가 너무 짧음 (N={len(x)}). 최소 33 샘플 필요.")
    
    # 1. 60Hz 노치 필터
    x = filtfilt(_NOTCH_B, _NOTCH_A, x)
    
    # 2. 밴드패스 필터
    x = filtfilt(_BUTTER_B, _BUTTER_A, x)
    
    # 3. 아티팩트 감지 및 마스킹
    if remove_artifacts and len(x) > FS * 2:  # 최소 2초 필요
        eog_mask = detect_eog_artifact(x)
        motion_mask = detect_motion_artifact(x)
        artifact_mask = eog_mask | motion_mask
        
        # 아티팩트 구간의 값을 선형 보간으로 대체
        if np.any(artifact_mask):
            valid_idx = np.where(~artifact_mask)[0]
            if len(valid_idx) > 1:
                x[artifact_mask] = np.interp(
                    np.where(artifact_mask)[0],
                    valid_idx,
                    x[valid_idx]
                )
    
    # 4. 중앙값 제거
    x = x - np.median(x)
    
    # 5. 클리핑 및 정규화
    x = np.clip(x, -100, 100)
    x_mean = x.mean()
    x_std = x.std() + 1e-6
    x = (x - x_mean) / x_std
    
    return x.astype(np.float32)


# ============================================================
# 2. 향상된 특징 엔지니어링
# ============================================================

def extract_advanced_features(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    고급 특징 추출:
    - 주파수 대역별 파워 (Delta, Theta, Alpha, Beta, Gamma)
    - 파워 스펙트럼 밀도 통계
    - 시간 영역 통계 특징
    - 신호 복잡도 지표
    """
    features = []
    
    # 1. 주파수 대역 정의 (Hz)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40),
    }
    
    # 2. Welch 파워 스펙트럼 계산
    freqs, pxx = welch(signal, fs, nperseg=min(256, len(signal)))
    
    # 3. 대역별 파워 추출
    total_power = np.sum(pxx)
    for band_name, (f_low, f_high) in bands.items():
        band_power = np.sum(pxx[(freqs >= f_low) & (freqs <= f_high)])
        band_power_ratio = band_power / (total_power + 1e-9)
        features.append(band_power)
        features.append(band_power_ratio)
    
    # 4. 파워 스펙트럼 통계
    features.append(np.max(pxx))
    features.append(np.mean(pxx))
    features.append(np.std(pxx))
    
    # 5. 시간 영역 특징
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.max(signal) - np.min(signal))  # 범위
    
    # 6. 통계 고차 모멘트
    features.append(np.mean(np.abs(np.diff(signal))))  # 평균 절대 변화
    features.append(np.sum(signal ** 2))  # 에너지
    
    # 7. 엔트로피 (신호 복잡도)
    # Approximate entropy
    m = 2
    r = 0.2 * np.std(signal)
    def _apprx_entropy(u, m, r):
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[u[j] for j in range(i, i + m - 1 + 1)] for i in range(len(u) - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (len(u) - m + 1.0) for x_i in x]
            return (len(u) - m + 1.0) ** (-1) * sum(np.log(C))
        
        return abs(_phi(m + 1) - _phi(m))
    
    try:
        features.append(_apprx_entropy(signal, m, r))
    except:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# ============================================================
# 3. 데이터 로딩 및 전처리
# ============================================================

def load_and_preprocess_csv(csv_files: list, label: int, 
                           window_size: int = 1280,  # 5초
                           stride: int = 256,        # 1초
                           remove_artifacts: bool = True,
                           sample_size: int = None) -> tuple:
    """
    CSV 파일들을 로드하고 윈도우별로 전처리
    
    Args:
        csv_files: CSV 파일 경로 리스트
        label: 0 (awake) 또는 1 (sleep)
        window_size: 윈도우 크기 (샘플)
        stride: 스트라이드 (샘플)
        remove_artifacts: 아티팩트 제거 여부
    
    Returns:
        (windows, labels): (N, window_size, 2) 배열과 레이블
    """
    windows = []
    labels = []
    
    for csv_file in csv_files:
        print(f"  로딩: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"    ❌ 파일 없음")
            continue
        
        df = pd.read_csv(csv_file)
        
        # AF7, AF8 채널 추출
        af7 = df['AF7'].values.astype(np.float32)
        af8 = df['AF8'].values.astype(np.float32)
        
        print(f"    샘플: {len(af7)}, 지속시간: {len(af7)/FS:.1f}초")
        
        # 윈도우 생성
        for start in range(0, len(af7) - window_size, stride):
            end = start + window_size
            
            # 각 채널 전처리
            af7_proc = advanced_preprocess(af7[start:end], remove_artifacts)
            af8_proc = advanced_preprocess(af8[start:end], remove_artifacts)
            
            # 2D 배열로 결합
            window_data = np.stack([af7_proc, af8_proc], axis=1)
            
            # 특징 추출
            af7_features = extract_advanced_features(af7_proc)
            af8_features = extract_advanced_features(af8_proc)
            
            # 특징 결합
            combined_features = np.concatenate([af7_features, af8_features])
            
            windows.append(combined_features)
            labels.append(label)
    
    # 샘플 크기 제한 (속도 향상용)
    if sample_size is not None and len(windows) > sample_size:
        indices = np.random.choice(len(windows), sample_size, replace=False)
        windows = [windows[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)


# ============================================================
# 4. 데이터 증강
# ============================================================

def augment_data(X: np.ndarray, y: np.ndarray, 
                factor: float = 0.1) -> tuple:
    """
    데이터 증강: 가우시안 노이즈 추가
    """
    X_aug = [X]
    y_aug = [y]
    
    # Muse2 노이즈 특성을 반영한 노이즈 추가
    noise_std = factor * np.std(X)
    X_noisy = X + np.random.normal(0, noise_std, X.shape)
    
    X_aug.append(X_noisy)
    y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)


# ============================================================
# 5. 모델 구축
# ============================================================

def build_model(input_shape: int) -> keras.Model:
    """
    견고한 신경망 모델 구축
    
    아키텍처:
    - Dense layers with BatchNormalization
    - Dropout for regularization
    - L2 regularization
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        
        # Block 1
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Block 4
        layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


# ============================================================
# 6. 메인 학습 루프
# ============================================================

def main(args):
    print("=" * 70)
    print("MUSE2 졸음운전 모델 V2 - 고급 학습 파이프라인")
    print("=" * 70)
    
    # 데이터 로딩
    print("\n📊 데이터 로딩 및 전처리...")
    print("깨어있는 상태 데이터:")
    X_awake, y_awake = load_and_preprocess_csv(
        args.awake, 
        label=0,
        remove_artifacts=True,
        sample_size=args.sample_size
    )
    
    print("수면 상태 데이터:")
    X_sleep, y_sleep = load_and_preprocess_csv(
        args.sleep,
        label=1,
        remove_artifacts=True,
        sample_size=args.sample_size
    )
    
    # 데이터 결합
    X = np.vstack([X_awake, X_sleep])
    y = np.hstack([y_awake, y_sleep])
    
    print(f"\n✅ 로드된 샘플: {len(X)}")
    print(f"  - Awake: {np.sum(y == 0)}")
    print(f"  - Sleep: {np.sum(y == 1)}")
    print(f"  - 특징 차원: {X.shape[1]}")
    
    # 클래스 불균형 확인 및 샘플링
    awake_count = np.sum(y == 0)
    sleep_count = np.sum(y == 1)
    
    if awake_count > sleep_count * 2:
        print(f"\n⚖️  클래스 불균형 감지. Awake 샘플 언더샘플링...")
        awake_idx = np.where(y == 0)[0]
        sleep_idx = np.where(y == 1)[0]
        
        # Awake 샘플을 sleep 샘플의 1.5배로 제한
        target_awake = int(sleep_count * 1.5)
        selected_awake = np.random.choice(awake_idx, target_awake, replace=False)
        
        selected_idx = np.concatenate([selected_awake, sleep_idx])
        X = X[selected_idx]
        y = y[selected_idx]
        
        print(f"  - 조정 후 Awake: {np.sum(y == 0)}")
        print(f"  - 조정 후 Sleep: {np.sum(y == 1)}")
    
    # 데이터 증강 (선택사항)
    if args.augment:
        print(f"\n🎲 데이터 증강 (노이즈 추가)...")
        X, y = augment_data(X, y, factor=0.05)
        print(f"  - 증강 후 샘플: {len(X)}")
    
    # 학습/검증/테스트 분할
    print("\n🔀 데이터 분할...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  - Train: {len(X_train)}")
    print(f"  - Validation: {len(X_val)}")
    print(f"  - Test: {len(X_test)}")
    
    # 정규화
    print("\n🔧 정규화...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 클래스 가중치 계산
    class_weight = {
        0: len(y_train) / (2 * np.sum(y_train == 0) + 1e-6),
        1: len(y_train) / (2 * np.sum(y_train == 1) + 1e-6),
    }
    print(f"  - 클래스 가중치: {class_weight}")
    
    # 모델 구축
    print("\n🏗️  모델 구축...")
    model = build_model(X_train.shape[1])
    model.summary()
    
    # 컴파일
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    # 콜백
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # 학습
    print("\n🚀 모델 학습...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n📈 모델 평가...")
    train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUC={train_auc:.4f}")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
    print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}")
    
    # 모델 저장
    print(f"\n💾 모델 저장: {args.output}")
    model.save(args.output)
    
    # 전처리 정보 저장
    import pickle
    scaler_path = args.output.replace('.keras', '_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler 저장: {scaler_path}")
    
    print("\n✅ 완료!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MUSE2 모델 V2 학습')
    parser.add_argument('--awake', nargs='+', required=True,
                       help='깨어있는 상태 CSV 파일들')
    parser.add_argument('--sleep', nargs='+', required=True,
                       help='수면 상태 CSV 파일들')
    parser.add_argument('--output', default='muse_model_v2.keras',
                       help='출력 모델 경로')
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에포크')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--augment', action='store_true',
                       help='데이터 증강 활성화')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='각 파일당 최대 샘플 수 (속도 향상용)')
    
    args = parser.parse_args()
    main(args)
