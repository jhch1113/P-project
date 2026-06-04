"""
정확도 비교 테스트
=================
기본 전처리 vs 고급 전처리 정확도 비교

사용법:
  python test_accuracy_improvement.py
  python test_accuracy_improvement.py --aggressive
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# 모델 경로 설정
os.environ["MUSE_MODEL_PATH"] = "dummy_model.keras"

# 모듈 임포트
from muse_inference_api import preprocess, make_windows, predict_windows, minmax_scale, hysteresis, get_model
from advanced_preprocessing import AdvancedPreprocessor
from advanced_postprocessing import AdvancedPostprocessor

FS = 256
SEQ_LEN = FS * 5  # 1280
STRIDE = FS        # 256


def load_csv_data(csv_file: str, label: int) -> tuple:
    """CSV 파일에서 AF7, AF8 로드 및 윈도우 생성"""
    if not os.path.exists(csv_file):
        print(f"  ❌ 파일 없음: {csv_file}")
        return None, None
    
    df = pd.read_csv(csv_file)
    af7 = df['AF7'].values.astype(np.float32)
    af8 = df['AF8'].values.astype(np.float32)
    
    # (N, 2) 형태로 스택
    raw = np.stack([af7, af8], axis=1).astype(np.float32)
    
    # 윈도우 생성
    windows = []
    labels = []
    
    for start in range(0, len(raw) - SEQ_LEN, STRIDE):
        end = start + SEQ_LEN
        window = raw[start:end]
        windows.append(window)
        labels.append(label)
    
    return np.array(windows), np.array(labels)


def test_basic_preprocessing(X_test: np.ndarray, y_test: np.ndarray, 
                            hys_high: float = 0.55, hys_low: float = 0.45) -> dict:
    """기본 전처리 사용한 테스트"""
    print("\n📊 [기본 전처리] 테스트 중...")
    
    # 전처리
    X_prep = []
    for x in X_test:
        try:
            x_pre, _ = preprocess(x, use_advanced=False)
            X_prep.append(x_pre)
        except Exception as e:
            print(f"  ⚠️  전처리 실패: {e}")
            continue
    
    X_prep = np.array(X_prep)
    
    if len(X_prep) == 0:
        return None
    
    # 추론
    preds = predict_windows(X_prep)
    preds_scaled = minmax_scale(preds)
    states = hysteresis(preds_scaled, hys_high, hys_low)
    
    # 메트릭 계산
    y_subset = y_test[:len(states)]
    
    acc = accuracy_score(y_subset, states)
    prec = precision_score(y_subset, states, zero_division=0)
    rec = recall_score(y_subset, states, zero_division=0)
    f1 = f1_score(y_subset, states, zero_division=0)
    
    try:
        auc = roc_auc_score(y_subset, preds[:len(states)])
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_subset, states)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': preds[:len(states)],
        'states': states,
        'n_samples': len(states),
    }


def test_advanced_preprocessing(X_test: np.ndarray, y_test: np.ndarray,
                               hys_high: float = 0.55, hys_low: float = 0.45,
                               aggressive: bool = False) -> dict:
    """고급 전처리 사용한 테스트"""
    print(f"\n🚀 [고급 전처리] 테스트 중... (aggressive={aggressive})")
    
    # 전처리
    preprocessor = AdvancedPreprocessor(verbose=False)
    X_prep = []
    quality_scores = []
    
    for x in X_test:
        try:
            x_pre, quality = preprocess(x, use_advanced=True)
            X_prep.append(x_pre)
            quality_scores.append(quality)
        except Exception as e:
            print(f"  ⚠️  전처리 실패: {e}")
            continue
    
    X_prep = np.array(X_prep)
    quality_scores = np.array(quality_scores)
    
    if len(X_prep) == 0:
        return None
    
    # 추론
    preds = predict_windows(X_prep)
    preds_scaled = minmax_scale(preds)
    
    # 품질을 고려한 히스테리시스
    postprocessor = AdvancedPostprocessor(history_size=5, confidence_threshold=0.5)
    states = []
    confidences = []
    
    for p_scaled, q in zip(preds_scaled, quality_scores):
        state, conf, _ = postprocessor.process(p_scaled, q, hys_high, hys_low)
        states.append(state)
        confidences.append(conf)
    
    states = np.array(states)
    
    # 메트릭 계산
    y_subset = y_test[:len(states)]
    
    acc = accuracy_score(y_subset, states)
    prec = precision_score(y_subset, states, zero_division=0)
    rec = recall_score(y_subset, states, zero_division=0)
    f1 = f1_score(y_subset, states, zero_division=0)
    
    try:
        auc = roc_auc_score(y_subset, preds[:len(states)])
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_subset, states)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': preds[:len(states)],
        'states': states,
        'quality_scores': quality_scores,
        'confidences': confidences,
        'n_samples': len(states),
    }


def print_results(name: str, results: dict):
    """결과 출력"""
    if results is None:
        print(f"  ❌ {name}: 실패")
        return
    
    print(f"\n{'='*60}")
    print(f"📈 {name}")
    print(f"{'='*60}")
    print(f"  정확도 (Accuracy):  {results['accuracy']*100:6.2f}%")
    print(f"  정밀도 (Precision): {results['precision']*100:6.2f}%")
    print(f"  재현율 (Recall):    {results['recall']*100:6.2f}%")
    print(f"  F1 스코어:         {results['f1']*100:6.2f}%")
    print(f"  AUC 점수:          {results['auc']:.4f}")
    print(f"  샘플 수:           {results['n_samples']}")
    
    cm = results['confusion_matrix']
    print(f"\n  혼동 행렬 (Confusion Matrix):")
    print(f"    예측\\실제  |  Awake(0)  |  Sleep(1)")
    print(f"    ──────────|────────────|──────────")
    print(f"    Awake(0)  |    {cm[0,0]:6d}    |   {cm[0,1]:6d}")
    print(f"    Sleep(1)  |    {cm[1,0]:6d}    |   {cm[1,1]:6d}")


def test_ensemble_preprocessing(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """앙상블 전처리 테스트 - 여러 방법 결합"""
    print(f"\n🎭 [앙상블 전처리] 테스트 중...")
    
    # 여러 전처리 방법으로 예측
    predictions = []
    
    # 방법 1: 기본 전처리
    X_prep1 = []
    for x in X_test:
        x_pre, _ = preprocess(x, use_advanced=False)
        X_prep1.append(x_pre)
    X_prep1 = np.array(X_prep1)
    preds1 = predict_windows(X_prep1)
    preds1_scaled = minmax_scale(preds1)
    states1 = hysteresis(preds1_scaled, 0.6, 0.4)
    predictions.append(states1)
    
    # 방법 2: 고급 전처리 (일반)
    X_prep2 = []
    quality_scores = []
    for x in X_test:
        x_pre, quality = preprocess(x, use_advanced=True)
        X_prep2.append(x_pre)
        quality_scores.append(quality)
    X_prep2 = np.array(X_prep2)
    quality_scores = np.array(quality_scores)
    
    preds2 = predict_windows(X_prep2)
    preds2_scaled = minmax_scale(preds2)
    
    postprocessor = AdvancedPostprocessor(history_size=5, confidence_threshold=0.5)
    states2 = []
    for p_scaled, q in zip(preds2_scaled, quality_scores):
        state, _, _ = postprocessor.process(p_scaled, q, 0.65, 0.35)
        states2.append(state)
    states2 = np.array(states2)
    predictions.append(states2)
    
    # 방법 3: 고급 전처리 (aggressive)
    X_prep3 = []
    quality_scores3 = []
    for x in X_test:
        x_pre, quality = preprocess(x, use_advanced=True)
        X_prep3.append(x_pre)
        quality_scores3.append(quality)
    X_prep3 = np.array(X_prep3)
    quality_scores3 = np.array(quality_scores3)
    
    preds3 = predict_windows(X_prep3)
    preds3_scaled = minmax_scale(preds3)
    
    postprocessor3 = AdvancedPostprocessor(history_size=5, confidence_threshold=0.5)
    states3 = []
    for p_scaled, q in zip(preds3_scaled, quality_scores3):
        state, _, _ = postprocessor3.process(p_scaled, q, 0.7, 0.3)
        states3.append(state)
    states3 = np.array(states3)
    predictions.append(states3)
    
    # 앙상블: 다수결 투표
    predictions = np.array(predictions)  # (3, N)
    ensemble_states = []
    ensemble_confidences = []
    
    for i in range(len(X_test)):
        votes = predictions[:, i]
        # 다수결
        if np.sum(votes == 1) > np.sum(votes == 0):
            final_state = 1
        else:
            final_state = 0
        
        # 신뢰도: 일치도
        agreement = np.sum(votes == final_state) / len(votes)
        ensemble_states.append(final_state)
        ensemble_confidences.append(agreement)
    
    ensemble_states = np.array(ensemble_states)
    
    # 메트릭 계산
    y_subset = y_test[:len(ensemble_states)]
    
    acc = accuracy_score(y_subset, ensemble_states)
    prec = precision_score(y_subset, ensemble_states, zero_division=0)
    rec = recall_score(y_subset, ensemble_states, zero_division=0)
    f1 = f1_score(y_subset, ensemble_states, zero_division=0)
    
    try:
        auc = roc_auc_score(y_subset, preds2[:len(ensemble_states)])  # 대표 AUC 사용
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_subset, ensemble_states)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': preds2[:len(ensemble_states)],
        'states': ensemble_states,
        'confidences': ensemble_confidences,
        'n_samples': len(ensemble_states),
    }
    """그리드 서치로 최적의 히스테리시스 임계값 찾기"""
    print(f"\n🔍 최적 임계값 탐색 중... (고급={use_advanced}, aggressive={aggressive})")
    
    best_f1 = 0
    best_high = 0.5
    best_low = 0.5
    
    # 그리드 서치 범위
    high_range = np.arange(0.4, 0.8, 0.05)
    low_range = np.arange(0.2, 0.6, 0.05)
    
    for high in high_range:
        for low in low_range:
            if low >= high:
                continue
                
            try:
                if use_advanced:
                    # 고급 전처리 테스트
                    X_prep = []
                    quality_scores = []
                    
                    for x in X_test[:100]:  # 샘플로 테스트 (속도 위해)
                        x_pre, quality = preprocess(x, use_advanced=True)
                        X_prep.append(x_pre)
                        quality_scores.append(quality)
                    
                    X_prep = np.array(X_prep)
                    quality_scores = np.array(quality_scores)
                    
                    preds = predict_windows(X_prep)
                    preds_scaled = minmax_scale(preds)
                    
                    postprocessor = AdvancedPostprocessor(history_size=5, confidence_threshold=0.5)
                    states = []
                    
                    for p_scaled, q in zip(preds_scaled, quality_scores):
                        state, _, _ = postprocessor.process(p_scaled, q, high, low)
                        states.append(state)
                    
                    states = np.array(states)
                    y_subset = y_test[:len(states)]
                    
                else:
                    # 기본 전처리 테스트
                    X_prep = []
                    for x in X_test[:100]:  # 샘플로 테스트
                        x_pre, _ = preprocess(x, use_advanced=False)
                        X_prep.append(x_pre)
                    
                    X_prep = np.array(X_prep)
                    preds = predict_windows(X_prep)
                    preds_scaled = minmax_scale(preds)
                    states = hysteresis(preds_scaled, high, low)
                    y_subset = y_test[:len(states)]
                
                f1 = f1_score(y_subset, states, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_high = high
                    best_low = low
                    
            except Exception as e:
                continue
    
    print(f"  ✅ 최적 임계값: high={best_high:.2f}, low={best_low:.2f}, F1={best_f1:.3f}")
    return best_high, best_low


def compare_results(basic: dict, advanced: dict):
    """결과 비교"""
    if basic is None or advanced is None:
        print("\n❌ 테스트 실패: 비교할 수 없음")
        return
    
    print(f"\n{'='*60}")
    print("🎯 개선 효과 비교")
    print(f"{'='*60}")
    
    acc_diff = (advanced['accuracy'] - basic['accuracy']) * 100
    prec_diff = (advanced['precision'] - basic['precision']) * 100
    rec_diff = (advanced['recall'] - basic['recall']) * 100
    f1_diff = (advanced['f1'] - basic['f1']) * 100
    
    print(f"  정확도:  {basic['accuracy']*100:6.2f}% → {advanced['accuracy']*100:6.2f}% ({acc_diff:+6.2f}%p)")
    print(f"  정밀도:  {basic['precision']*100:6.2f}% → {advanced['precision']*100:6.2f}% ({prec_diff:+6.2f}%p)")
    print(f"  재현율:  {basic['recall']*100:6.2f}% → {advanced['recall']*100:6.2f}% ({rec_diff:+6.2f}%p)")
    print(f"  F1:     {basic['f1']*100:6.2f}% → {advanced['f1']*100:6.2f}% ({f1_diff:+6.2f}%p)")
    
    if acc_diff > 0:
        print(f"\n  ✅ 정확도 향상: {acc_diff:.2f}%p")
    else:
        print(f"\n  ⚠️  정확도 저하: {acc_diff:.2f}%p")
    
    print(f"\n  샘플 분포:")
    print(f"    기본:   Awake={np.sum(basic['states']==0)}, Sleep={np.sum(basic['states']==1)}")
    print(f"    고급:   Awake={np.sum(advanced['states']==0)}, Sleep={np.sum(advanced['states']==1)}")


def main(args):
    print("\n" + "="*60)
    print("🔬 Muse2 정확도 개선 테스트")
    print("="*60)
    
    # 테스트 데이터 로드
    print("\n📂 테스트 데이터 로드...")
    
    # 여러 awake 파일 로드
    awake_files = ['awake_study.csv', 'awake_study2.csv', 'awake_study3.csv', 'awake_study4.csv']
    X_awake_list = []
    y_awake_list = []
    
    for file in awake_files:
        X, y = load_csv_data(file, label=0)
        if X is not None:
            X_awake_list.append(X)
            y_awake_list.append(y)
    
    # 여러 sleep 파일 로드
    sleep_files = ['bedtime.csv', 'bedtime2.csv']
    X_sleep_list = []
    y_sleep_list = []
    
    for file in sleep_files:
        X, y = load_csv_data(file, label=1)
        if X is not None:
            X_sleep_list.append(X)
            y_sleep_list.append(y)
    
    if not X_awake_list or not X_sleep_list:
        print("\n❌ 테스트 데이터를 찾을 수 없습니다.")
        print("   awake_study.csv, bedtime.csv 등의 파일이 필요합니다.")
        return
    
    # 데이터 결합
    X_awake = np.vstack(X_awake_list) if len(X_awake_list) > 1 else X_awake_list[0]
    y_awake = np.hstack(y_awake_list) if len(y_awake_list) > 1 else y_awake_list[0]
    X_sleep = np.vstack(X_sleep_list) if len(X_sleep_list) > 1 else X_sleep_list[0]
    y_sleep = np.hstack(y_sleep_list) if len(y_sleep_list) > 1 else y_sleep_list[0]
    
    # 데이터 균형 맞추기 (각 클래스에서 같은 수의 샘플 사용)
    min_samples = min(len(X_awake), len(X_sleep))
    print(f"  - 데이터 균형 맞추기: 각 클래스당 {min_samples} 샘플 사용")
    
    # 랜덤하게 샘플링하여 균형 맞추기
    np.random.seed(42)  # 재현성을 위해
    awake_indices = np.random.choice(len(X_awake), min_samples, replace=False)
    sleep_indices = np.random.choice(len(X_sleep), min_samples, replace=False)
    
    X_awake_balanced = X_awake[awake_indices]
    y_awake_balanced = y_awake[awake_indices]
    X_sleep_balanced = X_sleep[sleep_indices]
    y_sleep_balanced = y_sleep[sleep_indices]
    
    X_test = np.vstack([X_awake_balanced, X_sleep_balanced])
    y_test = np.hstack([y_awake_balanced, y_sleep_balanced])
    
    print(f"✅ 데이터 로드 완료")
    print(f"  - Awake 윈도우: {len(X_awake)}")
    print(f"  - Sleep 윈도우: {len(X_sleep)}")
    print(f"  - 총 윈도우: {len(X_test)}")
    
    # 모델 로드 확인
    try:
        model = get_model()
        print(f"✅ 모델 로드: {model.name if hasattr(model, 'name') else 'MUSE Model'}")
    except Exception as e:
        print(f"⚠️  모델 로드 실패: {e}")
        print("   기본 모델 경로를 확인하세요: /mnt/user-data/uploads/MUSE_activity_model.keras")
        return
    
    # 테스트 실행
    print("\n" + "="*60)
    print("🧪 테스트 실행")
    print("="*60)
    
    # 기본 전처리 - 더 나은 임계값 사용
    basic_results = test_basic_preprocessing(X_test, y_test, 
                                            hys_high=0.6, hys_low=0.4)
    print_results("기본 전처리 (Basic Preprocessing)", basic_results)
    
    # 고급 전처리 - 더 나은 임계값 사용
    advanced_results = test_advanced_preprocessing(X_test, y_test,
                                                  hys_high=0.65, hys_low=0.35,
                                                  aggressive=True)
    print_results("고급 전처리 (Advanced Preprocessing)", advanced_results)
    
    # 앙상블 전처리
    ensemble_results = test_ensemble_preprocessing(X_test, y_test)
    print_results("앙상블 전처리 (Ensemble Preprocessing)", ensemble_results)
    
    # 비교
    if basic_results is not None and advanced_results is not None and ensemble_results is not None:
        compare_results(basic_results, advanced_results)
        print(f"\n{'='*60}")
        print("🎯 앙상블 vs 고급 전처리 비교")
        print(f"{'='*60}")
        
        acc_diff = (ensemble_results['accuracy'] - advanced_results['accuracy']) * 100
        prec_diff = (ensemble_results['precision'] - advanced_results['precision']) * 100
        rec_diff = (ensemble_results['recall'] - advanced_results['recall']) * 100
        f1_diff = (ensemble_results['f1'] - advanced_results['f1']) * 100
        
        print(f"  정확도:  {advanced_results['accuracy']*100:6.2f}% → {ensemble_results['accuracy']*100:6.2f}% ({acc_diff:+6.2f}%p)")
        print(f"  정밀도:  {advanced_results['precision']*100:6.2f}% → {ensemble_results['precision']*100:6.2f}% ({prec_diff:+6.2f}%p)")
        print(f"  재현율:  {advanced_results['recall']*100:6.2f}% → {ensemble_results['recall']*100:6.2f}% ({rec_diff:+6.2f}%p)")
        print(f"  F1:     {advanced_results['f1']*100:6.2f}% → {ensemble_results['f1']*100:6.2f}% ({f1_diff:+6.2f}%p)")
        
        if acc_diff > 0:
            print(f"\n  ✅ 앙상블 향상: {acc_diff:.2f}%p")
        else:
            print(f"\n  ⚠️  앙상블 저하: {acc_diff:.2f}%p")
    
    print("\n" + "="*60)
    print("✅ 테스트 완료")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Muse2 정확도 개선 테스트')
    parser.add_argument('--aggressive', action='store_true',
                       help='공격적 전처리 사용 (더 많은 노이즈 제거)')
    
    args = parser.parse_args()
    main(args)
