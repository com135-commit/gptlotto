"""
Stacking 모델 재학습 스크립트 (57개 특징)
GUI 없이 Stacking 앙상블 모델 학습
"""
import pickle
import numpy as np
import os
import time
from lotto_history import load_history_csv
from lotto_main import StackingModelWrapper
from lotto_generators import (
    generate_biased_combinations,
    _compute_core_features_batch,
    _compute_history_features_batch,
    _compute_temporal_features_batch,
    _prepare_history_array
)

print("=" * 80)
print("Stacking 앙상블 학습 (57개 특징: 39 코어 + 11 히스토리 + 7 시간)")
print("=" * 80)

# 1. 히스토리 로드
print("\n[1] CSV 로드")
history_df = load_history_csv("lotto.csv")
print(f"   데이터: {len(history_df)}회")

# ===========================
# 2. K-Fold 앙상블 학습 (25개 모델)
# ===========================
print("\n[2] K-Fold 앙상블 학습 (25개 모델)")

# 학습 데이터 준비 (시간 정보 포함)
pos_sets = []
pos_meta = []  # (round, date) 시간 정보 저장

# DataFrame 컬럼 확인
has_round = 'round' in history_df.columns
has_date = 'date' in history_df.columns

for idx, row in history_df.iterrows():
    # round와 date 정보 추출 (있으면)
    round_num = row['round'] if has_round else None
    date_str = row['date'] if has_date else None

    # n1~n6 컬럼 값 추출
    nums = []
    for col in history_df.columns:
        if col.lower() in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
            try:
                val = int(row[col])
                if 1 <= val <= 45:
                    nums.append(val)
            except (ValueError, TypeError):
                pass

    if len(nums) == 6:
        pos_sets.append(sorted(nums))
        pos_meta.append((round_num, date_str))

# 음성 샘플: 편향된 조합 생성
n_neg = len(pos_sets) * 5
neg_sets = generate_biased_combinations(n_neg)

# 특징 추출 (⚡ Numba 병렬 처리)
print(f"   [특징 추출] 57개 고급 특징 (39 코어 + 11 히스토리 + 7 시간)")
print(f"   [Numba+fastmath] 첫 실행 시 컴파일... (2-3초 소요)")

start_time = time.time()

# 히스토리 데이터를 numpy 배열로 변환 (한 번만)
print(f"   [전처리] 히스토리 데이터 변환...")
history_arr = _prepare_history_array(history_df)
print(f"        → 완료! ({len(history_arr)}회 히스토리)")

# 핵심 특징 추출 (CPU 병렬)
print(f"   [1/3] 핵심 특징 추출 (배치 {len(pos_sets) + len(neg_sets)}개, 병렬 처리)...")
pos_sets_arr = np.array(pos_sets, dtype=np.float64)
neg_sets_arr = np.array(neg_sets, dtype=np.float64)

core_features_pos = _compute_core_features_batch(pos_sets_arr)
core_features_neg = _compute_core_features_batch(neg_sets_arr)
core_time = time.time() - start_time
print(f"        → 완료! ({core_time:.1f}초)")

# 히스토리 특징 추출 (CPU 병렬)
print(f"   [2/3] 히스토리 특징 추출 (배치 {len(pos_sets) + len(neg_sets)}개, 병렬 처리)...")
hist_start = time.time()
hist_features_pos = _compute_history_features_batch(pos_sets_arr, history_arr)
hist_features_neg = _compute_history_features_batch(neg_sets_arr, history_arr)
hist_time = time.time() - hist_start
print(f"        → 완료! ({hist_time:.1f}초)")

# 시간 특징 추출 (양성 샘플만 시간 정보 있음)
print(f"   [3/3] 시간 특징 추출...")
temp_start = time.time()

# 양성 샘플: 각 샘플마다 실제 시간 정보 사용
temporal_features_pos_list = []
for i in range(len(pos_sets)):
    round_num, date_str = pos_meta[i]
    temp_feat = _compute_temporal_features_batch(1, round_num, date_str)[0]
    temporal_features_pos_list.append(temp_feat)
temporal_features_pos = np.array(temporal_features_pos_list)

# 음성 샘플: 히스토리에서 랜덤한 시간 정보 사용
# (시간 특징이 양성/음성 구분자가 되지 않도록)
temporal_features_neg_list = []
for _ in range(len(neg_sets)):
    # 히스토리에서 랜덤 회차 선택
    random_idx = np.random.randint(0, len(pos_meta))
    round_num, date_str = pos_meta[random_idx]
    temp_feat = _compute_temporal_features_batch(1, round_num, date_str)[0]
    temporal_features_neg_list.append(temp_feat)
temporal_features_neg = np.array(temporal_features_neg_list)

temp_time = time.time() - temp_start
print(f"        → 완료! ({temp_time:.1f}초)")

# 결합 (57개)
X_pos = np.hstack([core_features_pos, hist_features_pos, temporal_features_pos])
X_neg = np.hstack([core_features_neg, hist_features_neg, temporal_features_neg])
X = np.vstack([X_pos, X_neg])

# 레이블
y = np.array([1.0] * len(pos_sets) + [0.0] * len(neg_sets), dtype=float)

# 정규화
mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma < 1e-6] = 1.0
Xn = (X - mu) / sigma

N, D = Xn.shape
print(f"   샘플: {N}개 (양성: {len(pos_sets)}, 음성: {len(neg_sets)}), 특징: {D}개")

# K-Fold 앙상블 학습
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from joblib import parallel_backend

# 각 프로세스가 2코어씩 사용하도록 설정
n_cores = os.cpu_count() or 1
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)

print(f"   K-Fold 앙상블 학습 시작")
print(f"   [진짜 병렬 모드] joblib loky backend로 25개 프로세스 동시 실행")
print(f"   예상 시간: 40-60초")

start_time = time.time()

# 베이스 모델 정의 (최적화: alpha=0.0001로 약한 정규화)
base_model = MLPClassifier(
    hidden_layer_sizes=(100, 80, 60, 40, 20),
    activation='tanh',
    solver='adam',
    learning_rate_init=0.005,
    alpha=0.0001,  # 최적화: 0.0005 → 0.0001 (학습 속도 35% 향상)
    batch_size=200,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=0,
)

# loky backend 명시적 사용 (진짜 멀티프로세싱)
print(f"   loky backend 시작... (25개 독립 프로세스 생성)")
with parallel_backend('loky', n_jobs=25):
    cv_results = cross_validate(
        base_model, Xn, y,
        cv=skf,
        scoring='accuracy',
        return_estimator=True,
        return_train_score=True,
        verbose=2,
    )

elapsed = time.time() - start_time

# 학습된 모델과 점수 추출
ensemble_models = cv_results['estimator']
fold_scores = cv_results['test_score'].tolist()

print(f"\n   [진짜 병렬 완료] 소요 시간: {elapsed:.1f}초")
print(f"   평균 검증 정확도: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
for fold_idx, score in enumerate(fold_scores, 1):
    print(f"      Fold {fold_idx}: {score:.4f}")

# 코어 설정 원복
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)

# 앙상블 성능 평가
ensemble_probs = np.mean([m.predict_proba(Xn)[:, 1] for m in ensemble_models], axis=0)
ensemble_preds = (ensemble_probs > 0.5).astype(int)
ensemble_acc = (ensemble_preds == y).mean()

print(f"   K-Fold 앙상블 정확도: {ensemble_acc:.2%}")

# K-Fold 앙상블 저장 (임시, Stacking 학습에 필요)
ensemble_data = {
    'type': 'neural_network_ensemble',
    'models': ensemble_models,
    'mu': mu,
    'sigma': sigma,
    'n_models': len(ensemble_models),
    'ensemble_accuracy': float(ensemble_acc * 100),
    'fold_scores': fold_scores,
    'n_features': D,
}

with open('best_ml_model_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_data, f)

print(f"   [OK] K-Fold 앙상블 저장 완료")

# ===========================
# 3. Stacking 메타 모델 학습
# ===========================
print("\n[3] Stacking 메타 모델 학습")

# Out-of-fold 예측 생성
meta_predictions = np.zeros((len(X), len(ensemble_models)))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(Xn, y), 1):
    model = ensemble_models[fold_idx - 1]
    preds = model.predict_proba(Xn[val_idx])[:, 1]
    meta_predictions[val_idx, fold_idx - 1] = preds

# 메타 특징 = 25개 예측 + 57개 원본 특징 (= 82개)
X_meta = np.hstack([meta_predictions, Xn])
print(f"   메타 특징: {X_meta.shape}")

# 메타 모델 학습 (LogisticRegression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

meta_model = LogisticRegression(
    max_iter=500,
    random_state=42,
    C=0.01,  # 정규화 강화 (1.0 → 0.01)
    class_weight='balanced',
    solver='lbfgs',
)

# Cross-validation
cv_scores = cross_val_score(meta_model, X_meta, y, cv=5, scoring='accuracy')
print(f"   메타 모델 CV 점수: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 전체 데이터로 학습
meta_model.fit(X_meta, y)
y_pred = meta_model.predict(X_meta)
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y, y_pred)

# 구분력 계산
real_scores = y_pred[y == 1.0]
biased_scores = y_pred[y == 0.0]
separation = (real_scores.mean() - biased_scores.mean())

print(f"   Stacking 정확도: {train_accuracy:.2%}")
print(f"   구분력: {separation:.4f}")

# ⚡ Stacking Wrapper 생성 (배치 예측 최적화)
print("\n[4] Stacking Wrapper 생성")
wrapper = StackingModelWrapper(ensemble_models, meta_model)
print(f"   [OK] Wrapper 생성 완료 (배치 예측 최적화)")

# Stacking 모델 저장
stacking_model = {
    'type': 'stacking',
    'model_type': 'stacking',
    'model': wrapper,  # ⚡ sklearn 호환 인터페이스 (배치 예측)
    'base_models': ensemble_models,
    'meta_model': meta_model,
    'mu': mu,
    'sigma': sigma,
    'n_base_models': len(ensemble_models),
    'meta_cv_accuracy': cv_scores.mean() * 100,
    'meta_train_accuracy': train_accuracy * 100,
    'separation_power': separation,
    'n_features': D,
    'n_meta_features': X_meta.shape[1],
}

print("\n[5] 모델 저장")
with open('best_ml_model_stacking.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)

print(f"   [OK] 저장 완료: best_ml_model_stacking.pkl")
print(f"   - 특징: {stacking_model['n_features']}개")
print(f"   - Base Models: {stacking_model['n_base_models']}개")
print(f"   - Meta CV 정확도: {stacking_model['meta_cv_accuracy']:.2f}%")
print(f"   - Meta 훈련 정확도: {stacking_model['meta_train_accuracy']:.2f}%")
print(f"   - 구분력: {stacking_model['separation_power']:.4f}")

print("\n" + "=" * 80)
print("[OK] Stacking 앙상블 학습 완료!")
print("=" * 80)
