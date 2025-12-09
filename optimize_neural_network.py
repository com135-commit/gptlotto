"""
Neural Network 하이퍼파라미터 최적화
모든 CSV 데이터를 사용하여 최고 성능의 모델 생성

최적화 대상:
- 레이어 구조 (hidden_layer_sizes)
- 학습률 (learning_rate_init)
- 활성화 함수 (activation)
- 최적화 알고리즘 (solver)
- 배치 크기 (batch_size)
- 정규화 (alpha)
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import time
import pickle

from lotto_history import load_history_csv, compute_realistic_popularity_weights
from lotto_generators import generate_biased_combinations, _set_features

print("=" * 80)
print("Neural Network 하이퍼파라미터 최적화")
print("=" * 80)

# 1. 전체 데이터 로드
print("\n[1] 전체 히스토리 로드")
history_df = load_history_csv("lotto.csv")
history_weights = compute_realistic_popularity_weights(history_df)
print(f"   전체 데이터: {len(history_df)}회")

# 2. 양성 샘플 (과거 당첨 번호 전체)
print("\n[2] 학습 데이터 준비")
pos_sets = []
for row in history_df.itertuples(index=False):
    nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
    if len(nums) == 6:
        pos_sets.append(nums)

print(f"   양성 샘플: {len(pos_sets)}개 (전체 당첨 번호)")

# 3. 음성 샘플 (편향된 조합)
n_neg = len(pos_sets) * 5
neg_sets = generate_biased_combinations(n_neg)
print(f"   음성 샘플: {len(neg_sets)}개 (편향 조합)")

# 4. 특징 추출
X_list = []
y_list = []

print("\n[3] 특징 추출 중...")
for s in pos_sets:
    X_list.append(_set_features(s, history_weights, history_df))
    y_list.append(1.0)

for s in neg_sets:
    X_list.append(_set_features(s, history_weights, history_df))
    y_list.append(0.0)

X = np.vstack(X_list)
y = np.array(y_list, dtype=float)

# 특징 정규화
mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma < 1e-6] = 1.0
Xn = (X - mu) / sigma

print(f"   샘플: {len(y):,}개 (양성 {int(y.sum())}, 음성 {int(len(y)-y.sum())})")
print(f"   특징: {Xn.shape[1]}개")

# 5. 하이퍼파라미터 그리드 정의
print("\n[4] 하이퍼파라미터 그리드 정의")

# 방법 1: Grid Search (모든 조합 시도 - 시간 오래 걸림)
# 방법 2: Randomized Search (랜덤 샘플링 - 빠름)

# 우선 Randomized Search로 넓게 탐색
param_distributions = {
    # 레이어 구조 (10가지)
    'hidden_layer_sizes': [
        (50, 30, 10),      # 현재 (기본)
        (100, 50, 25),     # 더 크게
        (100, 50, 25, 10), # 더 깊게
        (200, 100, 50),    # 매우 크게
        (30, 20, 10),      # 작게
        (64, 32, 16),      # 2의 거듭제곱
        (128, 64, 32),     # 더 큰 2의 거듭제곱
        (80, 40, 20, 10),  # 4층
        (100, 80, 60, 40, 20), # 5층 (매우 깊음)
        (150, 100, 50, 25), # 큰 4층
    ],

    # 활성화 함수 (3가지)
    'activation': [
        'relu',      # 현재 (기본)
        'tanh',      # 대안 1
        'logistic',  # 대안 2 (sigmoid)
    ],

    # 최적화 알고리즘 (2가지)
    'solver': [
        'adam',  # 현재 (기본, 빠름)
        'lbfgs', # 대안 (작은 데이터셋에 좋음)
    ],

    # 학습률 (8가지)
    'learning_rate_init': [
        0.001,  # 매우 작게
        0.003,
        0.005,
        0.01,   # 현재 (기본)
        0.02,
        0.03,
        0.05,
        0.1,    # 크게
    ],

    # 정규화 (L2 penalty) (6가지)
    'alpha': [
        0.00001,  # 매우 약함
        0.0001,   # 현재 (기본)
        0.0005,
        0.001,
        0.005,
        0.01,     # 강함
    ],

    # 배치 크기 (3가지)
    'batch_size': [
        'auto',  # 현재 (기본, min(200, n_samples))
        100,
        200,
    ],

    # 기타 고정 파라미터
    'max_iter': [300],  # 충분히 큼
    'early_stopping': [True],
    'validation_fraction': [0.1],
    'random_state': [42],
}

print(f"   레이어 구조: {len(param_distributions['hidden_layer_sizes'])}가지")
print(f"   활성화 함수: {len(param_distributions['activation'])}가지")
print(f"   최적화 알고리즘: {len(param_distributions['solver'])}가지")
print(f"   학습률: {len(param_distributions['learning_rate_init'])}가지")
print(f"   정규화: {len(param_distributions['alpha'])}가지")
print(f"   배치 크기: {len(param_distributions['batch_size'])}가지")

# 전체 조합 수
total_combinations = (
    len(param_distributions['hidden_layer_sizes']) *
    len(param_distributions['activation']) *
    len(param_distributions['solver']) *
    len(param_distributions['learning_rate_init']) *
    len(param_distributions['alpha']) *
    len(param_distributions['batch_size'])
)
print(f"\n   전체 조합 수: {total_combinations:,}개")

# 샘플링 개수 (전체의 10% 정도)
n_iter = min(100, total_combinations // 10)
print(f"   시도할 조합: {n_iter}개 (랜덤 샘플링)")

# 6. Randomized Search 실행
print("\n[5] 하이퍼파라미터 최적화 시작")
print("   (시간이 오래 걸릴 수 있습니다...)")

start_time = time.time()

# Stratified K-Fold (클래스 비율 유지)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=MLPClassifier(),
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,  # 모든 CPU 사용
    verbose=2,  # 진행 상황 표시
    random_state=42,
    return_train_score=True,
)

random_search.fit(Xn, y)

elapsed_time = time.time() - start_time
print(f"\n   최적화 완료! (소요 시간: {elapsed_time/60:.1f}분)")

# 7. 결과 분석
print("\n" + "=" * 80)
print("최적화 결과")
print("=" * 80)

print(f"\n[최고 성능 모델]")
print(f"   교차 검증 점수: {random_search.best_score_:.4f}")
print(f"   최적 파라미터:")
for param, value in random_search.best_params_.items():
    print(f"     {param}: {value}")

# 훈련 점수와 검증 점수 비교
best_idx = random_search.best_index_
train_score = random_search.cv_results_['mean_train_score'][best_idx]
test_score = random_search.cv_results_['mean_test_score'][best_idx]
print(f"\n   훈련 점수: {train_score:.4f}")
print(f"   검증 점수: {test_score:.4f}")
print(f"   과적합 Gap: {(train_score - test_score):.4f}")

# Top 10 모델
print(f"\n[Top 10 모델]")
results = pd.DataFrame(random_search.cv_results_)
top10 = results.nlargest(10, 'mean_test_score')[
    ['mean_test_score', 'std_test_score', 'mean_train_score', 'params']
]
for i, row in top10.iterrows():
    print(f"{i+1:2d}. CV점수: {row['mean_test_score']:.4f} "
          f"(±{row['std_test_score']:.4f}), "
          f"훈련: {row['mean_train_score']:.4f}")

# 8. 최적 모델 저장
print("\n[6] 최적 모델 저장")
best_model = random_search.best_estimator_

# 전체 데이터로 재학습
print("   전체 데이터로 최종 학습 중...")
best_model.fit(Xn, y)
final_score = best_model.score(Xn, y)
print(f"   최종 훈련 점수: {final_score:.4f}")

# 모델 저장
model_data = {
    'type': 'neural_network',
    'sklearn_model': best_model,
    'mu': mu,
    'sigma': sigma,
    'accuracy': float(final_score),
    'cv_score': float(random_search.best_score_),
    'best_params': random_search.best_params_,
    'n_features': Xn.shape[1],
    'optimization_time': elapsed_time,
}

with open('best_ml_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   모델 저장: best_ml_model.pkl")

# 9. 랜덤성 테스트
print("\n[7] 최적 모델 랜덤성 테스트")

# 정상 무작위 조합 (최근 20개 당첨 번호)
normal_sets = []
for row in history_df.tail(20).itertuples(index=False):
    nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
    if len(nums) == 6:
        normal_sets.append(nums)

# 편향된 조합 (20개)
biased_test = generate_biased_combinations(20)

# 점수 계산
normal_features = [_set_features(s, history_weights, history_df) for s in normal_sets]
biased_features = [_set_features(s, history_weights, history_df) for s in biased_test]

normal_features_n = (np.array(normal_features) - mu) / sigma
biased_features_n = (np.array(biased_features) - mu) / sigma

normal_scores = best_model.predict_proba(normal_features_n)[:, 1]
biased_scores = best_model.predict_proba(biased_features_n)[:, 1]

print(f"\n   정상 무작위 조합: {np.mean(normal_scores):.4f} (범위: {np.min(normal_scores):.4f}~{np.max(normal_scores):.4f})")
print(f"   편향된 조합: {np.mean(biased_scores):.4f} (범위: {np.min(biased_scores):.4f}~{np.max(biased_scores):.4f})")
print(f"   차이: {np.mean(normal_scores) - np.mean(biased_scores):+.4f}")

if np.mean(normal_scores) - np.mean(biased_scores) > 0.5:
    print("\n   평가: 우수! 무작위성을 매우 잘 학습했습니다.")
elif np.mean(normal_scores) - np.mean(biased_scores) > 0.3:
    print("\n   평가: 좋음! 무작위성을 잘 학습했습니다.")
elif np.mean(normal_scores) - np.mean(biased_scores) > 0.1:
    print("\n   평가: 보통. 무작위성을 학습했습니다.")
else:
    print("\n   평가: 개선 필요. 무작위성 학습이 부족합니다.")

print("\n" + "=" * 80)
print("최적화 완료!")
print("=" * 80)

print("\n사용 방법:")
print("1. best_ml_model.pkl 파일이 생성되었습니다.")
print("2. lotto_generators.py의 train_ml_scorer()에서 이 모델을 로드하여 사용하세요.")
print(f"\n최적 파라미터를 lotto_generators.py에 직접 적용하려면:")
print("MLPClassifier(")
for param, value in random_search.best_params_.items():
    if param != 'random_state':
        print(f"    {param}={repr(value)},")
print("    random_state=42,")
print(")")
