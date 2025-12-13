"""
Neural Network K-Fold 앙상블 학습
교차 검증의 각 fold에서 학습된 25개 모델을 모두 저장하여 앙상블
"""
import numpy as np
import pandas as pd
import pickle
from lotto_history import load_history_csv
from lotto_generators import generate_random_sets, _set_features
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

print("=" * 70)
print("Neural Network K-Fold 앙상블 학습")
print("=" * 70)

# 1. 히스토리 로드
print("\n[1] 전체 히스토리 로드")
history_df = load_history_csv("lotto.csv")
print(f"   전체 데이터: {len(history_df)}회")

# 2. 학습 데이터 준비
print("\n[2] 학습 데이터 준비")

# 양성 샘플: 진짜 로또 번호
pos_sets = []
for row in history_df.itertuples(index=False):
    nums = []
    for val in row:
        try:
            v = int(val)
            if 1 <= v <= 45:
                nums.append(v)
        except (ValueError, TypeError):
            continue
    if len(nums) == 6:
        pos_sets.append(sorted(nums))

# 음성 샘플: 편향된 조합 (5배 생성)
n_neg = len(pos_sets) * 5
neg_sets = []

# 여러 편향 패턴 생성
patterns = [
    # 패턴 1: 큰 번호 편향 (31-45에 4개 이상)
    lambda: sorted(np.random.choice(range(31, 46), size=4, replace=False).tolist() +
                   np.random.choice(range(1, 31), size=2, replace=False).tolist()),
    # 패턴 2: 작은 번호 편향 (1-15에 4개 이상)
    lambda: sorted(np.random.choice(range(1, 16), size=4, replace=False).tolist() +
                   np.random.choice(range(16, 46), size=2, replace=False).tolist()),
    # 패턴 3: 연속 번호 많음 (3개 이상 연속)
    lambda: sorted([i for i in range(np.random.randint(1, 40), np.random.randint(1, 40) + 3)] +
                   np.random.choice([x for x in range(1, 46) if x not in range(np.random.randint(1, 40), np.random.randint(1, 40) + 3)], size=3, replace=False).tolist()),
    # 패턴 4: 짝수만 또는 홀수만
    lambda: sorted(np.random.choice([x for x in range(2, 46, 2)], size=6, replace=False).tolist()),
    lambda: sorted(np.random.choice([x for x in range(1, 46, 2)], size=6, replace=False).tolist()),
]

for _ in range(n_neg):
    try:
        pattern = np.random.choice(patterns)
        neg_set = pattern()
        if len(neg_set) == 6 and len(set(neg_set)) == 6 and all(1 <= n <= 45 for n in neg_set):
            neg_sets.append(neg_set)
    except Exception:
        # 패턴 생성 실패 시 완전 랜덤
        neg_sets.append(generate_random_sets(1, True, np.ones(45), None)[0])

# 특징 추출
weights = np.ones(45)
X_list = []
y_list = []

for s in pos_sets:
    X_list.append(_set_features(s, weights, history_df))
    y_list.append(1.0)

for s in neg_sets:
    X_list.append(_set_features(s, weights, history_df))
    y_list.append(0.0)

X = np.vstack(X_list)
y = np.array(y_list, dtype=float)

# 특징 정규화
mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma < 1e-6] = 1.0
Xn = (X - mu) / sigma

N, D = Xn.shape
print(f"   샘플: {N}개 (양성: {len(pos_sets)}, 음성: {len(neg_sets)}), 특징: {D}개")

# 3. K-Fold 앙상블 학습
print("\n[3] K-Fold 앙상블 학습 (25개 모델)")
print("   각 fold에서 학습된 모델을 모두 저장합니다...")

skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)

ensemble_models = []
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(Xn, y), 1):
    print(f"\n   Fold {fold_idx}/25 학습 중...")

    # 모델 생성 (최적화된 하이퍼파라미터)
    model = MLPClassifier(
        hidden_layer_sizes=(100, 80, 60, 40, 20),
        activation='tanh',
        solver='adam',
        learning_rate_init=0.005,
        alpha=0.0005,
        batch_size=200,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42 + fold_idx,  # 각 fold마다 다른 시드
    )

    # 학습
    model.fit(Xn[train_idx], y[train_idx])

    # 검증
    train_acc = model.score(Xn[train_idx], y[train_idx])
    val_acc = model.score(Xn[val_idx], y[val_idx])

    print(f"      훈련 정확도: {train_acc:.2%}")
    print(f"      검증 정확도: {val_acc:.2%}")
    print(f"      과적합 Gap: {(train_acc - val_acc):.2%}")

    ensemble_models.append(model)
    fold_scores.append(val_acc)

# 4. 앙상블 성능 평가
print("\n[4] 앙상블 성능 평가")

# 각 모델의 예측 확률 평균
ensemble_probs = np.mean([m.predict_proba(Xn)[:, 1] for m in ensemble_models], axis=0)
ensemble_preds = (ensemble_probs > 0.5).astype(int)
ensemble_acc = (ensemble_preds == y).mean()

print(f"   개별 모델 평균 검증 정확도: {np.mean(fold_scores):.2%} (+/- {np.std(fold_scores):.2%})")
print(f"   앙상블 전체 정확도: {ensemble_acc:.2%}")
print(f"   성능 향상: +{(ensemble_acc - np.mean(fold_scores)) * 100:.2f}%p")

# 5. 모델 저장
print("\n[5] 앙상블 모델 저장")

ensemble_data = {
    'type': 'neural_network_ensemble',
    'models': ensemble_models,  # 10개 모델 리스트
    'mu': mu,
    'sigma': sigma,
    'n_models': len(ensemble_models),
    'ensemble_accuracy': float(ensemble_acc),
    'fold_scores': fold_scores,
    'n_features': D,
}

with open('best_ml_model_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_data, f)

print(f"   저장 완료: best_ml_model_ensemble.pkl")
print(f"   모델 개수: {len(ensemble_models)}개")
print(f"   파일 크기: 약 {len(pickle.dumps(ensemble_data)) / 1024 / 1024:.1f} MB")

# 6. 성능 비교
print("\n[6] 기존 모델과 비교")
try:
    with open('best_ml_model.pkl', 'rb') as f:
        old_model = pickle.load(f)

    old_acc = old_model.get('cv_score', old_model.get('accuracy', 0.0))
    if isinstance(old_acc, list):
        old_acc = np.mean(old_acc)

    print(f"   기존 단일 모델 CV 점수: {old_acc:.2%}")
    print(f"   새 앙상블 모델 정확도: {ensemble_acc:.2%}")
    print(f"   성능 향상: +{(ensemble_acc - old_acc) * 100:.2f}%p")

    if ensemble_acc > old_acc:
        print(f"\n   ✅ 앙상블 모델이 더 우수합니다!")
    else:
        print(f"\n   ⚠️  기존 모델이 더 우수하거나 비슷합니다.")
except FileNotFoundError:
    print("   기존 모델 파일을 찾을 수 없습니다.")

print("\n" + "=" * 70)
print("앙상블 학습 완료!")
print("=" * 70)

print("\n[사용 방법]")
print("1. 앙상블 모델 로드:")
print("   with open('best_ml_model_ensemble.pkl', 'rb') as f:")
print("       ensemble_model = pickle.load(f)")
print("\n2. 예측:")
print("   # 25개 모델의 평균 예측")
print("   probs = [m.predict_proba(X_norm) for m in ensemble_model['models']]")
print("   ensemble_prob = np.mean(probs, axis=0)[:, 1]")
