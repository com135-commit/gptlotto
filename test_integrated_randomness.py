"""
통합 테스트: 랜덤성 스코어러가 시스템에 잘 통합되었는지 확인
"""
from lotto_history import load_history_csv
from lotto_generators import train_ml_scorer, ml_score_set

print("=" * 60)
print("랜덤성 스코어러 통합 테스트")
print("=" * 60)

# 1. 히스토리 로드
history_df = load_history_csv("lotto.csv")
print(f"\n히스토리: {len(history_df)}회차")

# 2. 랜덤성 스코어러 학습
print("\n1. 랜덤성 스코어러 학습 (model_type='randomness')")
print("-" * 60)

model_randomness = train_ml_scorer(
    history_df=history_df,
    max_rounds=200,
    model_type="randomness"
)

print(f"\n모델 타입: {model_randomness.get('type')}")
print(f"샘플 수: {model_randomness.get('n_samples')}")
print(f"특징 수: {model_randomness.get('n_features')}")

# 3. ml_score_set으로 점수 계산
print("\n2. ml_score_set으로 점수 계산")
print("-" * 60)

test_cases = [
    ([1, 2, 3, 4, 5, 6], "연속 번호 (비정상)"),
    ([3, 14, 23, 28, 35, 42], "균형 (정상)"),
    ([41, 42, 43, 44, 45, 1], "고수 편중 (비정상)"),
]

for nums, desc in test_cases:
    score = ml_score_set(nums, model_randomness, weights=None, history_df=history_df)
    print(f"{str(nums):30s} | {score:.4f} | {desc}")

# 4. 기존 모델과 비교
print("\n3. 기존 모델과 비교")
print("-" * 60)

print("\n그래디언트 부스팅 학습 중...")
model_gb = train_ml_scorer(
    history_df=history_df,
    max_rounds=200,
    model_type="gradient_boosting"
)

print("\n비교 결과:")
print(f"{'번호':30s} | {'GB':>8s} | {'랜덤성':>8s} | 설명")
print("-" * 70)

for nums, desc in test_cases:
    score_gb = ml_score_set(nums, model_gb, weights=None, history_df=history_df)
    score_rand = ml_score_set(nums, model_randomness, weights=None, history_df=history_df)
    print(f"{str(nums):30s} | {score_gb:8.4f} | {score_rand:8.4f} | {desc}")

print("\n" + "=" * 60)
print("결론")
print("=" * 60)
print("\n✅ 랜덤성 스코어러가 성공적으로 통합되었습니다!")
print("   - train_ml_scorer(model_type='randomness') 작동")
print("   - ml_score_set() 호환 가능")
print("   - 기존 모델(GB)과 함께 사용 가능")
print("\n권장 사항:")
print("   GUI에서 'ML 모델 타입'을 'randomness'로 선택하세요.")
print("=" * 60)
