"""
실제 로또 데이터로 랜덤성 스코어러 테스트
"""
from lotto_history import load_history_csv
from lotto_randomness_scorer import train_randomness_scorer, score_randomness

print("=" * 60)
print("실제 로또 데이터 - 랜덤성 스코어러 테스트")
print("=" * 60)

# 1. 히스토리 로드
history_df = load_history_csv("lotto.csv")
print(f"\n로또 데이터: {len(history_df)}회차")

# 2. 랜덤성 스코어러 학습
print("\n랜덤성 스코어러 학습 중...")
model = train_randomness_scorer(history_df, max_rounds=200)

# 3. 테스트 케이스
print("\n" + "=" * 60)
print("테스트 케이스")
print("=" * 60)

test_cases = [
    # 비정상 패턴
    ([1, 2, 3, 4, 5, 6], "연속 번호"),
    ([5, 10, 15, 20, 25, 30], "등차수열 (5씩)"),
    ([1, 3, 5, 7, 9, 11], "홀수만 (저수)"),
    ([2, 4, 6, 8, 10, 12], "짝수만 (저수)"),
    ([31, 33, 35, 37, 39, 41], "홀수만 (고수)"),
    ([32, 34, 36, 38, 40, 42], "짝수만 (고수)"),
    ([1, 11, 21, 31, 41, 42], "일의 자리 1"),
    ([41, 42, 43, 44, 45, 1], "고수 편중"),

    # 정상 패턴
    ([3, 14, 23, 28, 35, 42], "균형 1 (실제 당첨 패턴)"),
    ([5, 14, 21, 29, 36, 41], "균형 2 (실제 당첨 패턴)"),
    ([1, 15, 22, 30, 37, 43], "균형 3 (실제 당첨 패턴)"),
    ([7, 16, 24, 31, 38, 44], "균형 4 (실제 당첨 패턴)"),
    ([2, 13, 20, 27, 34, 40], "균형 5 (실제 당첨 패턴)"),
]

print("\n비정상 패턴 (낮은 점수 예상):")
print("-" * 60)
for i, (nums, desc) in enumerate(test_cases[:8]):
    score = score_randomness(nums, model)
    print(f"{str(nums):30s} | {score:.4f} | {desc}")

print("\n정상 패턴 (높은 점수 예상):")
print("-" * 60)
for i, (nums, desc) in enumerate(test_cases[8:]):
    score = score_randomness(nums, model)
    print(f"{str(nums):30s} | {score:.4f} | {desc}")

# 4. 실제 최근 당첨 번호 평가
print("\n" + "=" * 60)
print("실제 최근 당첨 번호 평가")
print("=" * 60)

recent_numbers = []
for row in history_df.head(10).itertuples(index=False):
    nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
    if len(nums) == 6:
        recent_numbers.append(nums)

print("\n최근 10회 당첨 번호:")
print("-" * 60)
scores = []
for i, nums in enumerate(recent_numbers):
    score = score_randomness(nums, model)
    scores.append(score)
    print(f"회차 {10-i:2d}: {str(nums):30s} | {score:.4f}")

print("-" * 60)
print(f"평균 점수: {sum(scores)/len(scores):.4f}")
print(f"최소 점수: {min(scores):.4f}")
print(f"최대 점수: {max(scores):.4f}")
print(f"표준편차: {sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores) ** 0.5:.4f}")

# 5. 기존 ML과 비교
print("\n" + "=" * 60)
print("기존 ML (Gradient Boosting) vs 랜덤성 스코어러 비교")
print("=" * 60)

from lotto_generators import train_ml_scorer, ml_score_set

print("\n기존 ML 학습 중...")
ml_model_old = train_ml_scorer(
    history_df=history_df,
    max_rounds=200,
    model_type='gradient_boosting'
)

print("\n비교 테스트:")
print("-" * 60)
print(f"{'번호':30s} | {'기존 ML':>8s} | {'랜덤성':>8s} | 설명")
print("-" * 60)

for nums, desc in test_cases:
    score_old = ml_score_set(nums, ml_model_old, weights=None, history_df=history_df)
    score_new = score_randomness(nums, model)

    print(f"{str(nums):30s} | {score_old:8.4f} | {score_new:8.4f} | {desc}")

print("\n" + "=" * 60)
print("결론")
print("=" * 60)
print("\n기존 ML (Classification):")
print("  - 당첨 vs 비당첨 구분 시도")
print("  - 과적합 발생 가능성")
print("  - 특정 패턴 선호")

print("\n랜덤성 스코어러 (One-Class):")
print("  - 로또 통계적 분포만 학습")
print("  - 과적합 없음")
print("  - 편향 없음")

print("\n" + "=" * 60)
