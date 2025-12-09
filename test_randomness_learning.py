"""
ML 랜덤성 학습 테스트

기존 분류 모델들(logistic, random_forest, gradient_boosting, neural_network)을
"랜덤성 학습" 모드로 훈련시켜 효과 확인

학습 목표:
- 양성 샘플: 과거 당첨 번호 (진짜 무작위)
- 음성 샘플: 편향된 조합 (모두 짝수, 연속 번호 등)
- 결과: 높은 점수 = 더 무작위적 = 더 좋음
"""
import numpy as np
import pandas as pd
from collections import Counter

from lotto_history import load_history_csv, compute_realistic_popularity_weights
from lotto_generators import train_ml_scorer, ml_score_set, generate_biased_combinations

print("=" * 70)
print("ML 랜덤성 학습 테스트")
print("=" * 70)

# 1. 히스토리 로드
print("\n1. 히스토리 로드")
history_df = load_history_csv("lotto.csv")
history_weights = compute_realistic_popularity_weights(history_df)
print(f"   과거 당첨 번호: {len(history_df)}회")

# 2. Neural Network 랜덤성 학습 테스트
model_types = ["neural_network"]  # Neural Network만 사용

for model_type in model_types:
    print("\n" + "=" * 70)
    print(f"모델: {model_type.upper()}")
    print("=" * 70)

    # 2-1. 랜덤성 학습 모드로 훈련
    print(f"\n[1] 랜덤성 학습 모드로 훈련 중...")
    ml_model = train_ml_scorer(
        history_df=history_df,
        max_rounds=200,
        model_type=model_type,
        randomness_learning=True  # 랜덤성 학습 ON
    )

    # 2-2. 테스트 샘플 준비
    print(f"\n[2] 테스트 샘플 평가")

    # 정상 무작위 조합 (최근 10개 당첨 번호)
    normal_sets = []
    for row in history_df.tail(10).itertuples(index=False):
        nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
        if len(nums) == 6:
            normal_sets.append(nums)

    # 편향된 조합 (10개)
    biased_sets = generate_biased_combinations(10)

    # 점수 계산
    normal_scores = []
    for s in normal_sets:
        score = ml_score_set(s, ml_model, weights=history_weights, history_df=history_df)
        normal_scores.append(score)

    biased_scores = []
    for s in biased_sets:
        score = ml_score_set(s, ml_model, weights=history_weights, history_df=history_df)
        biased_scores.append(score)

    # 결과 출력
    print(f"\n   정상 무작위 조합 (과거 당첨 번호):")
    print(f"     평균 점수: {np.mean(normal_scores):.4f}")
    print(f"     범위: {np.min(normal_scores):.4f} ~ {np.max(normal_scores):.4f}")

    print(f"\n   편향된 조합:")
    print(f"     평균 점수: {np.mean(biased_scores):.4f}")
    print(f"     범위: {np.min(biased_scores):.4f} ~ {np.max(biased_scores):.4f}")

    # 분리도 측정
    gap = np.mean(normal_scores) - np.mean(biased_scores)
    print(f"\n   [결과] 점수 차이: {gap:+.4f}")

    if gap > 0.1:
        print(f"   [우수] 무작위 조합이 {gap:.4f}점 더 높음")
    elif gap > 0:
        print(f"   [보통] 무작위 조합이 약간 더 높음 ({gap:.4f})")
    else:
        print(f"   [실패] 편향 조합이 더 높음 ({gap:.4f})")

    # 샘플 예시
    print(f"\n   [샘플 예시]")
    print(f"   정상: {normal_sets[0]} → 점수 {normal_scores[0]:.4f}")
    print(f"   편향: {biased_sets[0]} → 점수 {biased_scores[0]:.4f}")

print("\n" + "=" * 70)
print("테스트 완료")
print("=" * 70)

print("\n[해석]")
print("[OK] 정상 조합 점수 > 편향 조합 점수")
print("     -> 모델이 '무작위성'을 성공적으로 학습함")
print("\n[NG] 정상 조합 점수 < 편향 조합 점수")
print("     -> 모델이 무작위성을 제대로 학습하지 못함")
