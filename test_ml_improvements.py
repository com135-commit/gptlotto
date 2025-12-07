#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 개선사항 테스트
- 30개 특징
- 하드 네거티브 샘플링
- 개선된 학습 파라미터
"""

import pandas as pd
import numpy as np
from lotto_generators import train_ml_scorer, ml_score_set, _set_features

print("=" * 70)
print("ML 개선사항 테스트")
print("=" * 70)

# CSV 로드
print("\n[1단계] CSV 로딩...")
try:
    df = pd.read_csv('lotto.csv')
    print(f"✓ CSV 로드 완료: {len(df)}회")
except Exception as e:
    print(f"✗ CSV 로드 실패: {e}")
    exit(1)

# 특징 개수 확인
print("\n[2단계] 특징 개수 확인...")
test_nums = [3, 12, 19, 27, 33, 41]
features = _set_features(test_nums, weights=None, history_df=df)
print(f"✓ 특징 개수: {len(features)}개")
print(f"  예상: 30개 (기존 10개 + 신규 20개)")

if len(features) == 30:
    print("  ✓ 특징 확장 성공!")
else:
    print(f"  ✗ 특징 개수 불일치 (예상: 30, 실제: {len(features)})")

# 특징 내용 출력
print("\n[3단계] 특징 상세 (샘플 번호: [3, 12, 19, 27, 33, 41])...")
feature_names = [
    # 기존 10개
    "평균", "표준편차", "짝수비율", "Low구간", "Mid구간", "High구간",
    "간격평균", "간격표준편차", "히스토리평균", "히스토리최대",
    # 통계 5개
    "최소값", "최대값", "중앙값", "범위", "IQR",
    # 패턴 8개
    "연속개수", "최대연속", "끝자리다양성", "끝자리중복",
    "3배수", "5배수", "소수개수", "대칭성",
    # 간격 4개
    "간격최소", "간격최대", "간격중앙값", "간격CV",
    # 확률 2개
    "출현빈도", "최근출현",
    # 고차원 1개
    "합끝자리",
]

for i, (name, val) in enumerate(zip(feature_names, features)):
    print(f"  {i+1:2d}. {name:12s}: {val:.4f}")

# ML 모델 학습 테스트
print("\n[4단계] ML 모델 학습 (하드 네거티브 ON)...")
try:
    model_hard = train_ml_scorer(
        df,
        weights=None,
        n_neg_per_pos=5,
        max_rounds=200,
        epochs=120,
        lr=0.05,
        use_hard_negatives=True,
    )
    print(f"✓ 학습 완료 (하드 네거티브 샘플링)")
    print(f"  - 특징 수: {model_hard.get('n_features', 'N/A')}")
    print(f"  - 정확도: {model_hard.get('accuracy', 0):.2%}")
    print(f"  - Loss: {model_hard.get('loss', 0):.4f}")
except Exception as e:
    print(f"✗ 학습 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[5단계] ML 모델 학습 (하드 네거티브 OFF, 비교용)...")
try:
    model_basic = train_ml_scorer(
        df,
        weights=None,
        n_neg_per_pos=5,
        max_rounds=200,
        epochs=120,
        lr=0.05,
        use_hard_negatives=False,
    )
    print(f"✓ 학습 완료 (기본 랜덤 샘플링)")
    print(f"  - 정확도: {model_basic.get('accuracy', 0):.2%}")
    print(f"  - Loss: {model_basic.get('loss', 0):.4f}")
except Exception as e:
    print(f"✗ 학습 실패: {e}")

# 전체 데이터 사용 테스트
print(f"\n[6단계] 전체 데이터 사용 테스트 ({len(df)}회 전체)...")
try:
    model_full = train_ml_scorer(
        df,
        weights=None,
        n_neg_per_pos=5,
        max_rounds=None,  # 전체 사용
        epochs=120,
        lr=0.05,
        use_hard_negatives=True,
    )
    print(f"✓ 학습 완료 (전체 데이터)")
    print(f"  - 정확도: {model_full.get('accuracy', 0):.2%}")
    print(f"  - Loss: {model_full.get('loss', 0):.4f}")
except Exception as e:
    print(f"✗ 학습 실패: {e}")

# 점수 매기기 테스트
print("\n[7단계] 번호 조합 점수 테스트...")
test_sets = [
    [3, 12, 19, 27, 33, 41],    # 균형잡힌 조합
    [1, 2, 3, 4, 5, 6],          # 연속 번호
    [5, 10, 15, 20, 25, 30],     # 5의 배수
    [40, 41, 42, 43, 44, 45],    # 높은 구간만
]

for nums in test_sets:
    score = ml_score_set(nums, model_hard, weights=None, history_df=df)
    print(f"  {nums} → {score:.4f}")

# 성능 비교
print("\n[8단계] 모델 성능 비교...")
print(f"{'모델':<25s} {'정확도':<10s} {'Loss':<10s}")
print("-" * 45)
print(f"{'하드 네거티브 (200회)':<25s} {model_hard.get('accuracy', 0):<10.2%} {model_hard.get('loss', 0):<10.4f}")
print(f"{'기본 랜덤 (200회)':<25s} {model_basic.get('accuracy', 0):<10.2%} {model_basic.get('loss', 0):<10.4f}")
print(f"{'하드 네거티브 (전체)':<25s} {model_full.get('accuracy', 0):<10.2%} {model_full.get('loss', 0):<10.4f}")

# 개선율 계산
acc_improvement = (model_hard.get('accuracy', 0) - model_basic.get('accuracy', 0)) * 100
print(f"\n하드 네거티브 개선율: {acc_improvement:+.1f}%p")

print("\n" + "=" * 70)
print("테스트 완료!")
print("=" * 70)
