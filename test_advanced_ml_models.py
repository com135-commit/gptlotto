#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고급 ML 모델 테스트
- 로지스틱 회귀 (기본)
- 랜덤 포레스트
- 그래디언트 부스팅
- 신경망
"""

import pandas as pd
import numpy as np
from lotto_generators import train_ml_scorer, ml_score_set

print("=" * 70)
print("고급 ML 모델 테스트")
print("=" * 70)

# CSV 로드
print("\n[1단계] CSV 로딩...")
try:
    df = pd.read_csv('lotto.csv')
    print(f"✓ CSV 로드 완료: {len(df)}회")
except Exception as e:
    print(f"✗ CSV 로드 실패: {e}")
    exit(1)

# 모델 타입 리스트
model_types = [
    ("logistic", "로지스틱 회귀"),
    ("random_forest", "랜덤 포레스트"),
    ("gradient_boosting", "그래디언트 부스팅"),
    ("neural_network", "신경망"),
]

# 각 모델 학습 및 비교
models = {}
results = []

for model_type, model_name in model_types:
    print(f"\n{'='*70}")
    print(f"[{model_name}] 학습 시작")
    print(f"{'='*70}")

    try:
        model = train_ml_scorer(
            df,
            weights=None,
            n_neg_per_pos=5,
            max_rounds=200,
            epochs=120,
            lr=0.05,
            use_hard_negatives=True,
            model_type=model_type,
        )

        models[model_type] = model

        # 결과 저장
        accuracy = model.get('accuracy', 0)
        loss = model.get('loss', 0)
        cv_scores = model.get('cv_scores', [])

        result = {
            'type': model_type,
            'name': model_name,
            'accuracy': accuracy,
            'loss': loss,
            'cv_mean': np.mean(cv_scores) if cv_scores else 0,
            'cv_std': np.std(cv_scores) if cv_scores else 0,
        }
        results.append(result)

        print(f"✓ {model_name} 학습 완료!")
        print(f"  - 훈련 정확도: {accuracy:.2%}")
        if cv_scores:
            print(f"  - 교차 검증: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
        if loss:
            print(f"  - Loss: {loss:.4f}")

    except Exception as e:
        print(f"✗ {model_name} 학습 실패: {e}")
        import traceback
        traceback.print_exc()

# 성능 비교 테이블
print("\n" + "=" * 70)
print("모델 성능 비교")
print("=" * 70)

print(f"{'모델':<20s} {'훈련 정확도':<15s} {'교차 검증':<20s} {'Loss':<10s}")
print("-" * 70)

for r in results:
    name = r['name']
    acc = r['accuracy']
    cv_mean = r['cv_mean']
    cv_std = r['cv_std']
    loss = r['loss']

    if cv_mean > 0:
        cv_str = f"{cv_mean:.2%} (+/- {cv_std:.2%})"
    else:
        cv_str = "N/A"

    loss_str = f"{loss:.4f}" if loss > 0 else "N/A"

    print(f"{name:<20s} {acc:<15.2%} {cv_str:<20s} {loss_str:<10s}")

# 점수 비교 (동일한 번호 조합에 대해)
print("\n" + "=" * 70)
print("점수 비교 (동일 번호 조합)")
print("=" * 70)

test_sets = [
    ([3, 12, 19, 27, 33, 41], "균형잡힌 조합"),
    ([1, 2, 3, 4, 5, 6], "연속 번호"),
    ([5, 10, 15, 20, 25, 30], "5의 배수"),
    ([40, 41, 42, 43, 44, 45], "높은 구간만"),
]

print(f"\n{'번호 조합':<30s} ", end="")
for model_type, model_name in model_types:
    if model_type in models:
        print(f"{model_name[:8]:<12s} ", end="")
print()
print("-" * 90)

for nums, desc in test_sets:
    print(f"{str(nums):<30s} ", end="")

    for model_type, _ in model_types:
        if model_type in models:
            try:
                score = ml_score_set(nums, models[model_type], weights=None, history_df=df)
                print(f"{score:<12.4f} ", end="")
            except Exception as e:
                print(f"{'ERROR':<12s} ", end="")
    print(f"  ({desc})")

# 최고 성능 모델 선택
print("\n" + "=" * 70)
print("권장 모델")
print("=" * 70)

if results:
    # 교차 검증 평균으로 정렬 (없으면 훈련 정확도)
    best_model = max(results, key=lambda x: x['cv_mean'] if x['cv_mean'] > 0 else x['accuracy'])

    print(f"\n🏆 최고 성능: {best_model['name']}")
    print(f"   - 훈련 정확도: {best_model['accuracy']:.2%}")
    if best_model['cv_mean'] > 0:
        print(f"   - 교차 검증: {best_model['cv_mean']:.2%} (+/- {best_model['cv_std']:.2%})")

    print("\n💡 권장 사항:")
    if best_model['type'] == 'logistic':
        print("   ✓ 로지스틱 회귀: 빠르고 안정적, 일반 사용에 적합")
    elif best_model['type'] == 'random_forest':
        print("   ✓ 랜덤 포레스트: 높은 정확도, 과적합 위험 낮음")
    elif best_model['type'] == 'gradient_boosting':
        print("   ✓ 그래디언트 부스팅: 최고 정확도, 느림")
    elif best_model['type'] == 'neural_network':
        print("   ✓ 신경망: 복잡한 패턴 학습, 과적합 주의")

print("\n⚠️  주의사항:")
print("   - 교차 검증 점수가 훈련 정확도보다 낮으면 과적합 가능성")
print("   - 로또는 독립 시행이므로 높은 정확도 ≠ 당첨 보장")
print("   - sklearn 모델은 로지스틱보다 느림 (특히 신경망)")

print("\n" + "=" * 70)
print("테스트 완료!")
print("=" * 70)
