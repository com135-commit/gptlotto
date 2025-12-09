#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQLE 고도화: 추가 고전 알고리즘 10개
통계적/조합론적/히스토리 분석 기반 생성기
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter
from lotto_utils import get_rng

_rng = get_rng()


def _normalize_weights(weights):
    """가중치 정규화"""
    w = np.array(weights, dtype=float)
    w = np.maximum(w, 1e-9)
    return w / w.sum()


def _generate_weighted_sample(weights, exclude_set: set[int] | None = None) -> list[int]:
    """가중치 기반 6개 번호 샘플링"""
    xs = np.arange(1, 46)
    p = _normalize_weights(weights)

    if exclude_set:
        for v in exclude_set:
            if 1 <= v <= 45:
                p[v - 1] = 0.0
        p = _normalize_weights(p)

    chosen = []
    for _ in range(6):
        if p.sum() <= 0:
            break
        v = int(_rng.choice(xs, p=p))
        chosen.append(v)
        p[v - 1] = 0.0
        if p.sum() > 0:
            p = _normalize_weights(p)

    return sorted(chosen) if len(chosen) == 6 else sorted(chosen + [1] * (6 - len(chosen)))


# ==================== 통계적 고전 알고리즘 (3개) ====================

def gen_hot_cold_balanced(
    history_df: pd.DataFrame,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    핫/콜드 번호 균형 전략
    최근 N회 자주 나온 번호(핫) 3개 + 안 나온 번호(콜드) 3개
    """
    if history_df is None or history_df.empty:
        # 히스토리 없으면 일반 랜덤
        return _generate_weighted_sample(weights or np.ones(45), exclude_set)

    recent_N = 20
    recent = history_df.tail(recent_N).values.flatten()
    counts = Counter(int(x) for x in recent if 1 <= int(x) <= 45)

    # 핫 번호 (자주 나온 번호 top 15)
    hot_candidates = [n for n, c in counts.most_common(15) if n not in (exclude_set or set())]

    # 콜드 번호 (안 나온 번호)
    all_nums = set(range(1, 46)) - (exclude_set or set())
    appeared = set(counts.keys())
    cold_candidates = sorted(all_nums - appeared)

    # 핫 3개, 콜드 3개 선택
    try:
        hot = _rng.choice(hot_candidates, size=min(3, len(hot_candidates)), replace=False).tolist()
    except:
        hot = []

    try:
        cold = _rng.choice(cold_candidates, size=min(3, len(cold_candidates)), replace=False).tolist()
    except:
        cold = []

    # 부족하면 중간 번호로 채움
    combined = hot + cold
    if len(combined) < 6:
        mid_candidates = [n for n in range(1, 46) if n not in combined and n not in (exclude_set or set())]
        needed = 6 - len(combined)
        if mid_candidates:
            combined.extend(_rng.choice(mid_candidates, size=min(needed, len(mid_candidates)), replace=False).tolist())

    return sorted(combined[:6]) if combined else sorted(list(range(1, 7)))


def gen_consecutive_controlled(
    weights=None,
    exclude_set: set[int] | None = None,
    max_consecutive: int = 2,
) -> list[int]:
    """
    연속성 제어 전략
    연속 번호가 최대 N개까지만 허용 (기본 2개)
    """
    if weights is None:
        weights = np.ones(45)

    for attempt in range(100):
        nums = _generate_weighted_sample(weights, exclude_set)
        sorted_nums = sorted(nums)

        # 연속 번호 쌍 개수 확인
        consecutive_count = sum(
            1 for i in range(len(sorted_nums) - 1)
            if sorted_nums[i+1] - sorted_nums[i] == 1
        )

        if consecutive_count <= max_consecutive:
            return sorted_nums

    # 100번 시도 후에도 안 되면 그냥 반환
    return _generate_weighted_sample(weights, exclude_set)


def gen_sum_range_controlled(
    weights=None,
    exclude_set: set[int] | None = None,
    min_sum: int = 115,
    max_sum: int = 175,
) -> list[int]:
    """
    합계 범위 제어 전략
    6개 번호 합계를 통계적 정상 범위(115~175) 내로 제한
    """
    if weights is None:
        weights = np.ones(45)

    for attempt in range(100):
        nums = _generate_weighted_sample(weights, exclude_set)
        total = sum(nums)

        if min_sum <= total <= max_sum:
            return nums

    # 100번 시도 후에도 안 되면 그냥 반환
    return _generate_weighted_sample(weights, exclude_set)


# ==================== 조합론적 고전 알고리즘 (3개) ====================

def gen_zone_balanced(
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    구간 균형 전략
    1-15(저수), 16-30(중수), 31-45(고수) 각 구간에서 2개씩
    """
    if weights is None:
        weights = np.ones(45)

    zones = [
        list(range(1, 16)),   # 저수
        list(range(16, 31)),  # 중수
        list(range(31, 46)),  # 고수
    ]

    result = []
    for zone in zones:
        available = [n for n in zone if n not in (exclude_set or set())]
        if len(available) >= 2:
            zone_weights = _normalize_weights([weights[n-1] for n in available])
            selected = _rng.choice(available, size=2, replace=False, p=zone_weights).tolist()
            result.extend(selected)
        elif len(available) == 1:
            result.append(available[0])

    # 부족하면 랜덤으로 채움
    if len(result) < 6:
        remaining = [n for n in range(1, 46) if n not in result and n not in (exclude_set or set())]
        needed = 6 - len(result)
        if remaining:
            result.extend(_rng.choice(remaining, size=min(needed, len(remaining)), replace=False).tolist())

    return sorted(result[:6]) if result else sorted(list(range(1, 7)))


def gen_last_digit_diverse(
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    끝자리 다양성 전략
    끝자리 0~9를 최대한 다양하게 (중복 최소화)
    """
    if weights is None:
        weights = np.ones(45)

    used_last_digits = set()
    result = []
    available = [n for n in range(1, 46) if n not in (exclude_set or set())]

    # 1차: 끝자리가 겹치지 않는 번호 선택
    for num in available:
        if len(result) >= 6:
            break
        last_digit = num % 10
        if last_digit not in used_last_digits:
            # 가중치 기반 확률적 선택
            if _rng.random() < weights[num-1] / max(weights):
                result.append(num)
                used_last_digits.add(last_digit)

    # 2차: 부족하면 가중치 기반으로 채움
    if len(result) < 6:
        remaining = [n for n in available if n not in result]
        if remaining:
            needed = 6 - len(result)
            rem_weights = _normalize_weights([weights[n-1] for n in remaining])
            selected = _rng.choice(remaining, size=min(needed, len(remaining)), replace=False, p=rem_weights).tolist()
            result.extend(selected)

    return sorted(result[:6]) if result else sorted(list(range(1, 7)))


def gen_prime_mixed(
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    소수 혼합 전략
    소수 3개 + 합성수 3개 조합
    """
    if weights is None:
        weights = np.ones(45)

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    composites = [n for n in range(1, 46) if n not in primes]

    prime_avail = [p for p in primes if p not in (exclude_set or set())]
    comp_avail = [c for c in composites if c not in (exclude_set or set())]

    result = []

    # 소수 3개
    if len(prime_avail) >= 3:
        prime_weights = _normalize_weights([weights[p-1] for p in prime_avail])
        selected_primes = _rng.choice(prime_avail, size=3, replace=False, p=prime_weights).tolist()
        result.extend(selected_primes)
    elif prime_avail:
        result.extend(prime_avail[:3])

    # 합성수 3개
    if len(comp_avail) >= 3:
        comp_weights = _normalize_weights([weights[c-1] for c in comp_avail])
        selected_comps = _rng.choice(comp_avail, size=3, replace=False, p=comp_weights).tolist()
        result.extend(selected_comps)
    elif comp_avail:
        result.extend(comp_avail[:3])

    # 부족하면 채움
    if len(result) < 6:
        all_avail = [n for n in range(1, 46) if n not in result and n not in (exclude_set or set())]
        needed = 6 - len(result)
        if all_avail:
            result.extend(_rng.choice(all_avail, size=min(needed, len(all_avail)), replace=False).tolist())

    return sorted(result[:6]) if result else sorted(list(range(1, 7)))


# ==================== 히스토리 분석 기반 (4개) ====================

def gen_temporal_strategy(
    history_df: pd.DataFrame,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    시간 전략
    최근 5회 회피 + 10~30회 전 번호 선호
    """
    if history_df is None or history_df.empty:
        return _generate_weighted_sample(weights or np.ones(45), exclude_set)

    if weights is None:
        weights = np.ones(45)

    # 최근 5회 번호
    recent_5 = set()
    if len(history_df) >= 5:
        recent_5 = set(int(x) for x in history_df.tail(5).values.flatten() if 1 <= int(x) <= 45)

    # 10~30회 전 번호 (중기)
    mid_term = set()
    if len(history_df) >= 30:
        mid_term = set(int(x) for x in history_df.tail(30).head(20).values.flatten() if 1 <= int(x) <= 45)

    # 최근 5회 제외
    avoid = recent_5 | (exclude_set or set())

    # 중기 번호에 가중치 2배
    adjusted_weights = weights.copy()
    for num in mid_term:
        if 1 <= num <= 45:
            adjusted_weights[num-1] *= 2.0

    return _generate_weighted_sample(adjusted_weights, avoid)


def gen_frequency_inverted(
    history_df: pd.DataFrame,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    빈도 역전 전략 (평균 회귀)
    과거 덜 나온 번호에 높은 가중치
    """
    if history_df is None or history_df.empty:
        return _generate_weighted_sample(weights or np.ones(45), exclude_set)

    if weights is None:
        weights = np.ones(45)

    # 전체 히스토리에서 각 번호 출현 횟수
    all_nums = history_df.values.flatten()
    counts = Counter(int(x) for x in all_nums if 1 <= int(x) <= 45)

    # 역빈도 가중치 (덜 나온 번호가 높음)
    max_count = max(counts.values()) if counts else 1
    inverted_weights = np.array([
        max_count - counts.get(i+1, 0) + 1
        for i in range(45)
    ], dtype=float)

    # 기존 가중치와 혼합 (50:50)
    combined_weights = 0.5 * weights + 0.5 * inverted_weights

    return _generate_weighted_sample(combined_weights, exclude_set)


def gen_cycle_pattern(
    history_df: pd.DataFrame,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[int]:
    """
    주기 패턴 전략
    각 번호의 출현 주기를 분석하여 '나올 시기' 번호 선택
    """
    if history_df is None or history_df.empty or len(history_df) < 10:
        return _generate_weighted_sample(weights or np.ones(45), exclude_set)

    candidates = []

    for num in range(1, 46):
        if num in (exclude_set or set()):
            continue

        # 해당 번호가 나온 회차 인덱스들
        appearances = []
        for idx in range(len(history_df)):
            row = history_df.iloc[idx]
            if num in row.values:
                appearances.append(idx)

        if len(appearances) >= 2:
            # 평균 주기 계산
            cycles = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            avg_cycle = np.mean(cycles)

            # 마지막 출현 후 경과 회차
            last_appearance = appearances[-1]
            elapsed = len(history_df) - 1 - last_appearance

            # 주기에 가까우면 선택 확률 높임 (80% 이상 경과)
            if elapsed >= avg_cycle * 0.8:
                ratio = elapsed / avg_cycle if avg_cycle > 0 else 0
                candidates.append((num, ratio))

    if len(candidates) >= 6:
        # 주기 비율 높은 순으로 6개 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        return sorted([num for num, _ in candidates[:6]])
    elif candidates:
        # 부족하면 가중치로 채움
        selected = [num for num, _ in candidates]
        remaining = [n for n in range(1, 46) if n not in selected and n not in (exclude_set or set())]
        needed = 6 - len(selected)
        if remaining and weights is not None:
            rem_weights = _normalize_weights([weights[n-1] for n in remaining])
            selected.extend(_rng.choice(remaining, size=min(needed, len(remaining)), replace=False, p=rem_weights).tolist())
        return sorted(selected[:6])
    else:
        # 주기 분석 실패 시 일반 랜덤
        return _generate_weighted_sample(weights or np.ones(45), exclude_set)


def gen_ml_guided(
    ml_model,
    history_df: pd.DataFrame,
    weights=None,
    exclude_set: set[int] | None = None,
    n_candidates: int = 50,
) -> list[int]:
    """
    ML 가이드 전략
    ML 점수가 높은 조합을 몬테카를로 방식으로 탐색
    """
    if ml_model is None:
        return _generate_weighted_sample(weights or np.ones(45), exclude_set)

    from lotto_generators import ml_score_set

    if weights is None:
        weights = np.ones(45)

    best_set = None
    best_score = -1

    # N개 후보 생성하여 최고 ML 점수 선택
    for _ in range(n_candidates):
        candidate = _generate_weighted_sample(weights, exclude_set)
        try:
            score = ml_score_set(candidate, ml_model, weights=weights, history_df=history_df)
            if score > best_score:
                best_score = score
                best_set = candidate
        except:
            continue

    return best_set if best_set else _generate_weighted_sample(weights, exclude_set)
