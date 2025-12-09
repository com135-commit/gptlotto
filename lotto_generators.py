#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로또 번호 생성 알고리즘 모듈
- 기본/패턴 생성기
- 천재(GI), MDA, CC, PR, IS 알고리즘
- GAP-R, QH, HD, QP 계열, MQLE 등
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lotto_utils import get_rng

# 전역 랜덤 생성기 사용
_rng = get_rng()


# ------------------ 편향된 조합 생성 (랜덤성 학습용) ------------------
def generate_biased_combinations(n_sets: int) -> list[list[int]]:
    """
    명백히 편향된(비무작위적) 번호 조합 생성
    ML 모델이 "무작위성"을 학습하기 위한 음성 샘플로 사용

    편향 유형:
    1. 모두 짝수/홀수
    2. 연속 번호
    3. 한쪽 범위에 몰림 (저수/중수/고수만)
    4. 같은 끝자리
    5. 등차수열
    6. 배수만 (3의 배수, 5의 배수)
    7. 극단적 범위 조합
    """
    biased_sets = []

    # 각 편향 유형별로 생성
    types_per_category = max(1, n_sets // 10)

    # 1. 모두 짝수
    evens = [n for n in range(2, 46, 2)]  # 2, 4, 6, ..., 44
    for _ in range(types_per_category):
        biased_sets.append(sorted(_rng.choice(evens, size=6, replace=False).tolist()))

    # 2. 모두 홀수
    odds = [n for n in range(1, 46, 2)]  # 1, 3, 5, ..., 45
    for _ in range(types_per_category):
        biased_sets.append(sorted(_rng.choice(odds, size=6, replace=False).tolist()))

    # 3. 연속 번호 (6개 연속)
    for _ in range(types_per_category):
        start = _rng.integers(1, 40)  # 1-39
        biased_sets.append(list(range(start, start + 6)))

    # 4. 모두 저수 (1-15)
    low_nums = list(range(1, 16))
    for _ in range(types_per_category):
        biased_sets.append(sorted(_rng.choice(low_nums, size=6, replace=False).tolist()))

    # 5. 모두 고수 (31-45)
    high_nums = list(range(31, 46))
    for _ in range(types_per_category):
        biased_sets.append(sorted(_rng.choice(high_nums, size=6, replace=False).tolist()))

    # 6. 같은 끝자리 (예: 1, 11, 21, 31, 41, ...)
    for _ in range(types_per_category):
        last_digit = _rng.integers(0, 10)  # 0-9
        candidates = [n for n in range(1, 46) if n % 10 == last_digit]
        if len(candidates) >= 6:
            biased_sets.append(sorted(_rng.choice(candidates, size=6, replace=False).tolist()))

    # 7. 등차수열 (공차 5)
    for _ in range(types_per_category):
        start = _rng.integers(1, 15)
        diff = _rng.integers(3, 8)  # 공차 3-7
        seq = [start + i * diff for i in range(6)]
        if all(1 <= n <= 45 for n in seq):
            biased_sets.append(sorted(seq))

    # 8. 3의 배수만
    mult3 = [n for n in range(3, 46, 3)]  # 3, 6, 9, ..., 45
    for _ in range(types_per_category):
        if len(mult3) >= 6:
            biased_sets.append(sorted(_rng.choice(mult3, size=6, replace=False).tolist()))

    # 9. 5의 배수만
    mult5 = [n for n in range(5, 46, 5)]  # 5, 10, 15, ..., 45
    for _ in range(types_per_category):
        if len(mult5) >= 6:
            biased_sets.append(sorted(_rng.choice(mult5, size=6, replace=False).tolist()))

    # 10. 극단적 범위 (처음 3개 + 마지막 3개)
    for _ in range(types_per_category):
        low_3 = _rng.choice(range(1, 8), size=3, replace=False).tolist()
        high_3 = _rng.choice(range(38, 46), size=3, replace=False).tolist()
        biased_sets.append(sorted(low_3 + high_3))

    # ===== 미묘한 편향 (사람이 실제로 선택할 법한 패턴) =====

    # 11. 짝수 4개 + 홀수 2개 (완전 짝수보다 미묘함)
    for _ in range(types_per_category):
        even_4 = _rng.choice(evens, size=4, replace=False).tolist()
        odd_2 = _rng.choice(odds, size=2, replace=False).tolist()
        biased_sets.append(sorted(even_4 + odd_2))

    # 12. 생일 번호 (1-31만, 많은 사람이 생일로 선택)
    birthday_nums = list(range(1, 32))
    for _ in range(types_per_category):
        biased_sets.append(sorted(_rng.choice(birthday_nums, size=6, replace=False).tolist()))

    # 13. 저수 4개 + 고수 2개 (한쪽에 치우침)
    for _ in range(types_per_category):
        low_4 = _rng.choice(low_nums, size=4, replace=False).tolist()
        high_2 = _rng.choice(high_nums, size=2, replace=False).tolist()
        biased_sets.append(sorted(low_4 + high_2))

    # 14. 연속 2-3개 포함 (자동차 번호판 패턴)
    for _ in range(types_per_category):
        start = _rng.integers(1, 43)
        consecutive = list(range(start, start + 3))
        # 나머지 3개는 랜덤
        remaining = [n for n in range(1, 46) if n not in consecutive]
        others = _rng.choice(remaining, size=3, replace=False).tolist()
        biased_sets.append(sorted(consecutive + others))

    # 15. 같은 숫자 2개 (전화번호 끝자리 패턴)
    for _ in range(types_per_category):
        same_last = _rng.integers(0, 10)
        candidates = [n for n in range(1, 46) if n % 10 == same_last]
        if len(candidates) >= 2:
            same_2 = _rng.choice(candidates, size=2, replace=False).tolist()
            remaining = [n for n in range(1, 46) if n not in same_2]
            others = _rng.choice(remaining, size=4, replace=False).tolist()
            biased_sets.append(sorted(same_2 + others))

    # 16. 행운의 숫자 포함 (3, 7, 8, 9 중 3개 이상)
    lucky_nums = [3, 7, 8, 9, 13, 21, 27, 33]
    for _ in range(types_per_category):
        lucky_3 = _rng.choice(lucky_nums, size=min(3, len(lucky_nums)), replace=False).tolist()
        remaining = [n for n in range(1, 46) if n not in lucky_3]
        others = _rng.choice(remaining, size=6-len(lucky_3), replace=False).tolist()
        biased_sets.append(sorted(lucky_3 + others))

    # 17. 대칭 패턴 (예: 1,2,44,45 + 중간 2개)
    for _ in range(types_per_category):
        edge_4 = [1, 2, 44, 45]
        middle = _rng.choice(range(20, 26), size=2, replace=False).tolist()
        biased_sets.append(sorted(edge_4 + middle))

    # 18. 피보나치 수열 일부 (1,2,3,5,8,13,21,34)
    fib_nums = [1, 2, 3, 5, 8, 13, 21, 34]
    for _ in range(types_per_category):
        if len(fib_nums) >= 6:
            biased_sets.append(sorted(_rng.choice(fib_nums, size=6, replace=False).tolist()))

    # 19. 제곱수 선호 (1,4,9,16,25,36)
    square_nums = [1, 4, 9, 16, 25, 36]
    for _ in range(types_per_category):
        if len(square_nums) >= 4:
            square_4 = _rng.choice(square_nums, size=4, replace=False).tolist()
            remaining = [n for n in range(1, 46) if n not in square_4]
            others = _rng.choice(remaining, size=2, replace=False).tolist()
            biased_sets.append(sorted(square_4 + others))

    # 20. 소수만 (2,3,5,7,11,13,17,19,23,29,31,37,41,43)
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
    prime_nums = [n for n in range(2, 46) if is_prime(n)]
    for _ in range(types_per_category):
        if len(prime_nums) >= 6:
            biased_sets.append(sorted(_rng.choice(prime_nums, size=6, replace=False).tolist()))

    # n_sets 개수에 맞춰 조정
    if len(biased_sets) < n_sets:
        # 부족하면 추가 생성 (20가지 유형 반복)
        while len(biased_sets) < n_sets:
            type_choice = _rng.integers(0, 20)
            if type_choice == 0:
                biased_sets.append(sorted(_rng.choice(evens, size=6, replace=False).tolist()))
            elif type_choice == 1:
                biased_sets.append(sorted(_rng.choice(odds, size=6, replace=False).tolist()))
            elif type_choice == 11:
                even_4 = _rng.choice(evens, size=4, replace=False).tolist()
                odd_2 = _rng.choice(odds, size=2, replace=False).tolist()
                biased_sets.append(sorted(even_4 + odd_2))
            elif type_choice == 12:
                biased_sets.append(sorted(_rng.choice(birthday_nums, size=6, replace=False).tolist()))
            else:
                # 기본적으로 명백한 편향 중 하나
                start = _rng.integers(1, 40)
                biased_sets.append(list(range(start, start + 6)))

    # n_sets 개수만큼 반환
    return biased_sets[:n_sets]

# ------------------ 기본/패턴 생성기 ------------------
def generate_random_sets(
    n_sets: int,
    avoid_duplicates: bool = True,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[list[int]]:
    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    xs = np.arange(1, 46)
    probs = None
    if weights is not None:
        w = np.array(weights, dtype=float)
        w = np.maximum(w, 1e-9)
        probs = w / w.sum()

    while len(result) < n_sets:
        if probs is None:
            pick = _rng.choice(xs, size=6, replace=False).tolist()
        else:
            p = probs.copy()
            if exclude_set:
                for v in exclude_set:
                    if 1 <= v <= 45:
                        p[v - 1] = 0.0
                s = p.sum()
                if s > 0:
                    p = p / s
            chosen: list[int] = []
            for _ in range(6):
                if p.sum() <= 0:
                    break
                v = int(_rng.choice(xs, p=p))
                chosen.append(v)
                p[v - 1] = 0.0
                s = p.sum()
                if s > 0:
                    p = p / s
            pick = chosen

        sset = sorted(set(pick))
        if len(sset) != 6:
            continue
        if exclude_set and any(v in exclude_set for v in sset):
            continue
        t = tuple(sset)
        if avoid_duplicates and t in seen:
            continue
        seen.add(t)
        result.append(sset)
    return result


def generate_pattern_sets(
    n_sets: int,
    even_target: int | None = None,
    low_mid_high=(2, 2, 2),
    include_multiples=(0, 0),
    avoid_duplicates: bool = True,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[list[int]]:
    if sum(low_mid_high) != 6:
        raise ValueError("구간 합계가 6이어야 합니다.")

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    xs = np.arange(1, 46)
    w = np.array(weights if weights is not None else np.ones(45), dtype=float)
    w = np.maximum(w, 1e-9)

    def draw_from(subset, k):
        sup = np.array(subset, dtype=int)
        p = w[sup - 1].astype(float)
        if p.sum() == 0:
            p = np.ones_like(p, dtype=float)
        p = p / p.sum()
        chosen: list[int] = []
        sup_list = sup.tolist()
        p_list = p.tolist()
        for _ in range(k):
            c = int(_rng.choice(np.array(sup_list), p=np.array(p_list)))
            chosen.append(c)
            idx = sup_list.index(c)
            sup_list.pop(idx)
            p_list.pop(idx)
            if len(p_list) > 0:
                s = sum(p_list)
                if s == 0:
                    p_list = [1 / len(p_list)] * len(p_list)
                else:
                    p_list = [pi / s for pi in p_list]
        return chosen

    lows = list(range(1, 21))
    mids = list(range(21, 36))
    highs = list(range(36, 46))

    while len(result) < n_sets:
        pool: list[int] = []
        pool.extend(draw_from(lows, low_mid_high[0]))
        pool.extend(draw_from(mids, low_mid_high[1]))
        pool.extend(draw_from(highs, low_mid_high[2]))
        sset = sorted(pool)

        if even_target is not None:
            evens = sum(1 for x in sset if x % 2 == 0)
            if evens != even_target:
                continue

        if include_multiples[0] > 0 and sum(1 for x in sset if x % 3 == 0) < include_multiples[0]:
            continue
        if include_multiples[1] > 0 and sum(1 for x in sset if x % 7 == 0) < include_multiples[1]:
            continue

        t = tuple(sset)
        if exclude_set and any(v in exclude_set for v in t):
            continue
        if avoid_duplicates and t in seen:
            continue

        seen.add(t)
        result.append(list(sset))

    return result


# ------------------ 기존 천재 알고리듬 ------------------
def gen_GI(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    out: list[list[int]] = []
    xs = np.arange(1, 46)
    w = None
    if weights is not None:
        w = np.maximum(np.array(weights, dtype=float), 1e-9)
        if exclude_set:
            for v in exclude_set:
                if 1 <= v <= 45:
                    w[v - 1] = 0.0
        s = w.sum()
        w = w / s if s > 0 else None

    for _ in range(n_sets):
        start = _rng.integers(1, 10)
        gaps = [7, 7, 6 + _rng.integers(-1, 2), 6 + _rng.integers(-1, 2), 8 + _rng.integers(-1, 2)]
        seq = [start]
        for g in gaps:
            seq.append(seq[-1] + g)
        seq = [min(max(1, x), 45) for x in seq]
        seq = sorted(set(seq))
        while len(seq) < 6:
            cand = int(_rng.choice(xs, p=w)) if w is not None else int(_rng.integers(20, 36))
            if cand not in seq and (not exclude_set or cand not in exclude_set):
                seq.append(cand)
                seq.sort()
        out.append(seq[:6])
    return out


def gen_MDA(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    return generate_pattern_sets(
        n_sets,
        even_target=3,
        low_mid_high=(2, 2, 2),
        include_multiples=(0, 0),
        weights=weights,
        exclude_set=exclude_set,
    )


def gen_CC(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    out: list[list[int]] = []
    base3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
    base5 = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    xs = np.arange(1, 46)
    w = None
    if weights is not None:
        w = np.maximum(np.array(weights, dtype=float), 1e-9)
        if exclude_set:
            for v in exclude_set:
                if 1 <= v <= 45:
                    w[v - 1] = 0.0
        s = w.sum()
        w = w / s if s > 0 else None

    for _ in range(n_sets):
        s_set: set[int] = set()
        s_set.update(_rng.choice(base3, size=2, replace=False).tolist())
        s_set.update(_rng.choice(base5, size=2, replace=False).tolist())
        while len(s_set) < 6:
            cand = int(_rng.choice(xs, p=w)) if w is not None else int(_rng.integers(1, 46))
            if cand % 3 != 0 and cand % 5 != 0 and (exclude_set is None or cand not in exclude_set):
                s_set.add(cand)
        out.append(sorted(s_set))
    return out


def gen_PR(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    seeds = [
        [7, 14, 21, 27, 33, 41],
        [6, 13, 22, 29, 35, 44],
        [5, 8, 13, 21, 34, 45],
    ]
    xs = np.arange(1, 46)
    w = None
    if weights is not None:
        w = np.maximum(np.array(weights, dtype=float), 1e-9)
        if exclude_set:
            for v in exclude_set:
                if 1 <= v <= 45:
                    w[v - 1] = 0.0
        s = w.sum()
        w = w / s if s > 0 else None
    out: list[list[int]] = []
    for _ in range(n_sets):
        base = np.array(seeds[_rng.integers(0, len(seeds))], dtype=int)
        jitter = _rng.integers(-2, 3, size=6)
        arr = np.clip(base + jitter, 1, 45).tolist()
        arr = sorted(set(arr))
        while len(arr) < 6:
            cand = int(_rng.choice(xs, p=w)) if w is not None else int(_rng.integers(1, 46))
            if cand not in arr and (exclude_set is None or cand not in exclude_set):
                arr.append(cand)
                arr = sorted(arr)
        out.append(arr[:6])
    return out


def gen_IS(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    xs = np.arange(1, 46)
    prior = np.exp(-((xs - 28.0) ** 2) / (2 * 9.0))
    prior += (xs % 3 == 0).astype(float) * 0.35 + (xs % 7 == 0).astype(float) * 0.25 + 0.05
    if weights is not None:
        hist = np.maximum(np.array(weights, dtype=float), 1e-9)
        hist = hist / hist.sum()
        p = 0.5 * prior / prior.sum() + 0.5 * hist
    else:
        p = prior / prior.sum()
    if exclude_set:
        for v in exclude_set:
            if 1 <= v <= 45:
                p[v - 1] = 0.0
    if p.sum() == 0:
        p = np.ones_like(p) / len(p)

    out: list[list[int]] = []
    for _ in range(n_sets):
        s_set: set[int] = set()
        while len(s_set) < 6:
            pick = int(_rng.choice(xs, p=p))
            s_set.add(pick)
        out.append(sorted(s_set))
    return out


# ------------------ GAP-R ------------------
def gen_GAPR(
    n_sets: int,
    history_df: pd.DataFrame | None = None,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[list[int]]:
    if history_df is None or history_df.empty:
        return gen_GI(n_sets, weights=weights, exclude_set=exclude_set)

    gap_list: list[list[int]] = []
    for row in history_df.itertuples(index=False):
        nums = sorted(list(row))
        if len(nums) != 6:
            continue
        gaps = [nums[i + 1] - nums[i] for i in range(5)]
        if all(g > 0 for g in gaps):
            gap_list.append(gaps)
    if not gap_list:
        return gen_GI(n_sets, weights=weights, exclude_set=exclude_set)

    seen: set[tuple[int, ...]] = set()
    result: list[list[int]] = []

    while len(result) < n_sets:
        gaps = gap_list[_rng.integers(0, len(gap_list))]
        success = False
        for _try in range(50):
            start = int(_rng.integers(1, 11))
            seq = [start]
            for g in gaps:
                seq.append(seq[-1] + g)
            if any(x < 1 or x > 45 for x in seq):
                continue
            seq_sorted = sorted(set(seq))
            if len(seq_sorted) != 6:
                continue
            if exclude_set and any(v in exclude_set for v in seq_sorted):
                continue
            t = tuple(seq_sorted)
            if t in seen:
                continue
            seen.add(t)
            result.append(seq_sorted)
            success = True
            break
        if not success:
            fb = generate_random_sets(1, avoid_duplicates=True, weights=weights, exclude_set=exclude_set)[0]
            t = tuple(sorted(fb))
            if t not in seen:
                seen.add(t)
                result.append(sorted(fb))
    return result


# ------------------ QH 스코어 ------------------
def _qh_score(nums: list[int], weights=None) -> float:
    nums = sorted(nums)
    if len(nums) != 6:
        return 0.0
    evens = sum(1 for v in nums if v % 2 == 0)
    parity_score = 1.0 - abs(evens - 3) / 3.0

    low = sum(1 for v in nums if 1 <= v <= 20)
    mid = sum(1 for v in nums if 21 <= v <= 35)
    high = sum(1 for v in nums if 36 <= v <= 45)
    range_dev = abs(low - 2) + abs(mid - 2) + abs(high - 2)
    range_score = 1.0 - range_dev / 6.0

    gaps = np.diff(np.array(nums))
    if len(gaps) > 0:
        gap_score = 1.0 - abs(float(gaps.mean()) - 8.0) / 8.0
    else:
        gap_score = 0.0

    if weights is not None:
        w_arr = np.array(weights, dtype=float)
        loc = np.array([w_arr[v - 1] for v in nums])
        hist_score = float(loc.mean()) * len(w_arr)
    else:
        hist_score = 1.0

    clamp = lambda x: max(0.0, min(1.0, x))
    return (
        0.3 * clamp(parity_score)
        + 0.3 * clamp(range_score)
        + 0.2 * clamp(gap_score)
        + 0.2 * clamp(hist_score)
    )


def gen_QH(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    xs = np.arange(1, 46)
    if weights is not None:
        base_p = np.maximum(np.array(weights, dtype=float), 1e-9)
        if exclude_set:
            for v in exclude_set:
                if 1 <= v <= 45:
                    base_p[v - 1] = 0.0
        s = base_p.sum()
        base_p = (np.ones(45) / 45.0) if s <= 0 else (base_p / s)
    else:
        base_p = np.ones(45) / 45.0

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_iter = 80

    while len(result) < n_sets:
        p = base_p.copy()
        cur_set: set[int] = set()
        while len(cur_set) < 6:
            if p.sum() <= 0:
                p = np.ones(45) / 45.0
            v = int(_rng.choice(xs, p=p))
            if exclude_set and v in exclude_set:
                continue
            if v in cur_set:
                continue
            cur_set.add(v)
            p[v - 1] = 0.0
            s = p.sum()
            if s > 0:
                p = p / s
        cur = sorted(cur_set)
        best = _qh_score(cur, weights)
        for _ in range(max_iter):
            cand = set(cur)
            old = int(_rng.choice(list(cand)))
            cand.remove(old)
            p = base_p.copy()
            for u in cand:
                p[u - 1] = 0.0
            if exclude_set:
                for ex in exclude_set:
                    if 1 <= ex <= 45:
                        p[ex - 1] = 0.0
            if p.sum() <= 0:
                continue
            p = p / p.sum()
            new_v = int(_rng.choice(xs, p=p))
            cand.add(new_v)
            cand = sorted(cand)
            sc = _qh_score(cand, weights)
            if sc > best:
                cur = cand
                best = sc
        t = tuple(cur)
        if t in seen:
            continue
        if exclude_set and any(v in exclude_set for v in cur):
            continue
        seen.add(t)
        result.append(cur)
    return result


# ------------------ AI 세트 평점용 특징/모델 ------------------
def _set_features(
    nums: list[int],
    weights=None,
    history_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    번호 조합의 특징 추출 (30개 특징)

    기존 10개 + 신규 20개 = 총 30개 특징
    """
    nums = sorted(nums)
    arr = np.array(nums, dtype=float)

    # ===== 기존 특징 (10개) =====
    f_mean = arr.mean() / 45.0
    f_std = arr.std() / 20.0

    evens = sum(1 for v in nums if v % 2 == 0) / 6.0
    low = sum(1 for v in nums if 1 <= v <= 15) / 6.0
    mid = sum(1 for v in nums if 16 <= v <= 30) / 6.0
    high = sum(1 for v in nums if 31 <= v <= 45) / 6.0

    gaps = np.diff(arr)
    if len(gaps) > 0:
        f_gmean = (gaps.mean() - 8.0) / 8.0
        f_gstd = gaps.std() / 10.0
    else:
        f_gmean = 0.0
        f_gstd = 0.0

    if weights is not None:
        w_arr = np.array(weights, dtype=float)
        ww = np.array([w_arr[int(v) - 1] for v in nums])
        f_hmean = float(ww.mean()) * len(w_arr)
        f_hmax = float(ww.max()) * len(w_arr)
    else:
        f_hmean = 0.0
        f_hmax = 0.0

    # ===== 신규 특징 (20개) =====

    # 통계적 특징 (5개)
    f_min = arr.min() / 45.0
    f_max = arr.max() / 45.0
    f_median = float(np.median(arr)) / 45.0
    f_range = (arr.max() - arr.min()) / 45.0
    q1, q3 = np.percentile(arr, [25, 75])
    f_iqr = (q3 - q1) / 45.0

    # 번호 패턴 특징 (8개)
    # 연속 번호 개수
    consecutive_count = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    f_consecutive = consecutive_count / 5.0

    # 최대 연속 길이
    max_consecutive = 1
    current_consecutive = 1
    for i in range(len(nums)-1):
        if nums[i+1] - nums[i] == 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    f_max_consecutive = max_consecutive / 6.0

    # 끝자리 다양성
    last_digits = [n % 10 for n in nums]
    unique_last_digits = len(set(last_digits))
    f_last_digit_diversity = unique_last_digits / 6.0
    f_last_digit_dup = (6 - unique_last_digits) / 6.0

    # 배수 개수
    f_mult3 = sum(1 for n in nums if n % 3 == 0) / 6.0
    f_mult5 = sum(1 for n in nums if n % 5 == 0) / 6.0

    # 소수 개수
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
    f_primes = sum(1 for n in nums if is_prime(n)) / 6.0

    # 대칭성 (중앙값 23 기준)
    center = 23.0
    symmetry_errors = sum(abs((n - center) + (center - (46 - n))) for n in nums)
    f_symmetry = max(0.0, 1.0 - (symmetry_errors / (6 * 45)))

    # 간격 패턴 특징 (4개)
    if len(gaps) > 0:
        f_gap_min = gaps.min() / 10.0
        f_gap_max = gaps.max() / 10.0
        f_gap_median = float(np.median(gaps)) / 10.0
        f_gap_cv = (gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    else:
        f_gap_min = f_gap_max = f_gap_median = f_gap_cv = 0.0

    # 확률적 특징 (2개) - 편향 제거를 위해 주석 처리
    # f_freq_avg = 0.0
    # f_recent = 0.0
    # if history_df is not None and not history_df.empty:
    #     # 각 번호의 출현 빈도
    #     all_nums = []
    #     for row in history_df.itertuples(index=False):
    #         for val in row:
    #             try:
    #                 v = int(val)
    #                 if 1 <= v <= 45:
    #                     all_nums.append(v)
    #             except (ValueError, TypeError):
    #                 continue
    #
    #     if all_nums:
    #         from collections import Counter
    #         freq_counter = Counter(all_nums)
    #         avg_freq = np.mean([freq_counter.get(n, 0) for n in nums])
    #         max_freq = max(freq_counter.values()) if freq_counter else 1
    #         f_freq_avg = avg_freq / max_freq if max_freq > 0 else 0.0
    #
    #         # 최근 10회 출현도
    #         recent_nums = set()
    #         for row in history_df.head(10).itertuples(index=False):
    #             for val in row:
    #                 try:
    #                     v = int(val)
    #                     if 1 <= v <= 45:
    #                         recent_nums.add(v)
    #                 except (ValueError, TypeError):
    #                     continue
    #         f_recent = sum(1 for n in nums if n in recent_nums) / 6.0

    # 고차원 특징 (1개)
    f_sum_last_digit = (sum(nums) % 10) / 10.0

    # ===== 무작위성 특징 (5개 추가) =====

    # 1. 숫자 엔트로피 (Entropy) - 숫자 분포의 다양성
    unique_tens = len(set(n // 10 for n in nums))  # 십의 자리 다양성 (0-4)
    unique_ones = len(set(n % 10 for n in nums))   # 일의 자리 다양성 (0-9)
    f_entropy = (unique_tens / 5.0 + unique_ones / 10.0) / 2.0

    # 2. 카이제곱 균등성 (Chi-Square Uniformity) - 번호가 전체 범위에 고르게 분포하는지
    # 1-45를 3개 구간으로 나눔: 1-15, 16-30, 31-45
    observed = np.array([low * 6, mid * 6, high * 6])  # 실제 개수
    expected = np.array([2.0, 2.0, 2.0])  # 기대 개수 (6개/3구간=2개)
    chi2 = np.sum((observed - expected) ** 2 / (expected + 0.01))
    f_chi2_uniformity = 1.0 / (1.0 + chi2 / 5.0)  # 0-1 정규화 (낮을수록 균등)

    # 3. 런 테스트 (Runs Test) - 순서의 무작위성
    # 중앙값(23)보다 큰/작은 것으로 이진화
    binary = [1 if n > 23 else 0 for n in nums]
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    # 무작위라면 runs는 3-4개 정도 (최소 1, 최대 6)
    expected_runs = 3.5
    f_runs = 1.0 - min(1.0, abs(runs - expected_runs) / 3.5)

    # 4. 자기상관 (Autocorrelation) - 간격의 규칙성
    if len(gaps) > 1:
        # 간격의 표준편차가 클수록 무작위적
        f_autocorr = min(1.0, gaps.std() / 10.0)
    else:
        f_autocorr = 0.0

    # 5. 비트 엔트로피 (Bit Entropy) - 2진수 표현의 균형
    # 모든 숫자를 6비트로 표현하여 0과 1의 개수 비교
    all_bits = []
    for n in nums:
        bits = format(n, '06b')  # 6비트 표현 (0-63)
        all_bits.extend([int(b) for b in bits])
    ones = sum(all_bits)
    zeros = len(all_bits) - ones
    # 50:50에 가까울수록 1.0
    if max(ones, zeros) > 0:
        f_bit_entropy = min(ones, zeros) / max(ones, zeros)
    else:
        f_bit_entropy = 0.0

    # ===== 특징 벡터 구성 (25개 - 무작위성 특징 추가) =====
    # 제거된 특징 (큰 번호 편향):
    # - f_mean: 큰 번호에 유리
    # - f_max: 45번에 유리
    # - f_hmax: 큰 번호 가중치에 유리
    # - f_gmean: 큰 번호에 유리
    # - f_gstd: 간격 편향
    # - f_hmean: 가중치 편향
    # - f_freq_avg: 히스토리 편향
    # - f_recent: 최근 편향
    # - f_median: 큰 번호에 유리
    # - f_min: 작은 번호에 유리

    feats = np.array(
        [
            # 기본 분포 특징 (5개) - 중립적
            f_std, evens, low, mid, high,
            # low/mid/high는 유지 (균형 체크용)
            # 범위 재조정: 1-15 (저) / 16-30 (중) / 31-45 (고)

            # 통계적 특징 (2개) - 중립적
            f_range, f_iqr,

            # 번호 패턴 8개 - 중립적
            f_consecutive, f_max_consecutive,
            f_last_digit_diversity, f_last_digit_dup,
            f_mult3, f_mult5, f_primes, f_symmetry,

            # 간격 패턴 4개 - 중립적
            f_gap_min, f_gap_max, f_gap_median, f_gap_cv,

            # 고차원 1개 - 중립적
            f_sum_last_digit,

            # 무작위성 특징 5개 - 무작위성 측정
            f_entropy, f_chi2_uniformity, f_runs, f_autocorr, f_bit_entropy,
        ],
        dtype=float,
    )
    return feats


def train_ml_scorer(
    history_df: pd.DataFrame,
    weights=None,
    n_neg_per_pos: int = 5,
    max_rounds: int | None = 200,
    epochs: int = 120,
    lr: float = 0.05,
    use_hard_negatives: bool = True,
    model_type: str = "neural_network",  # 고정: Neural Network만 사용
    randomness_learning: bool = True,  # 랜덤성 학습 모드
) -> dict:
    """
    AI 세트 평점 학습 - Neural Network만 사용 (최고 성능)

    Parameters:
        history_df: 과거 당첨 번호 데이터프레임
        weights: 번호 가중치 (45개)
        n_neg_per_pos: 양성 샘플당 음성 샘플 비율
        max_rounds: 사용할 최근 회차 수 (None=전체)
        epochs: 학습 반복 횟수
        lr: 학습률
        use_hard_negatives: 하드 네거티브 샘플링 (랜덤성 학습 모드에서는 무시됨)
        model_type: 항상 "neural_network" (다른 값 지정해도 무시됨)
        randomness_learning: True면 "무작위성" 학습 (양성=당첨번호, 음성=편향조합)
                            False면 기존 방식 (양성=당첨번호, 음성=랜덤조합)

    Returns:
        학습된 Neural Network 모델 dict
    """
    if history_df is None or history_df.empty:
        raise ValueError("히스토리 없음: ML 학습 불가")

    if max_rounds is None or max_rounds <= 0:
        df = history_df
    else:
        df = history_df.tail(max_rounds)

    pos_sets = []
    for row in df.itertuples(index=False):
        nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
        if len(nums) == 6:
            pos_sets.append(nums)
    if not pos_sets:
        raise ValueError("유효한 양성 세트가 없습니다.")

    X_list = []
    y_list = []

    # 양성 샘플
    for s in pos_sets:
        X_list.append(_set_features(s, weights, history_df))
        y_list.append(1.0)

    # 음성 샘플 생성
    n_neg = len(pos_sets) * n_neg_per_pos

    if randomness_learning:
        # 랜덤성 학습 모드: 편향된 조합을 음성 샘플로 사용
        print(f"[ML 학습] 랜덤성 학습 모드 - 음성 샘플: 편향된 조합")
        neg_sets = generate_biased_combinations(n_neg)
    elif use_hard_negatives:
        # 하드 네거티브 샘플링: 50% 랜덤 + 50% 변형
        n_random = n_neg // 2
        n_mutated = n_neg - n_random

        # 1) 완전 랜덤 음성 샘플
        neg_sets = generate_random_sets(n_random, avoid_duplicates=True, weights=weights, exclude_set=None)

        # 2) 하드 네거티브: 양성 샘플을 약간 변형 (1-2개 번호만 변경)
        # → 모델이 세밀한 차이를 학습하도록 유도
        for _ in range(n_mutated):
            # 무작위로 양성 샘플 선택
            base = pos_sets[_rng.integers(0, len(pos_sets))].copy()

            # 1-2개 번호 변경
            num_changes = _rng.integers(1, 3)  # 1 or 2
            for _ in range(num_changes):
                # 기존 번호 중 하나 제거
                remove_idx = _rng.integers(0, len(base))
                old_num = base[remove_idx]

                # 새 번호 선택 (기존에 없는 번호)
                available = [n for n in range(1, 46) if n not in base]
                if available:
                    new_num = _rng.choice(available)
                    base[remove_idx] = new_num

            neg_sets.append(sorted(base))
    else:
        # 기존 방식: 완전 랜덤만
        neg_sets = generate_random_sets(n_neg, avoid_duplicates=True, weights=weights, exclude_set=None)

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

    print(f"[ML 학습] 샘플: {N}개 (양성: {len(pos_sets)}, 음성: {len(neg_sets)}), 특징: {D}개")
    if randomness_learning:
        print(f"[ML 학습] 모드: 랜덤성 학습 (양성=진짜무작위, 음성=편향조합)")
    else:
        print(f"[ML 학습] 모드: 분류 학습 (하드네거티브: {use_hard_negatives})")

    # Neural Network만 사용 (최고 성능)
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
    except ImportError:
        raise ImportError("[오류] scikit-learn 필요. pip install scikit-learn")

    # Neural Network 모델 생성
    mode_str = "랜덤성학습" if randomness_learning else "분류학습"
    print(f"[ML 학습] 모델: 신경망 ({mode_str}, 50-30-10 layers)")
    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(50, 30, 10),
        max_iter=200,
        learning_rate_init=0.01,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    # 학습
    print(f"[ML 학습] 샘플: {N}개, 특징: {D}개")
    sklearn_model.fit(Xn, y)

    # 교차 검증 (Stratified K-Fold)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(sklearn_model, Xn, y, cv=skf)
    train_acc = sklearn_model.score(Xn, y)

    print(f"[ML 학습 완료] 훈련 정확도: {train_acc:.2%}")
    print(f"[교차 검증] 평균: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # 과적합 탐지
    overfitting_gap = train_acc - cv_scores.mean()
    if overfitting_gap > 0.10:
        print(f"[경고] 과적합 의심! (Gap: {overfitting_gap:.2%})")
    elif overfitting_gap > 0.05:
        print(f"[주의] 약간의 과적합 (Gap: {overfitting_gap:.2%})")

    model = {
        "type": "neural_network",
        "sklearn_model": sklearn_model,
        "mu": mu,
        "sigma": sigma,
        "accuracy": float(train_acc),
        "cv_scores": cv_scores.tolist(),
        "n_features": D,
    }
    return model


def ml_score_set(
    nums: list[int],
    model: dict | None,
    weights=None,
    history_df: pd.DataFrame | None = None,
) -> float:
    """
    ML 모델로 번호 조합 점수 계산

    모든 모델 타입 지원:
    - logistic: 로지스틱 회귀
    - random_forest: 랜덤 포레스트
    - gradient_boosting: 그래디언트 부스팅
    - neural_network: 신경망
    """
    if model is None:
        return 0.0

    # 특징 추출
    feats = _set_features(nums, weights, history_df)
    mu = model.get("mu")
    sig = model.get("sigma")
    if mu is None or sig is None:
        return 0.0

    # 정규화
    x = (feats - mu) / sig
    x = x.reshape(1, -1)  # sklearn 호환 shape

    # 모델 타입에 따라 다른 방식으로 점수 계산
    model_type = model.get("type", "logistic")

    if model_type == "logistic":
        # 로지스틱 회귀
        w = model.get("w")
        b = model.get("b", 0.0)
        if w is None:
            return 0.0
        z = float(np.dot(w, x.flatten()) + b)
        s = 1.0 / (1.0 + np.exp(-z))
        return s

    elif model_type in ["random_forest", "gradient_boosting", "neural_network"]:
        # sklearn 모델
        sklearn_model = model.get("sklearn_model")
        if sklearn_model is None:
            return 0.0

        try:
            # predict_proba로 확률 반환 (클래스 1의 확률)
            proba = sklearn_model.predict_proba(x)[0, 1]
            return float(proba)
        except Exception:
            # predict_proba 실패 시 predict 사용
            pred = sklearn_model.predict(x)[0]
            return float(pred)

    else:
        return 0.0


# ------------------ HD(초다양성) ------------------
def gen_HD(
    n_sets: int,
    base_sets: list[list[int]] | None = None,
    weights=None,
    exclude_set: set[int] | None = None,
) -> list[list[int]]:
    xs = np.arange(1, 46)
    if weights is not None:
        base_p = np.maximum(np.array(weights, dtype=float), 1e-9)
        if exclude_set:
            for v in exclude_set:
                if 1 <= v <= 45:
                    base_p[v - 1] = 0.0
        s = base_p.sum()
        base_p = (np.ones(45) / 45.0) if s <= 0 else (base_p / s)
    else:
        base_p = np.ones(45) / 45.0
    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    base_list: list[set[int]] = []
    if base_sets:
        for s in base_sets:
            if len(s) == 6:
                base_list.append(set(s))

    def sample_one() -> list[int]:
        p = base_p.copy()
        cur_set: set[int] = set()
        while len(cur_set) < 6:
            if p.sum() <= 0:
                p = np.ones(45) / 45.0
            v = int(_rng.choice(xs, p=p))
            if exclude_set and v in exclude_set:
                continue
            if v in cur_set:
                continue
            cur_set.add(v)
            p[v - 1] = 0.0
            s = p.sum()
            if s > 0:
                p = p / s
        return sorted(cur_set)

    def diversity_score(nums: list[int]) -> int:
        s = set(nums)
        score = 0
        for bs in base_list:
            score += len(s & bs)
        for rs in result:
            score += len(s & set(rs))
        return score

    for _ in range(n_sets):
        best = None
        best_sc = None
        for _c in range(40):
            cand = sample_one()
            t = tuple(cand)
            if t in seen:
                continue
            if exclude_set and any(v in exclude_set for v in cand):
                continue
            sc = diversity_score(cand)
            if best_sc is None or sc < best_sc:
                best_sc = sc
                best = cand
        if best is None:
            best = sample_one()
        t = tuple(best)
        seen.add(t)
        result.append(best)
        base_list.append(set(best))
    return result


# ------------------ QP 계열 ------------------
def gen_QP(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    xs = np.arange(1, 46)
    base_w = np.maximum(np.array(weights, dtype=float), 1e-12) if weights is not None else np.ones(45)
    base_w = base_w / base_w.sum()
    psi = np.sqrt(base_w).astype(complex)
    phases = np.exp(1j * _rng.uniform(0.0, 2.0 * np.pi, size=45))
    Psi = psi * phases
    spec = np.fft.fft(Psi)
    spec2 = spec ** 2
    Psi2 = np.fft.ifft(spec2)
    prob = np.abs(Psi2)
    prob = prob if prob.sum() > 0 else base_w.copy()
    prob = prob / prob.sum()

    if exclude_set:
        for v in exclude_set:
            if 1 <= v <= 45:
                prob[v - 1] = 0.0
        s = prob.sum()
        if s <= 0:
            prob = np.ones(45)
            for v in exclude_set:
                if 1 <= v <= 45:
                    prob[v - 1] = 0.0
            s = prob.sum()
            prob = (np.ones(45) / 45.0) if s <= 0 else (prob / s)
        else:
            prob = prob / s

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(result) < n_sets:
        p = prob.copy()
        chosen: list[int] = []
        for _ in range(6):
            if p.sum() <= 0:
                break
            v = int(_rng.choice(xs, p=p))
            chosen.append(v)
            p[v - 1] = 0.0
            s = p.sum()
            if s > 0:
                p = p / s
        if len(chosen) != 6:
            continue
        sset = sorted(set(chosen))
        if len(sset) != 6:
            continue
        if exclude_set and any(v in exclude_set for v in sset):
            continue
        t = tuple(sset)
        if t in seen:
            continue
        seen.add(t)
        result.append(sset)
    return result


def gen_QP_tunnel(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    xs = np.arange(1, 46)
    base_p = np.maximum(np.array(weights, dtype=float), 1e-12) if weights is not None else np.ones(45)
    base_p = base_p / base_p.sum()
    if exclude_set:
        mask = np.ones(45)
        for v in exclude_set:
            if 1 <= v <= 45:
                mask[v - 1] = 0.08
        base_p = base_p * mask
        if base_p.sum() <= 0:
            base_p = mask
        if base_p.sum() <= 0:
            base_p = np.ones(45)
        base_p = base_p / base_p.sum()

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(result) < n_sets:
        p = base_p.copy()
        chosen: list[int] = []
        for _ in range(6):
            if p.sum() <= 0:
                break
            v = int(_rng.choice(xs, p=p))
            chosen.append(v)
            p[v - 1] = 0.0
            s = p.sum()
            if s > 0:
                p = p / s
        if len(chosen) != 6:
            continue
        sset = sorted(set(chosen))
        if len(sset) != 6:
            continue
        t = tuple(sset)
        if t in seen:
            continue
        seen.add(t)
        result.append(sset)
    return result


def gen_QP_entangle(
    n_sets: int,
    history_df: pd.DataFrame | None = None,
    weights=None,
    exclude_set: set[int] | None = None,
):
    if history_df is None or history_df.empty:
        return gen_QP(n_sets, weights=weights, exclude_set=exclude_set)
    xs = np.arange(1, 46)
    base_p = np.maximum(np.array(weights, dtype=float), 1e-12) if weights is not None else np.ones(45)
    base_p = base_p / base_p.sum()
    if exclude_set:
        for v in exclude_set:
            if 1 <= v <= 45:
                base_p[v - 1] = 0.0
        s = base_p.sum()
        if s <= 0:
            base_p = np.ones(45)
            for v in exclude_set:
                if 1 <= v <= 45:
                    base_p[v - 1] = 0.0
            s = base_p.sum()
            if s <= 0:
                base_p = np.ones(45)
        base_p = base_p / base_p.sum()

    pair_counts = np.zeros((46, 46), dtype=float)
    for row in history_df.itertuples(index=False):
        nums = sorted(set(int(x) for x in row if 1 <= int(x) <= 45))
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                pair_counts[a, b] += 1.0
                pair_counts[b, a] += 1.0

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    beta = 1.0
    while len(result) < n_sets:
        chosen: list[int] = []
        for _idx in range(6):
            if not chosen:
                p = base_p.copy()
            else:
                pair_score = np.zeros(46)
                for s_ in chosen:
                    pair_score += pair_counts[s_]
                ps = pair_score[1:]
                if ps.max() > 0:
                    ps_norm = ps / ps.max()
                else:
                    ps_norm = np.zeros_like(ps)
                comb = base_p * (1.0 + beta * ps_norm)
                for u in chosen:
                    comb[u - 1] = 0.0
                if exclude_set:
                    for ex in exclude_set:
                        if 1 <= ex <= 45:
                            comb[ex - 1] = 0.0
                if comb.sum() <= 0:
                    comb = base_p.copy()
                    for u in chosen:
                        comb[u - 1] = 0.0
                    if exclude_set:
                        for ex in exclude_set:
                            if 1 <= ex <= 45:
                                comb[ex - 1] = 0.0
                    if comb.sum() <= 0:
                        comb = np.ones(45)
                p = comb / comb.sum()
            if p.sum() <= 0:
                break
            v = int(_rng.choice(xs, p=p))
            if v in chosen:
                continue
            chosen.append(v)

        if len(chosen) != 6:
            continue
        sset = sorted(set(chosen))
        if len(sset) != 6:
            continue
        if exclude_set and any(v in exclude_set for v in sset):
            continue
        t = tuple(sset)
        if t in seen:
            continue
        seen.add(t)
        result.append(sset)

    return result


def gen_QH_QA(n_sets: int, weights=None, exclude_set: set[int] | None = None):
    xs = np.arange(1, 46)
    if weights is not None:
        base_p = np.maximum(np.array(weights, dtype=float), 1e-9)
        if exclude_set:
            for v in exclude_set:
                if 1 <= v <= 45:
                    base_p[v - 1] = 0.0
        s = base_p.sum()
        base_p = (np.ones(45) / 45.0) if s <= 0 else (base_p / s)
    else:
        base_p = np.ones(45) / 45.0

    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    T0 = 1.0
    T_end = 0.02
    steps = 120

    while len(result) < n_sets:
        p = base_p.copy()
        cur_set: set[int] = set()
        while len(cur_set) < 6:
            if p.sum() <= 0:
                p = np.ones(45) / 45.0
            v = int(_rng.choice(xs, p=p))
            if exclude_set and v in exclude_set:
                continue
            if v in cur_set:
                continue
            cur_set.add(v)
            p[v - 1] = 0.0
            s = p.sum()
            if s > 0:
                p = p / s
        cur = sorted(cur_set)
        cur_sc = _qh_score(cur, weights)
        for step in range(steps):
            cand = set(cur)
            old = int(_rng.choice(list(cand)))
            cand.remove(old)
            p = base_p.copy()
            for u in cand:
                p[u - 1] = 0.0
            if exclude_set:
                for ex in exclude_set:
                    if 1 <= ex <= 45:
                        p[ex - 1] = 0.0
            if p.sum() <= 0:
                continue
            p = p / p.sum()
            new_v = int(_rng.choice(xs, p=p))
            cand.add(new_v)
            cand = sorted(cand)
            new_sc = _qh_score(cand, weights)
            delta = new_sc - cur_sc
            T = T0 * (T_end / T0) ** (step / max(1, steps - 1))
            if delta >= 0 or _rng.random() < np.exp(delta / max(T, 1e-6)):
                cur = cand
                cur_sc = new_sc
        t = tuple(cur)
        if t in seen:
            continue
        if exclude_set and any(v in exclude_set for v in cur):
            continue
        seen.add(t)
        result.append(cur)
    return result


def gen_QP_jump(
    n_sets: int,
    history_df: pd.DataFrame | None = None,
    weights=None,
    exclude_set: set[int] | None = None,
):
    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def pick_one():
        if history_df is not None and not history_df.empty:
            if exclude_set:
                r = _rng.random()
                mode = "tunnel" if r < 0.4 else ("entangle" if r < 0.8 else "qh_qa")
            else:
                r = _rng.random()
                mode = "wave" if r < 0.5 else ("entangle" if r < 0.8 else "qh_qa")
        else:
            r = _rng.random()
            mode = "wave" if r < 0.7 else "qh_qa"

        if mode == "wave":
            return gen_QP(1, weights=weights, exclude_set=exclude_set)[0]
        elif mode == "tunnel":
            return gen_QP_tunnel(1, weights=weights, exclude_set=exclude_set)[0]
        elif mode == "entangle":
            return gen_QP_entangle(1, history_df=history_df, weights=weights, exclude_set=exclude_set)[0]
        else:
            return gen_QH_QA(1, weights=weights, exclude_set=exclude_set)[0]

    while len(result) < n_sets:
        cand = pick_one()
        t = tuple(sorted(cand))
        if t in seen:
            continue
        if exclude_set and any(v in exclude_set for v in cand):
            continue
        seen.add(t)
        result.append(sorted(cand))
    return result


# ------------------ MQLE ------------------
def gen_MQLE(
    n_sets: int,
    history_df: pd.DataFrame | None = None,
    weights=None,
    exclude_set: set[int] | None = None,
    base_sets: list[list[int]] | None = None,
    q_balance: float = 0.6,
    ml_model: dict | None = None,
    ml_weight: float = 0.3,
):
    q_balance = float(max(0.0, min(1.0, q_balance)))
    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    base_list: list[set[int]] = []
    if base_sets:
        for s in base_sets:
            if len(s) == 6:
                base_list.append(set(s))

    def diversity_penalty(nums: list[int]) -> int:
        s = set(nums)
        score = 0
        for bs in base_list:
            score += len(s & bs)
        for rs in result:
            score += len(s & set(rs))
        return score

    def candidate_from_modes() -> list[tuple[list[int], str]]:
        cands: list[tuple[list[int], str]] = []
        try:
            cands.append((gen_QP(1, weights, exclude_set)[0], "quantum"))
        except Exception:
            pass
        try:
            cands.append((gen_QP_tunnel(1, weights, exclude_set)[0], "quantum"))
        except Exception:
            pass
        if history_df is not None and history_df.empty is False:
            try:
                cands.append(
                    (
                        gen_QP_entangle(1, history_df, weights, exclude_set)[0],
                        "quantum",
                    )
                )
            except Exception:
                pass
        try:
            cands.append((gen_QH_QA(1, weights, exclude_set)[0], "quantum"))
        except Exception:
            pass

        if history_df is not None and history_df.empty is False:
            try:
                cands.append((gen_GAPR(1, history_df, weights, exclude_set)[0], "classical"))
            except Exception:
                pass
        try:
            cands.append((gen_HD(1, base_sets, weights, exclude_set)[0], "classical"))
        except Exception:
            pass
        try:
            cands.append((gen_GI(1, weights, exclude_set)[0], "classical"))
        except Exception:
            pass
        try:
            cands.append(
                (
                    generate_pattern_sets(
                        1,
                        even_target=3,
                        low_mid_high=(2, 2, 2),
                        include_multiples=(0, 0),
                        weights=weights,
                        exclude_set=exclude_set,
                    )[0],
                    "classical",
                )
            )
        except Exception:
            pass
        try:
            cands.append((generate_random_sets(1, True, weights, exclude_set)[0], "classical"))
        except Exception:
            pass

        if not cands:
            cands.append((generate_random_sets(1, True, weights, exclude_set)[0], "classical"))
        return cands

    pref_strength = 0.4

    # ML 가중치 정규화 (0.0 ~ 1.0 범위)
    ml_weight = float(max(0.0, min(1.0, ml_weight)))

    for k in range(n_sets):
        t_ratio = k / max(1, n_sets - 1)
        alpha_qh = 0.5 + 0.4 * (1.0 - t_ratio)
        alpha_div = 0.3 + 0.4 * t_ratio
        alpha_rand = 0.2
        alpha_ml = ml_weight

        cand_list = candidate_from_modes()
        best_cand = None
        best_score = None

        for cand, tag in cand_list:
            t = tuple(sorted(cand))
            if t in seen:
                continue
            if exclude_set and any(v in exclude_set for v in cand):
                continue

            qh = _qh_score(cand, weights)
            div = diversity_penalty(cand)
            rnd = _rng.random()
            ml = (
                ml_score_set(cand, ml_model, weights=weights, history_df=history_df)
                if ml_model
                else 0.0
            )

            base_score = (
                alpha_qh * qh
                + alpha_div * (-div / 10.0)
                + alpha_rand * rnd
                + alpha_ml * ml
            )

            shift = 2.0 * q_balance - 1.0
            if tag == "quantum":
                pref = 1.0 + pref_strength * shift
            else:
                pref = 1.0 - pref_strength * shift

            score = base_score * pref

            if (best_score is None) or (score > best_score):
                best_score = score
                best_cand = cand

        if best_cand is None:
            best_cand = generate_random_sets(1, True, weights, exclude_set)[0]

        sset = sorted(best_cand)
        t = tuple(sset)
        if t in seen:
            continue
        seen.add(t)
        result.append(sset)
        base_list.append(set(sset))

    return result

