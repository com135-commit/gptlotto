#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로또 시뮬레이션 로직 모듈
- simulate_chunk: 청크 단위 시뮬레이션
- run_simulation: 전체 시뮬레이션 실행
- build_synthetic_player_pool: 인공 참가자 풀 생성
- estimate_expected_winners: 예상 당첨자 수 계산
- rigged candidate 함수들 (CPU/GPU)
"""

from __future__ import annotations

# Numba CUDA 에러 메시지 숨기기
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from lotto_utils import get_rng
from lotto_generators import generate_random_sets

_rng = get_rng()

# Numba JIT 지원
try:
    from numba import jit
    HAS_NUMBA = True
    print("Numba JIT detected - simulation acceleration available")
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# GPU 지원 확인
try:
    import cupy as cp
except Exception:
    cp = None


@jit(nopython=True, cache=True)
def _count_matches_jit(draw, user_set):
    """Numba JIT: 한 회차의 추첨 번호와 사용자 세트 비교"""
    count = 0
    for num in draw:
        for user_num in user_set:
            if num == user_num:
                count += 1
                break
    return count


@jit(nopython=True, cache=True)
def _simulate_batch_jit(draws_batch, bonus_batch, sets_2d, include_bonus):
    """
    Numba JIT: 배치 단위 시뮬레이션 (핵심 루프)

    Parameters:
    - draws_batch: (B, 6) 추첨 번호들
    - bonus_batch: (B,) 보너스 번호들
    - sets_2d: (S, 6) 사용자 세트들 (정렬된 상태)
    - include_bonus: 보너스 번호 포함 여부

    Returns:
    - match_bins: (S, 7) 매칭 카운트
    - bonus5_counts: (S,) 5+bonus 카운트
    """
    B = draws_batch.shape[0]
    S = sets_2d.shape[0]

    match_bins = np.zeros((S, 7), dtype=np.int64)
    bonus5_counts = np.zeros(S, dtype=np.int64)

    for idx in range(S):
        user_set = sets_2d[idx]

        for b in range(B):
            draw = draws_batch[b]

            # 매칭 개수 계산
            match_count = _count_matches_jit(draw, user_set)
            match_bins[idx, match_count] += 1

            # 5+bonus 체크
            if include_bonus and match_count == 5:
                bonus = bonus_batch[b]
                # 보너스가 사용자 세트에 있고, 추첨 6개에는 없는지 확인
                bonus_in_set = False
                for user_num in user_set:
                    if bonus == user_num:
                        bonus_in_set = True
                        break

                if bonus_in_set:
                    # 보너스가 추첨 6개에 없는지 확인
                    bonus_in_draw = False
                    for draw_num in draw:
                        if bonus == draw_num:
                            bonus_in_draw = True
                            break

                    if not bonus_in_draw:
                        bonus5_counts[idx] += 1

    return match_bins, bonus5_counts


def simulate_chunk(
    draws: int,
    batch: int,
    seed: int,
    sets_array: list[np.ndarray],
    include_bonus: bool = True,
    progress_callback=None,  # 진행률 콜백 함수 추가
):
    rng = np.random.default_rng(seed)
    S = len(sets_array)

    match_bins = np.zeros((S, 7), dtype=np.int64)
    bonus5_counts = np.zeros(S, dtype=np.int64)
    agg_bins = np.zeros(7, dtype=np.int64)
    agg_bonus5 = 0

    nums_all = np.arange(1, 46, dtype=np.int16)

    remaining = draws
    completed = 0

    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b
        completed += b

        draws_batch = np.empty((b, 6), dtype=np.int16)
        bonus_batch = np.empty(b, dtype=np.int16) if include_bonus else None

        for i in range(b):
            barrel = rng.permutation(nums_all)
            balls = barrel[:7]
            main6 = np.sort(balls[:6])
            draws_batch[i] = main6
            if include_bonus:
                bonus_batch[i] = balls[6]

        # Numba JIT 최적화 버전 사용
        if HAS_NUMBA:
            # 사용자 세트를 2D 배열로 변환 (모두 6개씩, 정렬된 상태)
            sets_2d = np.array(sets_array, dtype=np.int16)

            # JIT 함수 호출
            batch_match_bins, batch_bonus5_counts = _simulate_batch_jit(
                draws_batch, bonus_batch if include_bonus else np.zeros(b, dtype=np.int16),
                sets_2d, include_bonus
            )

            # 결과 누적
            match_bins += batch_match_bins
            bonus5_counts += batch_bonus5_counts
            for idx in range(S):
                agg_bins += batch_match_bins[idx]
                agg_bonus5 += batch_bonus5_counts[idx]
        else:
            # 기존 Python 버전 (Numba 없을 때)
            for idx, s in enumerate(sets_array):
                matches_vec = np.isin(draws_batch, s).sum(axis=1)
                counts = np.bincount(matches_vec, minlength=7)
                match_bins[idx, :7] += counts
                agg_bins[:7] += counts

                if include_bonus:
                    bonus_in_set = np.isin(bonus_batch, s)
                    not_in_row = np.ones(b, dtype=bool)
                    for j in range(b):
                        row = draws_batch[j]
                        c = bonus_batch[j]
                        pos = np.searchsorted(row, c)
                        not_in_row[j] = not (pos < 6 and row[pos] == c)
                    cond = (matches_vec == 5) & bonus_in_set & not_in_row
                    bonus5_counts[idx] += int(cond.sum())
                    agg_bonus5 += int(cond.sum())

        # 진행률 콜백 호출
        if progress_callback:
            progress = (completed / draws) * 100
            progress_callback(completed, draws, progress)

    return match_bins, bonus5_counts, agg_bins, agg_bonus5


def run_simulation(
    draws: int,
    batch: int,
    workers: int,
    seed: int | None,
    sets_: list[list[int]],
):
    W = max(1, min(36, workers))
    sets_array = [np.array(s, dtype=np.int16) for s in sets_]
    per_worker = draws // W
    remainder = draws % W
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        futures = []
        for i in range(W):
            draws_i = per_worker + (1 if i < remainder else 0)
            if draws_i == 0:
                continue
            seed_i = None if seed is None else (seed + i * 9973)
            futures.append(
                ex.submit(
                    simulate_chunk, draws_i, batch, seed_i, sets_array, True
                )
            )
        for fut in as_completed(futures):
            parts.append(fut.result())

    match_bins = None
    bonus5_counts = None
    agg_bins = None
    agg_bonus5 = 0
    for mb, b5, ab, a5 in parts:
        match_bins = mb if match_bins is None else (match_bins + mb)
        bonus5_counts = b5 if bonus5_counts is None else (bonus5_counts + b5)
        agg_bins = ab if agg_bins is None else (agg_bins + ab)
        agg_bonus5 += a5

    total = draws
    rows = []
    for idx, s in enumerate(sets_):
        row = {"Set": f"Set_{idx + 1:02d}", "Numbers": " ".join(map(str, s))}
        for m in range(7):
            row[f"match_{m}_count"] = int(match_bins[idx, m])
            row[f"match_{m}_prob"] = match_bins[idx, m] / total
        row["match_5plusbonus_count"] = int(bonus5_counts[idx])
        row["match_5plusbonus_prob"] = bonus5_counts[idx] / total
        row["≥3_match_prob"] = match_bins[idx, 3:7].sum() / total
        rows.append(row)
    per_set_df = pd.DataFrame(rows)

    agg_row = {"Set": "AGG_ALL", "Numbers": "—"}
    for m in range(7):
        agg_row[f"match_{m}_count"] = int(agg_bins[m])
        agg_row[f"match_{m}_prob"] = agg_bins[m] / (total * len(sets_))
    agg_row["match_5plusbonus_count"] = int(agg_bonus5)
    agg_row["match_5plusbonus_prob"] = agg_bonus5 / (total * len(sets_))
    agg_row["≥3_match_prob"] = agg_bins[3:7].sum() / (total * len(sets_))
    agg_df = pd.DataFrame([agg_row])
    return per_set_df, agg_df


# ------------------ 리깅 시뮬용: 가상 플레이어 풀 생성 ------------------
def _build_synthetic_player_pool_chunk(
    n_players: int,
    weights,
    recent_exclude: set[int] | None = None,
) -> dict[tuple[int, ...], int]:
    """
    멀티프로세싱용: 일부 플레이어 수 만큼 티켓을 생성해서 부분 딕셔너리 반환.

    HM(히스토리 기반) 가중치 + 사람들이 실제로 자주 쓰는 패턴들을 섞어서
    '가상의 플레이어' 티켓 분포를 만든다.

    포함 패턴 (플레이어 타입, 총 20종):

      1) 순수 HM 가중 랜덤
      2) 생일형 (1~31 위주)
      3) 저/고 패턴 (저4·고2)
      4) 저/고 패턴 (저2·고4)
      5) 홀짝 패턴 (홀4·짝2)
      6) 홀짝 패턴 (짝4·홀2)
      7) 6연속 패턴 (가로줄 전체)
      8) 부분 연속 패턴 (3~4연속 + 나머지 랜덤)
      9) 끝수(같은 일의 자리) 패턴 (세로줄)
     10) 같은 10단위(디케이드) 클러스터 패턴 (3~4개 같은 10단위)
     11) Hot 번호 위주 (HM 상위 15개 중심)
     12) Cold 번호 위주 (HM 하위 15개 중심)
     13) 합계 범위 패턴 A (합계 100~160)
     14) 합계 + '쌍' 패턴 (합계 120~150, 인접/유사 쌍 포함)

     --- 추가 패턴 (해외/편향 패턴 + 회피 패턴) ---

     15) Hot 번호 회피 패턴 (상위 15개를 피하고 나머지에서 선택)
     16) 초핫(스트릭 유사) 패턴 (Top 5에서 최소 2개 이상 포함)
     17) 초저번호/폼 편향 패턴 (1~15에서 3~4개 + 30이상 번호 1개 포함)
     18) 극단 몰빵 디케이드 패턴 (한 10단위에서 5개 이상 뽑기)
     19) 극단 선택 패턴 (6개 모두 홀수 또는 6개 모두 저수(1~22))
     20) 최근 N회 출현 번호 회피 패턴 (recent_exclude 세트에서 최대한 피해서 선택)
    """

    rng = np.random.default_rng()
    recent_exclude = set(recent_exclude or set())

    # --- 기본 번호 가중치 정규화 ---
    if weights is None:
        w = np.ones(45, dtype=float) / 45.0
    else:
        w = np.array(weights, dtype=float)
        if w.size != 45 or w.sum() <= 0:
            w = np.ones(45, dtype=float) / 45.0
        else:
            w = w / w.sum()

    xs = np.arange(1, 46, dtype=int)     # 1~45
    p_all = w.copy()                     # 전체 번호용 기본 분포

    # --- 기본 구간/패턴용 집합 정의 ---
    odds = np.arange(1, 46, 2, dtype=int)      # 홀수
    evens = np.arange(2, 46, 2, dtype=int)     # 짝수
    low_nums = np.arange(1, 23, dtype=int)     # 저 : 1~22
    high_nums = np.arange(23, 46, dtype=int)   # 고 : 23~45

    # Hot / Cold 구간 (HM 가중치 기준 상위/하위 15개)
    sorted_idx = np.argsort(-w)          # 내림차순 인덱스
    hot_idx = sorted_idx[:15]
    cold_idx = sorted_idx[-15:]
    hot_nums = (hot_idx + 1).astype(int)     # 1~45 번호
    cold_nums = (cold_idx + 1).astype(int)

    # '초핫' (스트릭 비슷한 느낌): 상위 5개 정도
    super_hot_idx = sorted_idx[:5]
    super_hot_nums = (super_hot_idx + 1).astype(int)

    # Hot 회피용 (hot을 제외한 나머지)
    non_hot_nums = np.array(
        [n for n in xs if n not in set(hot_nums)],
        dtype=int
    )
    if non_hot_nums.size == 0:
        non_hot_nums = xs.copy()

    # 최근 출현 번호 회피용 후보 (전체에서 recent_exclude 뺀 것)
    recent_avoid_nums = np.array(
        [n for n in xs if n not in recent_exclude],
        dtype=int
    )
    if recent_avoid_nums.size < 6:
        # 너무 적으면 그냥 전체 사용 (회피 불가 상황)
        recent_avoid_nums = xs.copy()

    # 디케이드(10단위) 구간
    decade_ranges = [
        np.arange(1, 11, dtype=int),     # 1~10
        np.arange(11, 21, dtype=int),    # 11~20
        np.arange(21, 31, dtype=int),    # 21~30
        np.arange(31, 41, dtype=int),    # 31~40
        np.arange(41, 46, dtype=int),    # 41~45
    ]

    # --- 공통 유틸 함수들 ---

    def sample_from_candidates(cands, k: int) -> list[int]:
        """
        주어진 후보(cands)에서 HM 가중치(w)를 이용해
        중복 없이 k개 샘플링.
        """
        cands = np.array(sorted({int(v) for v in cands if 1 <= int(v) <= 45}), dtype=int)
        if cands.size == 0 or k <= 0:
            return []

        p = w[cands - 1].copy()
        if p.sum() <= 0:
            p = np.ones_like(p, dtype=float) / len(cands)
        else:
            p = p / p.sum()

        chosen: list[int] = []
        c_list = cands.tolist()
        p_list = p.tolist()

        for _ in range(min(k, len(c_list))):
            idx = int(rng.choice(len(c_list), p=p_list))
            chosen.append(c_list[idx])
            c_list.pop(idx)
            p_list.pop(idx)
            if p_list:
                s = sum(p_list)
                if s <= 0:
                    p_list = [1.0 / len(p_list)] * len(p_list)
                else:
                    p_list = [v / s for v in p_list]

        return chosen

    def finalize_ticket(base: list[int]) -> tuple[int, ...]:
        """
        임시로 만든 번호 리스트 base를:
          - 1~45 범위 체크
          - 중복 제거
          - 부족하면 HM 가중치 기반 랜덤으로 채워서
        최종 6개 번호 튜플로 만들어준다.
        (가능하면 recent_exclude 번호는 피함)
        """
        s = {int(v) for v in base if 1 <= int(v) <= 45}

        # 먼저 '최근 회피 번호가 아닌 번호'들로 채우기 시도
        while len(s) < 6:
            # recent_exclude를 피해서 뽑을 수 있는 후보
            cand_nums = [n for n in xs if (n not in s) and (n not in recent_exclude)]
            if not cand_nums:
                # 더 이상 피할 수 없으면 그냥 전체에서 채움
                cand_nums = [n for n in xs if n not in s]
            p = p_all[np.array(cand_nums) - 1]
            if p.sum() <= 0:
                p = np.ones(len(cand_nums), dtype=float) / len(cand_nums)
            else:
                p = p / p.sum()
            v = int(rng.choice(cand_nums, p=p))
            s.add(v)

        return tuple(sorted(s))

    def sum_constrained_ticket(sum_min: int, sum_max: int, max_try: int = 40) -> tuple[int, ...]:
        """
        합계가 [sum_min, sum_max] 범위에 들어오는 티켓을 만들려고 시도.
        실패하면 그냥 HM 랜덤 티켓으로 폴백.
        (finalize_ticket에서 recent_exclude 회피를 최대한 시도)
        """
        for _ in range(max_try):
            cand = sample_from_candidates(xs, 6)
            if not cand:
                continue
            s_ = sum(cand)
            if sum_min <= s_ <= sum_max:
                return finalize_ticket(cand)
        # 실패 시 순수 랜덤
        return finalize_ticket(sample_from_candidates(xs, 6))

    def pair_style_ticket(sum_min: int, sum_max: int, max_try: int = 40) -> tuple[int, ...]:
        """
        '번호 쌍' 스타일:
          - 인접 또는 비슷한 번호 2개를 먼저 선택 (예: 7,8 또는 21,23 등)
          - 나머지는 HM 랜덤으로 채움
          - 합계도 [sum_min, sum_max] 안쪽이면 사용
        """
        for _ in range(max_try):
            base: list[int] = []

            # 쌍의 첫 번째 번호 (HM 가중치 기반)
            a = int(rng.choice(xs, p=p_all))
            # 두 번째는 인접/근처 번호 중 하나
            cand_pair = [v for v in [a - 2, a - 1, a + 1, a + 2] if 1 <= v <= 45]
            if not cand_pair:
                continue
            b = int(rng.choice(cand_pair))
            base.extend([a, b])

            # 나머지를 HM 랜덤으로 채움
            rest_cnt = 6 - len(set(base))
            if rest_cnt > 0:
                base += sample_from_candidates(xs, rest_cnt)

            ticket = finalize_ticket(base)
            s_ = sum(ticket)
            if sum_min <= s_ <= sum_max:
                return ticket

        # 실패 시 일반 합계 제한에 폴백
        return sum_constrained_ticket(sum_min, sum_max, max_try)

    pool: dict[tuple[int, ...], int] = {}

    for _ in range(n_players):
        r = rng.random()

        # 확률 분할 (합 = 1.0)
        # 1)  0.000 ~ 0.210 : 순수 HM 가중 랜덤 (21%)
        # 2)  0.210 ~ 0.360 : 생일형 (1~31) (15%)
        # 3)  0.360 ~ 0.460 : 저4 고2 (10%)
        # 4)  0.460 ~ 0.560 : 저2 고4 (10%)
        # 5)  0.560 ~ 0.640 : 홀4 짝2 (8%)
        # 6)  0.640 ~ 0.720 : 짝4 홀2 (8%)
        # 7)  0.720 ~ 0.750 : 6연속 (3%)
        # 8)  0.750 ~ 0.790 : 부분 연속 (4%)
        # 9)  0.790 ~ 0.830 : 같은 끝수 (4%)
        # 10) 0.830 ~ 0.870 : 10단위 클러스터 (3~4개) (4%)
        # 11) 0.870 ~ 0.895 : Hot 위주 (2.5%)
        # 12) 0.895 ~ 0.920 : Cold 위주 (2.5%)
        # 13) 0.920 ~ 0.940 : 합계 100~160 (2%)
        # 14) 0.940 ~ 0.955 : 합계+쌍 (120~150) (1.5%)
        # 15) 0.955 ~ 0.970 : Hot 회피 패턴 (1.5%)
        # 16) 0.970 ~ 0.977 : 초핫 패턴 (0.7%)
        # 17) 0.977 ~ 0.983 : 초저번호/폼 편향 (0.6%)
        # 18) 0.983 ~ 0.989 : 극단 디케이드 몰빵 (0.6%)
        # 19) 0.989 ~ 0.995 : 극단 선택 (올홀/올저) (0.6%)
        # 20) 0.995 ~ 1.000 : 최근 N회 번호 회피 패턴 (0.5%)

        if r < 0.21:
            # 1) 순수 HM 가중 랜덤
            ticket = sample_from_candidates(xs, 6)
            key = finalize_ticket(ticket)

        elif r < 0.36:
            # 2) 생일형 (1~31)
            birthday_range = np.arange(1, 32, dtype=int)
            ticket = sample_from_candidates(birthday_range, 6)
            key = finalize_ticket(ticket)

        elif r < 0.46:
            # 3) 저고 패턴(저4 고2)
            base_low = sample_from_candidates(low_nums, 4)
            base_high = sample_from_candidates(high_nums, 2)
            ticket = base_low + base_high
            key = finalize_ticket(ticket)

        elif r < 0.56:
            # 4) 저고 패턴(저2 고4)
            base_low = sample_from_candidates(low_nums, 2)
            base_high = sample_from_candidates(high_nums, 4)
            ticket = base_low + base_high
            key = finalize_ticket(ticket)

        elif r < 0.64:
            # 5) 홀짝 패턴 (홀4 짝2)
            ticket = sample_from_candidates(odds, 4) + sample_from_candidates(evens, 2)
            key = finalize_ticket(ticket)

        elif r < 0.72:
            # 6) 홀짝 패턴 (짝4 홀2)
            ticket = sample_from_candidates(evens, 4) + sample_from_candidates(odds, 2)
            key = finalize_ticket(ticket)

        elif r < 0.75:
            # 7) 6연속 패턴 (가로줄 전체)
            start = int(rng.integers(1, 40))  # 40까지면 start+5 = 45
            ticket = list(range(start, start + 6))
            key = finalize_ticket(ticket)

        elif r < 0.79:
            # 8) 부분 연속 패턴 (3~4개 연속 + 나머지 랜덤)
            seg_len = int(rng.integers(3, 5))  # 3 또는 4개 연속
            max_start = 46 - seg_len
            start = int(rng.integers(1, max_start))
            seq = list(range(start, start + seg_len))
            rest_cnt = 6 - len(set(seq))
            rest = sample_from_candidates(xs, rest_cnt)
            ticket = seq + rest
            key = finalize_ticket(ticket)

        elif r < 0.83:
            # 9) 끝수(세로줄) 패턴 (같은 일의 자리 숫자 위주)
            d = int(rng.integers(0, 10))  # 0~9
            cand = [n for n in range(1, 46) if n % 10 == d]
            base = sample_from_candidates(cand, min(4, len(cand)))
            ticket = base  # 나머지는 finalize에서 채움
            key = finalize_ticket(ticket)

        elif r < 0.87:
            # 10) 같은 10단위(디케이드) 클러스터 패턴 (3~4개 같은 10단위)
            dec = decade_ranges[int(rng.integers(0, len(decade_ranges)))]
            main_cnt = int(rng.integers(3, 5))  # 3 또는 4개
            main = sample_from_candidates(dec, main_cnt)
            rest = sample_from_candidates(xs, 6 - len(set(main)))
            ticket = main + rest
            key = finalize_ticket(ticket)

        elif r < 0.895:
            # 11) Hot 번호 위주 패턴 (상위 15개 번호에서 4개 + 그 외 2개)
            base_hot = sample_from_candidates(hot_nums, 4)
            others = [n for n in xs if n not in base_hot]
            base_rest = sample_from_candidates(others, 2)
            ticket = base_hot + base_rest
            key = finalize_ticket(ticket)

        elif r < 0.92:
            # 12) Cold 번호 위주 패턴 (하위 15개 번호에서 3~4개)
            cold_cnt = int(rng.integers(3, 5))   # 3 또는 4개
            base_cold = sample_from_candidates(cold_nums, cold_cnt)
            rest = sample_from_candidates(xs, 6 - len(set(base_cold)))
            ticket = base_cold + rest
            key = finalize_ticket(ticket)

        elif r < 0.94:
            # 13) 합계 범위 패턴 A (합계 100~160)
            key = sum_constrained_ticket(100, 160)

        elif r < 0.955:
            # 14) 합계 + '쌍' 패턴 (합계 120~150 + 인접/유사쌍)
            key = pair_style_ticket(120, 150)

        elif r < 0.97:
            # 15) Hot 회피 패턴 (상위 15개 번호를 피하고 나머지에서 선택)
            base = sample_from_candidates(non_hot_nums, 6)
            key = finalize_ticket(base)

        elif r < 0.977:
            # 16) 초핫(스트릭 느낌) 패턴:
            #     Top 5(또는 5개 내 초핫 번호)에서 최소 2개 + 나머지 랜덤
            base_hot2 = sample_from_candidates(super_hot_nums, 2)
            rest = sample_from_candidates(xs, 6 - len(set(base_hot2)))
            ticket = base_hot2 + rest
            key = finalize_ticket(ticket)

        elif r < 0.983:
            # 17) 초저번호/폼 편향 패턴:
            #     1~15에서 3~4개 + 30 이상 번호 1개 + 나머지 랜덤
            low15 = np.arange(1, 16, dtype=int)   # 1~15
            main_cnt = int(rng.integers(3, 5))   # 3 또는 4개
            base_low15 = sample_from_candidates(low15, main_cnt)
            high30 = np.arange(30, 46, dtype=int)
            base_high30 = sample_from_candidates(high30, 1)
            rest = sample_from_candidates(xs, 6 - len(set(base_low15 + base_high30)))
            ticket = base_low15 + base_high30 + rest
            key = finalize_ticket(ticket)

        elif r < 0.989:
            # 18) 극단 디케이드 몰빵 패턴:
            #     한 10단위 구간에서 5개 이상 뽑는 극단 패턴
            dec = decade_ranges[int(rng.integers(0, len(decade_ranges)))]
            main = sample_from_candidates(dec, min(5, len(dec)))
            rest = sample_from_candidates(xs, 6 - len(set(main)))
            ticket = main + rest
            key = finalize_ticket(ticket)

        elif r < 0.995:
            # 19) 극단 선택 패턴:
            #     50%: 6개 모두 홀수 / 50%: 6개 모두 저수(1~22)
            if rng.random() < 0.5:
                # 올 홀수
                base = sample_from_candidates(odds, 6)
                key = finalize_ticket(base)
            else:
                # 올 저수(1~22)
                base = sample_from_candidates(low_nums, 6)
                key = finalize_ticket(base)

        else:
            # 20) 최근 N회 출현 번호 회피 패턴
            #     recent_exclude에 속한 번호를 최대한 피해서 뽑기
            base = sample_from_candidates(recent_avoid_nums, 6)
            key = finalize_ticket(base)

        pool[key] = pool.get(key, 0) + 1

    return pool





def build_synthetic_player_pool(
    n_players: int,
    weights,
    workers: int | None = None,
    recent_exclude: set[int] | None = None,
) -> dict[tuple[int, ...], int]:
    """
    HM/현실 분포 기반으로 '가상의 플레이어들'이 어떤 조합을 샀는지 풀 구성.
    - n_players: 가상 플레이어 수 (예: 200,000)
    - weights : 번호 인기 분포(길이 45, 합=1)
    - workers : 프로세스 개수 (None이면 단일 프로세스, 보통 36으로 사용)
    - recent_exclude: 최근 N회 출현 번호 등, 회피하고 싶은 번호 세트
    - return  : { (n1,..,n6): 그 조합을 산 사람 수 }
    """
    if weights is None:
        w = np.ones(45, dtype=float) / 45.0
    else:
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones(45, dtype=float) / 45.0
        else:
            w = w / w.sum()

    recent_exclude = set(recent_exclude or set())

    if workers is None or workers <= 1:
        # 단일 프로세스 버전
        tickets = generate_random_sets(
            n_players,
            avoid_duplicates=False,
            weights=w,
            exclude_set=None,
        )
        pool: dict[tuple[int, ...], int] = {}
        for ticket in tickets:
            # finalize_ticket 역할은 chunk에서만 쓰므로 여기서는 단순 집계
            key = tuple(sorted(ticket))
            pool[key] = pool.get(key, 0) + 1
        return pool

    # 멀티프로세스 버전
    W = max(1, workers)
    per_worker = n_players // W
    remainder = n_players % W

    pool: dict[tuple[int, ...], int] = {}
    with ProcessPoolExecutor(max_workers=W) as ex:
        futures = []
        for i in range(W):
            n_i = per_worker + (1 if i < remainder else 0)
            if n_i <= 0:
                continue
            futures.append(
                ex.submit(_build_synthetic_player_pool_chunk, n_i, w, recent_exclude)
            )
        for fut in as_completed(futures):
            part = fut.result()
            for key, cnt in part.items():
                pool[key] = pool.get(key, 0) + cnt
    return pool



def estimate_expected_winners_from_pool(
    draw_nums: list[int],
    ticket_pool: dict[tuple[int, ...], int],
    scale_factor: float,
) -> float:
    """
    주어진 당첨 번호(draw_nums)에 대해:
    - ticket_pool에서 '6개 모두 일치'하는 사람 수를 찾고
    - 실제 전국 판매량 수준으로 스케일해서 예상 1등 인원 λ를 반환.
    """
    key = tuple(sorted(draw_nums))
    winners_sim = ticket_pool.get(key, 0)
    lam = float(winners_sim) * float(scale_factor)
    return lam


# 후보 당첨 번호 샘플링 멀티프로세스용 (CPU)
def _rigged_candidate_chunk(
    n_samples: int,
    weights,
    ticket_pool: dict[tuple[int, ...], int],
    scale_factor: float,
    tmin: int,
    tmax: int,
    ml_model=None,
    ml_weight: float = 0.0,
    history_df=None,
) -> list[tuple[list[int], float]]:
    """
    n_samples 만큼 후보 당첨 번호를 생성하고,
    각 번호에 대한 예상 1등 인원 λ를 계산해
    [tmin, tmax] 범위에 들어가는 것만 반환.

    ML 적용:
    - ml_model이 있으면 ML 점수도 함께 계산
    - ml_weight로 ML 점수의 중요도 조절
    - ML 점수가 높은 번호를 우선 선택
    """
    if weights is None:
        w = None
    else:
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones(45, dtype=float) / 45.0
        else:
            w = w / w.sum()

    results: list[tuple[list[int], float]] = []
    if n_samples <= 0:
        return results

    # ML 사용 여부 결정
    use_ml = ml_model is not None and ml_weight > 0 and history_df is not None

    # 더 많이 생성 (ML 필터링 고려)
    if use_ml:
        # ML 필터링으로 걸러질 것을 대비해 3배 생성
        gen_samples = n_samples * 3
    else:
        gen_samples = n_samples

    draws_list = generate_random_sets(
        gen_samples,
        avoid_duplicates=False,
        weights=w,
        exclude_set=None,
    )

    # ML 점수 계산 (필요시)
    if use_ml:
        from lotto_generators import ml_score_set
        scored_draws = []
        for draw in draws_list:
            lam = estimate_expected_winners_from_pool(draw, ticket_pool, scale_factor)
            if tmin <= lam <= tmax:
                # ML 점수 계산
                try:
                    ml_score = ml_score_set(draw, ml_model, weights, history_df)
                except Exception:
                    ml_score = 0.5  # 기본값

                # 복합 점수: ML 점수 반영
                # lam이 목표 중앙값에 가까울수록 좋고, ML 점수도 높을수록 좋음
                center = 0.5 * (tmin + tmax)
                lam_score = 1.0 - abs(lam - center) / max(1, tmax - tmin)

                combined_score = (1 - ml_weight) * lam_score + ml_weight * ml_score
                scored_draws.append((draw, lam, combined_score))

        # 복합 점수 기준 정렬 (높은 점수 우선)
        scored_draws.sort(key=lambda x: x[2], reverse=True)

        # 상위 n_samples개 선택
        for draw, lam, _ in scored_draws[:n_samples]:
            results.append((draw, lam))
    else:
        # 기존 방식 (ML 없이)
        for draw in draws_list:
            lam = estimate_expected_winners_from_pool(draw, ticket_pool, scale_factor)
            if tmin <= lam <= tmax:
                results.append((draw, lam))

    return results


# ★ GPU 버전 후보 생성 (CuPy 벡터화 버전)
def _rigged_candidate_gpu(
    n_samples: int,
    weights,
    ticket_pool: dict[tuple[int, ...], int],
    scale_factor: float,
    tmin: int,
    tmax: int,
) -> list[tuple[list[int], float]]:
    """
    GPU 벡터화 버전 (Gumbel-top-k 사용)
      - CuPy를 사용해서 후보 당첨 번호를 한 번에 여러 세트씩 생성
      - 각 세트는 1~45 중에서 '가중치 기반 + 중복 없는 6개'
      - ticket_pool 조회 및 λ 계산은 CPU에서 dict로 수행 (기존 로직 그대로 유지)
    """
    if cp is None:
        raise RuntimeError("CuPy(cupy)가 설치되어 있지 않습니다.")

    if n_samples <= 0:
        return []

    # 1) 가중치 정규화
    if weights is None:
        w = np.ones(45, dtype=float) / 45.0
    else:
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones(45, dtype=float) / 45.0
        else:
            w = w / w.sum()

    # GPU로 올리기
    w_gpu = cp.asarray(w, dtype=cp.float32)          # (45,)
    log_w_gpu = cp.log(w_gpu)                        # log p_j

    results: list[tuple[list[int], float]] = []

    # 너무 큰 n_samples를 한 번에 처리하면 메모리 부담이 크므로 배치로 나눠서 처리
    batch_size = 20_000
    remaining = n_samples

    while remaining > 0:
        b = min(batch_size, remaining)
        remaining -= b

        # 2) Gumbel 분포 샘플링: g ~ Gumbel(0, 1)
        U = cp.random.random((b, 45), dtype=cp.float32)  # (b,45)
        g = -cp.log(-cp.log(U))
        scores = log_w_gpu[cp.newaxis, :] + g            # (b,45)

        # 3) 각 행마다 상위 6개 인덱스
        k = 6
        idx_part = cp.argpartition(scores, -k, axis=1)[:, -k:]  # (b, k)
        draws_gpu = cp.sort(idx_part + 1, axis=1)               # (b, k)

        # 4) CPU로 가져와 λ 계산
        draws_cpu = draws_gpu.get()  # (b,6)

        for row in draws_cpu:
            draw = row.tolist()
            lam = estimate_expected_winners_from_pool(draw, ticket_pool, scale_factor)
            if tmin <= lam <= tmax:
                results.append((draw, lam))

    return results


# ------------------ GUI ------------------
