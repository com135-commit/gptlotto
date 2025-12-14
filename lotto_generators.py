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
from numba import njit, prange

# Export 추가
__all__ = [
    'generate_random_sets',
    'generate_biased_combinations',
    '_set_features',
    '_compute_core_features',
    '_compute_core_features_batch',
    '_compute_history_features',
    '_compute_history_features_batch',
    '_compute_temporal_features_batch',
    '_prepare_history_array',
    'train_ml_scorer',
]

# 고도화 고전 알고리즘 import
from lotto_advanced_classical import (
    gen_hot_cold_balanced,
    gen_consecutive_controlled,
    gen_sum_range_controlled,
    gen_zone_balanced,
    gen_last_digit_diverse,
    gen_prime_mixed,
    gen_temporal_strategy,
    gen_frequency_inverted,
    gen_cycle_pattern,
    gen_ml_guided,
)

# 전역 랜덤 생성기 사용
_rng = get_rng()


# ============= Numba 최적화 함수 =============

@njit(cache=True, fastmath=True)
def _is_prime_numba(n):
    """소수 판별 (numba 최적화)"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@njit(cache=True, fastmath=True)
def _compute_core_features(nums_arr):
    """
    핵심 특징 계산 (numba JIT 컴파일)

    pandas 의존성이 없는 순수 계산만 수행 → 5-10배 빠름
    """
    n_nums = len(nums_arr)

    # 기본 통계
    f_mean = nums_arr.mean() / 45.0
    f_std = nums_arr.std() / 20.0
    f_min = nums_arr.min() / 45.0
    f_max = nums_arr.max() / 45.0
    f_median = np.median(nums_arr) / 45.0
    f_range = (nums_arr.max() - nums_arr.min()) / 45.0

    # 분위수
    q1 = np.percentile(nums_arr, 25)
    q3 = np.percentile(nums_arr, 75)
    f_iqr = (q3 - q1) / 45.0

    # 구간 분포
    evens = 0.0
    low = 0.0
    mid = 0.0
    high = 0.0
    for v in nums_arr:
        if int(v) % 2 == 0:
            evens += 1.0
        if 1 <= v <= 15:
            low += 1.0
        elif 16 <= v <= 30:
            mid += 1.0
        elif 31 <= v <= 45:
            high += 1.0
    evens /= n_nums
    low /= n_nums
    mid /= n_nums
    high /= n_nums

    # 간격 분석
    gaps = np.diff(nums_arr)
    if len(gaps) > 0:
        f_gmean = (gaps.mean() - 8.0) / 8.0
        f_gstd = gaps.std() / 10.0
        f_gap_min = gaps.min() / 10.0
        f_gap_max = gaps.max() / 10.0
        f_gap_median = np.median(gaps) / 10.0
        f_gap_cv = (gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    else:
        f_gmean = f_gstd = f_gap_min = f_gap_max = f_gap_median = f_gap_cv = 0.0

    # 연속 번호
    consecutive_count = 0
    max_consecutive = 1
    current_consecutive = 1
    for i in range(n_nums - 1):
        if nums_arr[i+1] - nums_arr[i] == 1:
            consecutive_count += 1
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
        else:
            current_consecutive = 1
    f_consecutive = consecutive_count / 5.0
    f_max_consecutive = max_consecutive / 6.0

    # 끝자리 다양성
    last_digits = np.zeros(n_nums, dtype=np.int32)
    for i in range(n_nums):
        last_digits[i] = int(nums_arr[i]) % 10
    unique_last_digits = len(np.unique(last_digits))
    f_last_digit_diversity = unique_last_digits / 6.0
    f_last_digit_dup = (6 - unique_last_digits) / 6.0

    # 배수
    f_mult3 = 0.0
    f_mult5 = 0.0
    for n in nums_arr:
        if int(n) % 3 == 0:
            f_mult3 += 1.0
        if int(n) % 5 == 0:
            f_mult5 += 1.0
    f_mult3 /= n_nums
    f_mult5 /= n_nums

    # 소수
    f_primes = 0.0
    for n in nums_arr:
        if _is_prime_numba(int(n)):
            f_primes += 1.0
    f_primes /= n_nums

    # 대칭성
    center = 23.0
    symmetry_errors = 0.0
    for n in nums_arr:
        symmetry_errors += abs((n - center) + (center - (46 - n)))
    f_symmetry = max(0.0, 1.0 - (symmetry_errors / (n_nums * 45)))

    # 엔트로피
    unique_tens = len(np.unique(nums_arr // 10))
    unique_ones = len(np.unique(nums_arr % 10))
    f_entropy = (unique_tens / 5.0 + unique_ones / 10.0) / 2.0

    # 카이제곱 균등성
    observed = np.array([low * n_nums, mid * n_nums, high * n_nums])
    expected = np.array([2.0, 2.0, 2.0])
    chi2 = np.sum((observed - expected) ** 2 / (expected + 0.01))
    f_chi2_uniformity = 1.0 / (1.0 + chi2 / 5.0)

    # 런 테스트
    binary = np.zeros(n_nums, dtype=np.int32)
    for i in range(n_nums):
        binary[i] = 1 if nums_arr[i] > 23 else 0
    runs = 1
    for i in range(1, n_nums):
        if binary[i] != binary[i-1]:
            runs += 1
    expected_runs = 3.5
    f_runs = 1.0 - min(1.0, abs(runs - expected_runs) / 3.5)

    # 자기상관
    if len(gaps) > 1:
        f_autocorr = min(1.0, gaps.std() / 10.0)
    else:
        f_autocorr = 0.0

    # 비트 엔트로피
    ones = 0
    zeros = 0
    for n in nums_arr:
        bits = int(n)
        for _ in range(6):
            if bits & 1:
                ones += 1
            else:
                zeros += 1
            bits >>= 1
    if max(ones, zeros) > 0:
        f_bit_entropy = min(ones, zeros) / max(ones, zeros)
    else:
        f_bit_entropy = 0.0

    # 합 끝자리
    f_sum_last_digit = (nums_arr.sum() % 10) / 10.0

    # 간단한 추가 특징 (4개)
    # 피보나치
    fib_nums_arr = np.array([1, 2, 3, 5, 8, 13, 21, 34], dtype=np.float64)
    f_fibonacci = 0.0
    for n in nums_arr:
        for fib in fib_nums_arr:
            if abs(n - fib) < 0.1:
                f_fibonacci += 1.0
                break
    f_fibonacci /= 6.0

    # 클러스터링 (간격 3 이하 비율)
    if len(gaps) > 0:
        small_gaps = 0.0
        for g in gaps:
            if g <= 3:
                small_gaps += 1.0
        f_clustering = small_gaps / len(gaps)
    else:
        f_clustering = 0.0

    # 번호 간 평균 거리
    if len(nums_arr) > 1:
        total_dist = 0.0
        count = 0
        for i in range(len(nums_arr)):
            for j in range(i+1, len(nums_arr)):
                total_dist += nums_arr[j] - nums_arr[i]
                count += 1
        f_avg_distance = (total_dist / count) / 45.0 if count > 0 else 0.0
    else:
        f_avg_distance = 0.0

    # 거듭제곱 수
    power_arr = np.array([1, 4, 8, 9, 16, 25, 27, 36], dtype=np.float64)
    f_power = 0.0
    for n in nums_arr:
        for p in power_arr:
            if abs(n - p) < 0.1:
                f_power += 1.0
                break
    f_power /= 6.0

    # ===== 고급 특징 추가 (10개) =====

    # 등차수열 최대 길이
    max_arith_len = 1
    for i in range(len(nums_arr)):
        for j in range(i+1, len(nums_arr)):
            if j == i:
                continue
            diff = nums_arr[j] - nums_arr[i]
            current_len = 2
            last_num = nums_arr[j]
            for k in range(j+1, len(nums_arr)):
                if abs(nums_arr[k] - last_num - diff) < 0.1:
                    current_len += 1
                    last_num = nums_arr[k]
            if current_len > max_arith_len:
                max_arith_len = current_len
    f_arithmetic = max_arith_len / 6.0

    # 번호 분산도 (9개 구간 엔트로피)
    zone_counts = np.zeros(9, dtype=np.float64)
    for n in nums_arr:
        zone_idx = int((n - 1) / 5)
        if zone_idx >= 9:
            zone_idx = 8
        zone_counts[zone_idx] += 1.0
    zone_entropy = 0.0
    for count in zone_counts:
        if count > 0:
            p = count / len(nums_arr)
            zone_entropy -= p * np.log2(p + 1e-10)
    f_zone_entropy = zone_entropy / np.log2(9)

    # 십의 자리 균형
    tens_counts = np.zeros(5, dtype=np.float64)
    for n in nums_arr:
        tens_idx = int(n / 10)
        if tens_idx >= 5:
            tens_idx = 4
        tens_counts[tens_idx] += 1.0
    tens_std = tens_counts.std()
    f_tens_balance = 1.0 / (1.0 + tens_std)

    # 일의 자리 균형
    ones_counts = np.zeros(10, dtype=np.float64)
    for n in nums_arr:
        ones_idx = int(n) % 10
        ones_counts[ones_idx] += 1.0
    ones_std = ones_counts.std()
    f_ones_balance = 1.0 / (1.0 + ones_std)

    # 홀짝 교대 패턴
    alternating = 0
    for i in range(len(nums_arr) - 1):
        if (int(nums_arr[i]) % 2) != (int(nums_arr[i+1]) % 2):
            alternating += 1
    f_alternating = alternating / (len(nums_arr) - 1) if len(nums_arr) > 1 else 0.0

    # 범위 비율
    f_range_ratio = (nums_arr.max() - nums_arr.min()) / nums_arr.mean() if nums_arr.mean() > 0 else 0.0
    f_range_ratio = min(5.0, f_range_ratio) / 5.0

    # 중앙 집중도
    median_val = np.median(nums_arr)
    close_to_median = 0.0
    for n in nums_arr:
        if abs(n - median_val) <= 5:
            close_to_median += 1.0
    f_centrality = close_to_median / len(nums_arr)

    # 극단값 포함
    f_extremes = 0.0
    for n in nums_arr:
        if n <= 5 or n >= 41:
            f_extremes += 1.0
    f_extremes /= len(nums_arr)

    # 3의 배수와 5의 배수 비율
    f_mult_ratio = (f_mult3 + 0.01) / (f_mult5 + 0.01)
    f_mult_ratio = min(3.0, f_mult_ratio) / 3.0

    # 소수 밀집도 (연속 소수)
    primes_adjacent = 0
    for i in range(len(nums_arr) - 1):
        if _is_prime_numba(int(nums_arr[i])) and _is_prime_numba(int(nums_arr[i+1])):
            primes_adjacent += 1
    f_prime_density = primes_adjacent / (len(nums_arr) - 1) if len(nums_arr) > 1 else 0.0

    # 39개 numba 최적화 특징 반환
    return np.array([
        f_std, evens, low, mid, high,
        f_range, f_iqr,
        f_consecutive, f_max_consecutive,
        f_last_digit_diversity, f_last_digit_dup,
        f_mult3, f_mult5, f_primes, f_symmetry,
        f_gap_min, f_gap_max, f_gap_median, f_gap_cv,
        f_sum_last_digit,
        f_entropy, f_chi2_uniformity, f_runs, f_autocorr, f_bit_entropy,
        f_fibonacci, f_clustering, f_avg_distance, f_power,
        f_arithmetic, f_zone_entropy, f_tens_balance, f_ones_balance,
        f_alternating, f_range_ratio, f_centrality, f_extremes,
        f_mult_ratio, f_prime_density
    ], dtype=np.float64)


@njit(parallel=True, cache=True, fastmath=True)
def _compute_core_features_batch(nums_batch):
    """
    여러 조합을 병렬로 처리 (진짜 멀티코어!)

    Parameters:
        nums_batch: (N, 6) 배열 - N개의 6개 번호 조합

    Returns:
        (N, 39) 특징 배열

    ⚡ prange로 병렬화 → 36코어 최대 활용!
    """
    n_samples = len(nums_batch)
    features = np.zeros((n_samples, 39), dtype=np.float64)

    # prange로 병렬화 - 각 코어가 다른 샘플 처리!
    for i in prange(n_samples):
        features[i] = _compute_core_features(nums_batch[i])

    return features


def _prepare_history_array(history_df: pd.DataFrame | None) -> np.ndarray:
    """
    히스토리 DataFrame을 numpy 배열로 변환 (한 번만 실행)

    Returns:
        (N, 6) numpy array - 각 행은 [n1, n2, n3, n4, n5, n6]
    """
    if history_df is None or history_df.empty:
        return np.zeros((0, 6), dtype=np.float64)

    history_array = []
    for _, row in history_df.iterrows():
        nums = []
        for val in row:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    nums.append(v)
            except (ValueError, TypeError):
                continue
        if len(nums) == 6:
            history_array.append(sorted(nums))

    return np.array(history_array, dtype=np.float64)


@njit(cache=True, fastmath=True)
def _compute_history_features_from_array(nums_arr: np.ndarray, history_arr: np.ndarray) -> np.ndarray:
    """
    히스토리 기반 특징 계산 (Numba 최적화 버전)

    Args:
        nums_arr: (6,) 번호 배열
        history_arr: (N, 6) 히스토리 배열

    Returns:
        (11,) 특징 벡터
    """
    if len(history_arr) == 0:
        return np.zeros(11, dtype=np.float64)

    n_history = len(history_arr)
    gaps = np.diff(nums_arr)

    # 1. 최근 N회 출현 빈도
    recent_n = min(50, n_history)
    freq_counter = np.zeros(46, dtype=np.int32)  # 1~45번
    for i in range(recent_n):
        for j in range(6):
            num = int(history_arr[i, j])
            if 1 <= num <= 45:
                freq_counter[num] += 1

    total_freq = 0.0
    for num in nums_arr:
        n = int(num)
        if 1 <= n <= 45:
            total_freq += freq_counter[n]
    avg_freq = total_freq / 6.0
    max_freq = float(np.max(freq_counter))
    f_recent_freq = avg_freq / max_freq if max_freq > 0 else 0.0

    # 2. 미출현 기간
    no_show_sum = 0.0
    for num in nums_arr:
        n = int(num)
        last_seen = 100
        for i in range(n_history):
            found = False
            for j in range(6):
                if history_arr[i, j] == n:
                    last_seen = i
                    found = True
                    break
            if found:
                break
        no_show_sum += last_seen
    f_no_show_avg = (no_show_sum / 6.0) / 100.0

    # 3. 번호 쌍 동시 출현
    pair_counts = []
    for i in range(6):
        for j in range(i+1, 6):
            n1 = int(nums_arr[i])
            n2 = int(nums_arr[j])
            pair_count = 0
            for h in range(min(100, n_history)):
                has_n1 = False
                has_n2 = False
                for k in range(6):
                    if history_arr[h, k] == n1:
                        has_n1 = True
                    if history_arr[h, k] == n2:
                        has_n2 = True
                if has_n1 and has_n2:
                    pair_count += 1
            pair_counts.append(pair_count)

    if len(pair_counts) > 0:
        avg_pair = np.mean(np.array(pair_counts, dtype=np.float64))
        max_pair = float(np.max(np.array(pair_counts)))
        f_pair_freq = avg_pair / max_pair if max_pair > 0 else 0.0
    else:
        f_pair_freq = 0.0

    # 4. 간격 패턴 유사도
    if len(gaps) > 0:
        similarities = []
        for h in range(min(20, n_history)):
            row_gaps = np.diff(history_arr[h])
            if len(row_gaps) == len(gaps):
                dot = np.dot(gaps, row_gaps)
                norm1 = np.linalg.norm(gaps)
                norm2 = np.linalg.norm(row_gaps)
                if norm1 > 0 and norm2 > 0:
                    sim = dot / (norm1 * norm2)
                    similarities.append(max(0.0, sim))
        if len(similarities) > 0:
            f_gap_similarity = np.mean(np.array(similarities))
        else:
            f_gap_similarity = 0.0
    else:
        f_gap_similarity = 0.0

    # 5. 합계 추세
    current_sum = float(np.sum(nums_arr))
    recent_sums = []
    for h in range(min(20, n_history)):
        s = float(np.sum(history_arr[h]))
        recent_sums.append(s)
    if len(recent_sums) > 0:
        avg_sum = np.mean(np.array(recent_sums))
        f_sum_trend = (current_sum - avg_sum) / 100.0
    else:
        f_sum_trend = 0.0

    # 6. 최근 중복도
    overlap_counts = []
    for h in range(min(10, n_history)):
        overlap = 0
        for num in nums_arr:
            n = int(num)
            for k in range(6):
                if history_arr[h, k] == n:
                    overlap += 1
                    break
        overlap_counts.append(overlap)
    if len(overlap_counts) > 0:
        f_recent_overlap = np.mean(np.array(overlap_counts, dtype=np.float64)) / 6.0
    else:
        f_recent_overlap = 0.0

    # 7. 합계 편차
    if len(recent_sums) > 0:
        std_sum = np.std(np.array(recent_sums))
        f_sum_deviation = abs(current_sum - avg_sum) / (std_sum + 1.0)
    else:
        f_sum_deviation = 0.0

    # 8. 빈도 분산
    freqs = []
    for num in nums_arr:
        n = int(num)
        if 1 <= n <= 45:
            freqs.append(float(freq_counter[n]))
    if len(freqs) > 0:
        f_freq_variance = np.std(np.array(freqs)) / 10.0
    else:
        f_freq_variance = 0.0

    # 9. 연속 패턴
    consecutive_in_history = 0
    for h in range(min(20, n_history)):
        for k in range(5):
            if history_arr[h, k+1] - history_arr[h, k] == 1:
                consecutive_in_history += 1
    current_consecutive = 0
    for k in range(5):
        if nums_arr[k+1] - nums_arr[k] == 1:
            current_consecutive += 1
    avg_consecutive = consecutive_in_history / min(20, n_history) if n_history > 0 else 0
    f_consecutive_pattern = current_consecutive - avg_consecutive

    # 10. 구간 편차
    zones = [0, 0, 0]  # low, mid, high
    for num in nums_arr:
        n = int(num)
        if 1 <= n <= 15:
            zones[0] += 1
        elif 16 <= n <= 30:
            zones[1] += 1
        elif 31 <= n <= 45:
            zones[2] += 1

    avg_zones = np.zeros(3, dtype=np.float64)
    for h in range(min(20, n_history)):
        for k in range(6):
            n = history_arr[h, k]
            if 1 <= n <= 15:
                avg_zones[0] += 1.0
            elif 16 <= n <= 30:
                avg_zones[1] += 1.0
            elif 31 <= n <= 45:
                avg_zones[2] += 1.0
    avg_zones /= min(20, n_history)

    zone_diff = 0.0
    for i in range(3):
        zone_diff += abs(zones[i] - avg_zones[i])
    f_zone_deviation = zone_diff / 6.0

    # 11. 추세 (상승/하강)
    if len(recent_sums) >= 5:
        # 선형 회귀 (간단 버전)
        x = np.arange(5, dtype=np.float64)
        y = np.array(recent_sums[:5], dtype=np.float64)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        slope = np.sum((x - mean_x) * (y - mean_y)) / (np.sum((x - mean_x) ** 2) + 1e-6)
        f_trend = slope / 10.0
    else:
        f_trend = 0.0

    return np.array([
        f_recent_freq, f_no_show_avg, f_pair_freq, f_gap_similarity,
        f_sum_trend, f_recent_overlap, f_sum_deviation, f_freq_variance,
        f_consecutive_pattern, f_zone_deviation, f_trend
    ], dtype=np.float64)


@njit(parallel=True, cache=True, fastmath=True)
def _compute_history_features_batch(nums_batch: np.ndarray, history_arr: np.ndarray) -> np.ndarray:
    """
    히스토리 특징 배치 처리 (병렬)

    Args:
        nums_batch: (N, 6) 번호 배열
        history_arr: (M, 6) 히스토리 배열

    Returns:
        (N, 11) 특징 행렬
    """
    n_samples = len(nums_batch)
    features = np.zeros((n_samples, 11), dtype=np.float64)

    # prange로 병렬화
    for i in prange(n_samples):
        features[i] = _compute_history_features_from_array(nums_batch[i], history_arr)

    return features


def _compute_history_features(nums: list[int], history_df: pd.DataFrame | None) -> np.ndarray:
    """
    히스토리 기반 특징 계산 (11개)

    pandas 의존성 때문에 numba 불가, 하지만 호출 횟수가 적어서 괜찮음
    """
    if history_df is None or history_df.empty:
        return np.zeros(11, dtype=np.float64)

    nums_set = set(nums)
    gaps = np.diff(np.array(nums, dtype=np.float64))

    # 1. 최근 N회 출현 빈도
    recent_n = min(50, len(history_df))
    recent_nums = []
    for row in history_df.head(recent_n).itertuples(index=False):
        for val in row:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    recent_nums.append(v)
            except (ValueError, TypeError):
                continue

    if recent_nums:
        from collections import Counter
        freq_counter = Counter(recent_nums)
        avg_freq = np.mean([freq_counter.get(n, 0) for n in nums])
        max_freq = max(freq_counter.values()) if freq_counter else 1
        f_recent_freq = avg_freq / max_freq if max_freq > 0 else 0.0
    else:
        f_recent_freq = 0.0

    # 2. 미출현 기간
    no_show_periods = []
    for n in nums:
        last_seen = -1
        for idx, row in enumerate(history_df.itertuples(index=False)):
            row_nums = set()
            for val in row:
                try:
                    v = int(val)
                    if 1 <= v <= 45:
                        row_nums.add(v)
                except (ValueError, TypeError):
                    continue
            if n in row_nums:
                last_seen = idx
                break
        no_show_periods.append(last_seen if last_seen >= 0 else 100)
    f_no_show_avg = np.mean(no_show_periods) / 100.0

    # 3. 번호 쌍 동시 출현
    pair_counts = []
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            pair = (nums[i], nums[j])
            pair_count = 0
            for row in history_df.head(100).itertuples(index=False):
                row_nums = set()
                for val in row:
                    try:
                        v = int(val)
                        if 1 <= v <= 45:
                            row_nums.add(v)
                    except (ValueError, TypeError):
                        continue
                if pair[0] in row_nums and pair[1] in row_nums:
                    pair_count += 1
            pair_counts.append(pair_count)

    if pair_counts:
        avg_pair = np.mean(pair_counts)
        max_pair = max(pair_counts) if pair_counts else 1
        f_pair_freq = avg_pair / max_pair if max_pair > 0 else 0.0
    else:
        f_pair_freq = 0.0

    # 4. 간격 패턴 유사도
    if len(gaps) > 0:
        similarities = []
        for row in history_df.head(20).itertuples(index=False):
            row_nums = []
            for val in row:
                try:
                    v = int(val)
                    if 1 <= v <= 45:
                        row_nums.append(v)
                except (ValueError, TypeError):
                    continue
            if len(row_nums) == 6:
                row_nums_sorted = sorted(row_nums)
                row_gaps = np.diff(row_nums_sorted)
                if len(row_gaps) == len(gaps):
                    dot = np.dot(gaps, row_gaps)
                    norm1 = np.linalg.norm(gaps)
                    norm2 = np.linalg.norm(row_gaps)
                    if norm1 > 0 and norm2 > 0:
                        sim = dot / (norm1 * norm2)
                        similarities.append(max(0.0, sim))
        f_gap_similarity = np.mean(similarities) if similarities else 0.0
    else:
        f_gap_similarity = 0.0

    # 5. 합계 추세
    recent_sums = []
    for row in history_df.head(20).itertuples(index=False):
        row_nums = []
        for val in row:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    row_nums.append(v)
            except (ValueError, TypeError):
                continue
        if row_nums:
            recent_sums.append(sum(row_nums))

    if recent_sums:
        current_sum = sum(nums)
        avg_sum = np.mean(recent_sums)
        std_sum = np.std(recent_sums) if len(recent_sums) > 1 else 1.0
        if std_sum > 0:
            z_score = (current_sum - avg_sum) / std_sum
            f_sum_trend = max(0.0, min(1.0, (z_score + 2.0) / 4.0))
        else:
            f_sum_trend = 0.5
    else:
        f_sum_trend = 0.5

    # 6-11. 추가 고급 히스토리 특징

    # 6. 최근 10회 중복 개수
    recent_10 = set()
    for row in history_df.head(10).itertuples(index=False):
        for val in row:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    recent_10.add(v)
            except (ValueError, TypeError):
                continue
    f_recent_overlap = len(nums_set & recent_10) / 6.0

    # 7. 히스토리 평균 대비 합계
    all_sums = []
    for row in history_df.itertuples(index=False):
        row_nums = []
        for val in row:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    row_nums.append(v)
            except (ValueError, TypeError):
                continue
        if len(row_nums) == 6:
            all_sums.append(sum(row_nums))

    if all_sums:
        overall_avg = np.mean(all_sums)
        f_sum_deviation = abs(sum(nums) - overall_avg) / overall_avg if overall_avg > 0 else 0.0
        f_sum_deviation = min(1.0, f_sum_deviation)
    else:
        f_sum_deviation = 0.0

    # 8. 출현 빈도 분산
    if recent_nums:
        freqs = [freq_counter.get(n, 0) for n in nums]
        f_freq_variance = np.std(freqs) / (np.mean(freqs) + 1)
        f_freq_variance = min(1.0, f_freq_variance)
    else:
        f_freq_variance = 0.0

    # 9. 연속 출현 패턴 (n-1회와 n회에 공통 번호)
    consecutive_overlaps = []
    for i in range(min(20, len(history_df) - 1)):
        set1 = set()
        set2 = set()
        for val in history_df.iloc[i]:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    set1.add(v)
            except (ValueError, TypeError):
                continue
        for val in history_df.iloc[i+1]:
            try:
                v = int(val)
                if 1 <= v <= 45:
                    set2.add(v)
            except (ValueError, TypeError):
                continue
        if len(set1) == 6 and len(set2) == 6:
            consecutive_overlaps.append(len(set1 & set2))

    if consecutive_overlaps:
        avg_consecutive_overlap = np.mean(consecutive_overlaps)
        my_prev_overlap = 0
        if len(history_df) > 0:
            prev_set = set()
            for val in history_df.iloc[0]:
                try:
                    v = int(val)
                    if 1 <= v <= 45:
                        prev_set.add(v)
                except (ValueError, TypeError):
                    continue
            my_prev_overlap = len(nums_set & prev_set)
        f_consecutive_pattern = abs(my_prev_overlap - avg_consecutive_overlap) / 6.0
        f_consecutive_pattern = 1.0 - min(1.0, f_consecutive_pattern)
    else:
        f_consecutive_pattern = 0.5

    # 10. 구간별 출현 빈도 편차
    zone_freqs = [0] * 3  # low(1-15), mid(16-30), high(31-45)
    if recent_nums:
        for n in recent_nums:
            if 1 <= n <= 15:
                zone_freqs[0] += 1
            elif 16 <= n <= 30:
                zone_freqs[1] += 1
            elif 31 <= n <= 45:
                zone_freqs[2] += 1

        zone_expected = len(recent_nums) / 3
        my_zones = [0, 0, 0]
        for n in nums:
            if 1 <= n <= 15:
                my_zones[0] += 1
            elif 16 <= n <= 30:
                my_zones[1] += 1
            elif 31 <= n <= 45:
                my_zones[2] += 1

        expected_ratio = [zone_freqs[i] / zone_expected for i in range(3)]
        my_ratio = [my_zones[i] / 2 for i in range(3)]  # 6개/3=2 expected per zone

        chi2 = sum((my_ratio[i] - expected_ratio[i])**2 / (expected_ratio[i] + 0.1) for i in range(3))
        f_zone_deviation = 1.0 / (1.0 + chi2)
    else:
        f_zone_deviation = 0.5

    # 11. 최근 추세 (최근 20회 평균과 전체 평균 비교)
    if len(history_df) > 20:
        recent_20_nums = []
        for row in history_df.head(20).itertuples(index=False):
            for val in row:
                try:
                    v = int(val)
                    if 1 <= v <= 45:
                        recent_20_nums.append(v)
                except (ValueError, TypeError):
                    continue

        if recent_20_nums:
            recent_avg = np.mean(recent_20_nums)
            overall_avg_num = np.mean(recent_nums) if recent_nums else 23.0
            trend_direction = (recent_avg - overall_avg_num) / 45.0
            f_trend = (trend_direction + 1.0) / 2.0  # -1~1 → 0~1
        else:
            f_trend = 0.5
    else:
        f_trend = 0.5

    return np.array([
        f_recent_freq, f_no_show_avg, f_pair_freq, f_gap_similarity, f_sum_trend,
        f_recent_overlap, f_sum_deviation, f_freq_variance, f_consecutive_pattern,
        f_zone_deviation, f_trend
    ], dtype=np.float64)


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
    for _, row in history_df.iterrows():
        # n1~n6 컬럼만 추출 (round, date 제외)
        nums = []
        for col in history_df.columns:
            if col.lower() in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                try:
                    val = int(row[col])
                    if 1 <= val <= 45:
                        nums.append(val)
                except (ValueError, TypeError):
                    pass

        nums = sorted(nums)
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
def _compute_temporal_features(
    round_num: int | None,
    date_str: str | None,
    history_df: pd.DataFrame | None,
) -> np.ndarray:
    """
    시간 기반 특징 계산 (7개)

    시간적 패턴: 요일, 월, 계절, 회차 등

    Args:
        round_num: 현재 회차 번호
        date_str: 날짜 문자열 (예: "2025.12.13")
        history_df: 히스토리 DataFrame

    Returns:
        (7,) 시간 특징 벡터
    """
    import math
    from datetime import datetime

    # 기본값 (시간 정보 없을 때)
    if date_str is None or round_num is None:
        return np.zeros(7, dtype=np.float64)

    try:
        # 날짜 파싱
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")

        # 1-2. 요일 (sin/cos 인코딩)
        # 0=월요일, 6=일요일
        weekday = date_obj.weekday()
        f_weekday_sin = math.sin(2 * math.pi * weekday / 7)
        f_weekday_cos = math.cos(2 * math.pi * weekday / 7)

        # 3-4. 월 (sin/cos 인코딩)
        month = date_obj.month
        f_month_sin = math.sin(2 * math.pi * month / 12)
        f_month_cos = math.cos(2 * math.pi * month / 12)

        # 5. 계절 (0=봄, 1=여름, 2=가을, 3=겨울)
        season = (month % 12) // 3  # 3,4,5월=봄, 6,7,8=여름, 9,10,11=가을, 12,1,2=겨울
        f_season = season / 3.0  # 0.0 ~ 1.0

        # 6. 회차 번호 정규화
        # 1~1500 범위로 가정
        f_round = round_num / 1500.0

        # 7. 연중 몇 번째 주 (정규화)
        # 1년 = 52주
        week_of_year = date_obj.isocalendar()[1]
        f_week_of_year = week_of_year / 52.0

        return np.array([
            f_weekday_sin, f_weekday_cos,
            f_month_sin, f_month_cos,
            f_season,
            f_round,
            f_week_of_year
        ], dtype=np.float64)

    except Exception:
        # 파싱 실패 시 0으로 반환
        return np.zeros(7, dtype=np.float64)


def _compute_temporal_features_batch(
    n_samples: int,
    round_num: int | None,
    date_str: str | None,
) -> np.ndarray:
    """
    배치로 시간 특징 계산 (N개 샘플 모두 같은 시간 정보 사용)

    Args:
        n_samples: 샘플 개수
        round_num: 예측 대상 회차
        date_str: 예측 대상 날짜

    Returns:
        (N, 7) 시간 특징 배열
    """
    # 단일 시간 특징 계산
    single_features = _compute_temporal_features(round_num, date_str, None)

    # N개 샘플 모두에 같은 시간 특징 적용
    return np.tile(single_features, (n_samples, 1))


def _set_features(
    nums: list[int],
    weights=None,
    history_df: pd.DataFrame | None = None,
    round_num: int | None = None,
    date_str: str | None = None,
) -> np.ndarray:
    """
    번호 조합의 특징 추출 (57개 특징)

    ⚡ Numba JIT 최적화 적용 → 5-10배 빠름!

    - Numba 최적화: 39개 특징 (C 속도)
    - 히스토리 기반: 11개 특징
    - 시간 기반: 7개 특징 (NEW!)
    """
    nums = sorted(nums)
    arr = np.array(nums, dtype=np.float64)

    # ===== NUMBA 최적화: 39개 특징 =====
    core_features = _compute_core_features(arr)

    # ===== 히스토리 기반: 11개 특징 =====
    hist_features = _compute_history_features(nums, history_df)

    # ===== 시간 기반: 7개 특징 =====
    temporal_features = _compute_temporal_features(round_num, date_str, history_df)

    # 결합 (57개)
    return np.concatenate([core_features, hist_features, temporal_features])


def train_ml_scorer(
    history_df: pd.DataFrame,
    weights=None,
    n_neg_per_pos: int = 5,
    max_rounds: int | None = 200,
    epochs: int = 120,
    lr: float = 0.05,
    use_hard_negatives: bool = True,
    model_type: str = "neural_network",
    randomness_learning: bool = True,
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
        use_hard_negatives: 하드 네거티브 샘플링
        model_type: 항상 "neural_network"
        randomness_learning: True면 "무작위성" 학습 (양성=당첨번호, 음성=편향조합)

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
    pos_meta = []  # (round, date) 시간 정보 저장

    # DataFrame 컬럼 확인
    has_round = 'round' in df.columns
    has_date = 'date' in df.columns

    for idx, row in df.iterrows():
        # round와 date 정보 추출 (있으면)
        round_num = row['round'] if has_round else None
        date_str = row['date'] if has_date else None

        # n1~n6 컬럼 값 추출
        nums = []
        for col in df.columns:
            if col.lower() in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                try:
                    val = int(row[col])
                    if 1 <= val <= 45:
                        nums.append(val)
                except (ValueError, TypeError):
                    pass

        nums = sorted(nums)
        if len(nums) == 6:
            pos_sets.append(nums)
            pos_meta.append((round_num, date_str))
    if not pos_sets:
        raise ValueError("유효한 양성 세트가 없습니다.")

    X_list = []
    y_list = []

    # 양성 샘플 (시간 정보 포함)
    for i, s in enumerate(pos_sets):
        round_num, date_str = pos_meta[i]
        X_list.append(_set_features(s, weights, history_df, round_num, date_str))
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

    # 음성 샘플 (시간 정보 없음 - 가상 조합이므로)
    for s in neg_sets:
        X_list.append(_set_features(s, weights, history_df, None, None))
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

    # Neural Network 모델 생성 (최적화된 하이퍼파라미터 적용)
    mode_str = "랜덤성학습" if randomness_learning else "분류학습"
    print(f"[ML 학습] 모델: 신경망 ({mode_str}, 100-80-60-40-20 layers, tanh)")
    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(100, 80, 60, 40, 20),  # 최적화: 5층 깊은 네트워크
        activation='tanh',                          # 최적화: tanh 활성화 함수
        solver='adam',
        learning_rate_init=0.005,                   # 최적화: 0.01 → 0.005
        alpha=0.0005,                               # 최적화: L2 정규화 강화
        batch_size=200,                             # 최적화: 고정 배치 크기
        max_iter=300,                               # 최적화: 200 → 300
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
    round_num: int | None = None,
    date_str: str | None = None,
) -> float:
    """
    ML 모델로 번호 조합 점수 계산

    모든 모델 타입 지원:
    - logistic: 로지스틱 회귀
    - random_forest: 랜덤 포레스트
    - gradient_boosting: 그래디언트 부스팅
    - neural_network: 신경망
    - neural_network_ensemble: K-Fold 앙상블
    - stacking: Stacking 앙상블 (K-Fold + Meta Model)

    Args:
        round_num: 예측 대상 회차 (예: 1203)
        date_str: 예측 대상 날짜 (예: "2025.12.20")
    """
    if model is None:
        return 0.0

    # 특징 추출 (시간 정보 포함)
    feats = _set_features(nums, weights, history_df, round_num, date_str)
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

    elif model_type == "neural_network_ensemble":
        # K-Fold 앙상블 모델 (10개 모델의 평균)
        models = model.get("models")
        if models is None or len(models) == 0:
            return 0.0

        try:
            # 모든 모델의 예측 확률 평균
            probs = []
            for m in models:
                proba = m.predict_proba(x)[0, 1]
                probs.append(proba)

            ensemble_prob = float(np.mean(probs))
            return ensemble_prob
        except Exception:
            return 0.0

    elif model_type == "stacking":
        # Stacking 앙상블 (Level 0: 베이스 모델 → Level 1: 메타 모델)
        base_models = model.get("base_models")
        meta_model = model.get("meta_model")

        if base_models is None or meta_model is None:
            return 0.0

        if len(base_models) == 0:
            return 0.0

        try:
            # Level 0: 베이스 모델들의 예측 생성
            base_predictions = []
            for m in base_models:
                proba = m.predict_proba(x)[0, 1]
                base_predictions.append(proba)

            base_predictions = np.array(base_predictions).reshape(1, -1)

            # 메타 입력: 베이스 예측 + 정규화된 원본 특징
            meta_input = np.hstack([base_predictions, x])

            # Level 1: 메타 모델로 최종 예측
            final_proba = meta_model.predict_proba(meta_input)[0, 1]
            return float(final_proba)
        except Exception:
            return 0.0

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


def ml_score_sets_batch(
    sets: list[list[int]],
    model: dict | None,
    weights=None,
    history_df: pd.DataFrame | None = None,
    round_num: int | None = None,
    date_str: str | None = None,
) -> list[float]:
    """
    여러 번호 세트의 ML 점수를 한번에 계산 (배치 처리)

    ⚡ StackingModelWrapper의 배치 예측 기능을 사용하여 17.5배 빠름!

    Parameters:
        sets: 번호 세트 리스트 (각 세트는 6개 정수 리스트)
        model: ML 모델 딕셔너리
        weights: 번호 가중치 (45개)
        history_df: 히스토리 데이터프레임
        round_num: 예측 대상 회차 (예: 1203)
        date_str: 예측 대상 날짜 (예: "2025.12.20")

    Returns:
        각 세트의 ML 점수 리스트 (0.0 ~ 1.0)
    """
    if model is None or len(sets) == 0:
        return [0.0] * len(sets)

    # 1. 모든 세트의 특징 추출 (57개 특징 × N개 세트, 시간 정보 포함)
    features_list = []
    for nums in sets:
        feats = _set_features(nums, weights, history_df, round_num, date_str)
        features_list.append(feats)

    X = np.array(features_list)  # Shape: (N, 57)

    # 2. 정규화
    mu = model.get("mu")
    sig = model.get("sigma")
    if mu is None or sig is None:
        return [0.0] * len(sets)

    Xn = (X - mu) / sig  # Shape: (N, 57)

    # 3. 배치 예측
    model_type = model.get("type", "logistic")

    if model_type == "stacking":
        # Stacking 모델: 'model' 키에 Wrapper가 있으면 사용
        if 'model' in model:
            try:
                # StackingModelWrapper의 배치 예측 (병렬 처리)
                probs = model['model'].predict_proba(Xn)[:, 1]
                return probs.tolist()
            except Exception:
                # Wrapper 실패 시 개별 처리로 폴백
                pass

        # Wrapper가 없으면 개별 처리
        scores = []
        for i in range(len(sets)):
            score = ml_score_set(sets[i], model, weights, history_df, round_num, date_str)
            scores.append(score)
        return scores

    elif model_type == "neural_network_ensemble":
        # K-Fold 앙상블: 각 모델로 배치 예측 후 평균
        models = model.get("models")
        if models is None or len(models) == 0:
            return [0.0] * len(sets)

        try:
            all_probs = []
            for m in models:
                probs = m.predict_proba(Xn)[:, 1]
                all_probs.append(probs)

            # 모든 모델의 평균
            ensemble_probs = np.mean(all_probs, axis=0)
            return ensemble_probs.tolist()
        except Exception:
            return [0.0] * len(sets)

    elif model_type == "logistic":
        # 로지스틱 회귀: 수동 계산
        w = model.get("w")
        b = model.get("b", 0.0)
        if w is None:
            return [0.0] * len(sets)

        z = Xn @ w + b  # Shape: (N,)
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs.tolist()

    elif model_type in ["random_forest", "gradient_boosting", "neural_network"]:
        # sklearn 모델: predict_proba 배치 호출
        sklearn_model = model.get("model")
        if sklearn_model is None:
            return [0.0] * len(sets)

        try:
            probs = sklearn_model.predict_proba(Xn)[:, 1]
            return probs.tolist()
        except Exception:
            return [0.0] * len(sets)

    else:
        return [0.0] * len(sets)


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
    for _, row in history_df.iterrows():
        # n1~n6 컬럼만 추출 (round, date 제외)
        nums = []
        for col in history_df.columns:
            if col.lower() in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                try:
                    val = int(row[col])
                    if 1 <= val <= 45:
                        nums.append(val)
                except (ValueError, TypeError):
                    pass
        nums = sorted(set(nums))
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
    round_num: int | None = None,
    date_str: str | None = None,
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

        # ========== 고도화 고전 알고리즘 10개 추가 ==========

        # 통계적 알고리즘 (3개)
        if history_df is not None and history_df.empty is False:
            try:
                cands.append((gen_hot_cold_balanced(history_df, weights, exclude_set), "classical"))
            except Exception:
                pass

        try:
            cands.append((gen_consecutive_controlled(weights, exclude_set), "classical"))
        except Exception:
            pass

        try:
            cands.append((gen_sum_range_controlled(weights, exclude_set), "classical"))
        except Exception:
            pass

        # 조합론적 알고리즘 (3개)
        try:
            cands.append((gen_zone_balanced(weights, exclude_set), "classical"))
        except Exception:
            pass

        try:
            cands.append((gen_last_digit_diverse(weights, exclude_set), "classical"))
        except Exception:
            pass

        try:
            cands.append((gen_prime_mixed(weights, exclude_set), "classical"))
        except Exception:
            pass

        # 히스토리 분석 기반 (4개)
        if history_df is not None and history_df.empty is False:
            try:
                cands.append((gen_temporal_strategy(history_df, weights, exclude_set), "classical"))
            except Exception:
                pass

            try:
                cands.append((gen_frequency_inverted(history_df, weights, exclude_set), "classical"))
            except Exception:
                pass

            try:
                cands.append((gen_cycle_pattern(history_df, weights, exclude_set), "classical"))
            except Exception:
                pass

            # ML 가이드 (ml_model이 있을 때만, 시간 정보 포함)
            if ml_model is not None:
                try:
                    cands.append((gen_ml_guided(ml_model, history_df, weights, exclude_set,
                                               round_num=round_num, date_str=date_str), "classical"))
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

        # ========== 순차 평가 (단순하고 효율적) ==========
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
                ml_score_set(cand, ml_model, weights=weights, history_df=history_df, round_num=round_num, date_str=date_str)
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

        # 최고 점수 후보가 없으면 랜덤 생성
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

