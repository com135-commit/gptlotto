#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로또 히스토리 데이터 처리 모듈
- CSV 로딩
- HM 엔진 (가중치 계산)
- 현실적인 인기도 가중치 계산
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ncols: list[str] = []
    for k in ["n1", "n2", "n3", "n4", "n5", "n6"]:
        found = None
        for c in df.columns:
            if c.lower() == k or c.lower().endswith(k):
                found = c
                break
        if not found:
            raise ValueError("CSV에 n1..n6 컬럼이 필요합니다.")
        ncols.append(found)
    out = df[ncols].copy()
    out = out.apply(pd.to_numeric, errors="coerce").dropna().astype(int)
    for c in out.columns:
        out = out[(out[c] >= 1) & (out[c] <= 45)]
    return out.reset_index(drop=True)


def compute_weights(
    history_df: pd.DataFrame,
    lookback: int | None,
    strategy: str,
    exclude_recent: int = 0,
):
    if history_df is None or history_df.empty or strategy == "사용 안 함":
        return None, set()
    df = history_df.copy()
    if lookback is not None and lookback > 0 and lookback < len(df):
        df = df.tail(lookback)
    excl_set: set[int] = set()
    if exclude_recent and exclude_recent > 0:
        tail = history_df.tail(exclude_recent)
        excl_set = set(np.unique(tail.values.flatten()))

    flat = df.values.flatten()

    freq = np.zeros(46, dtype=float)
    for v in flat:
        if 1 <= v <= 45:
            freq[int(v)] += 1.0
    freq = freq[1:]
    freq_norm = freq / freq.max() if freq.max() > 0 else np.ones(45) / 45.0

    last_idx = {n: -1 for n in range(1, 46)}
    for i in range(len(history_df)):
        row = history_df.iloc[i].values.tolist()
        for v in row:
            v = int(v)
            if 1 <= v <= 45:
                last_idx[v] = i
    dist = np.array(
        [
            (len(history_df) - 1 - last_idx[n]) if last_idx[n] >= 0 else len(history_df)
            for n in range(1, 46)
        ],
        dtype=float,
    )
    overdue_norm = dist / dist.max() if dist.max() > 0 else np.ones(45) / 45.0

    zone_counts = np.zeros(3)
    for v in flat:
        v = int(v)
        if 1 <= v <= 15:
            zone_counts[0] += 1
        elif 16 <= v <= 30:
            zone_counts[1] += 1
        elif 31 <= v <= 45:
            zone_counts[2] += 1
    z_norm = zone_counts / zone_counts.max() if zone_counts.max() > 0 else np.ones(3) / 3.0
    zone_score = np.zeros(45)
    for n in range(1, 46):
        if n <= 15:
            zone_score[n - 1] = z_norm[0]
        elif n <= 30:
            zone_score[n - 1] = z_norm[1]
        else:
            zone_score[n - 1] = z_norm[2]

    pair_counts = np.zeros((46, 46))
    for row in df.itertuples(index=False):
        nums = sorted(set(int(x) for x in row if 1 <= int(x) <= 45))
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                pair_counts[a, b] += 1.0
                pair_counts[b, a] += 1.0
    pair_sum = np.zeros(45)
    for n in range(1, 46):
        pair_sum[n - 1] = pair_counts[n, :].sum()
    pair_norm = pair_sum / pair_sum.max() if pair_sum.max() > 0 else np.zeros(45)

    master = (
        0.35 * freq_norm
        + 0.25 * overdue_norm
        + 0.20 * zone_score
        + 0.20 * pair_norm
    )
    master = np.maximum(master, 1e-9)
    master = master / master.sum()

    if strategy == "Balanced(중립화)":
        w = master
    elif strategy == "Hot(고빈도)":
        w = master * (0.7 * freq_norm + 0.3)
    elif strategy == "Cold(저빈도)":
        inv = (freq.max() - freq) if freq.max() > 0 else np.ones_like(freq)
        inv_norm = inv / (inv.max() if inv.max() > 0 else 1.0)
        w = master * (0.7 * inv_norm + 0.3)
    elif strategy == "Overdue(오래 안 나온)":
        w = master * (0.7 * overdue_norm + 0.3)
    else:
        w = master
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    return w, excl_set


# ------------------ 현실적인 인기 분포 (가상조작용) ------------------
def compute_realistic_popularity_weights(
    history_df: pd.DataFrame | None,
    hm_weights=None,
    user_sets: list[list[int]] | None = None,
) -> np.ndarray:
    """
    '사람들이 실제로 많이 고를 것 같은' 번호 인기 분포를 대충 흉내낸다.

    구성:
      - hm_weights : HM 엔진(히스토리 기반) 가중치
      - birthday   : 1~31 (생일 숫자) 선호
      - shape      : 중간 번호대(20 근처) 약간 더 선호
      - user_sets  : 세트 편집 탭의 사용자 취향 반영
    """
    # 1) HM 기반 베이스
    if hm_weights is not None:
        base = np.array(hm_weights, dtype=float)
        if base.size != 45 or base.sum() <= 0:
            base = np.ones(45, dtype=float)
    else:
        base = np.ones(45, dtype=float)
    base = base / base.sum()

    # 2) 생일 버프 (1~31 ↑, 32~45 ↓)
    birthday = np.zeros(45, dtype=float)
    for n in range(1, 46):
        if 1 <= n <= 31:
            birthday[n - 1] = 1.0
        else:
            birthday[n - 1] = 0.45
    birthday = birthday / birthday.sum()

    # 3) 번호대 형상 (대략 20대 중심 완만한 종 모양)
    xs = np.arange(1, 46)
    mu, sigma = 22.0, 10.0
    shape = np.exp(-((xs - mu) ** 2) / (2.0 * sigma ** 2))
    shape = shape / shape.sum()

    # 4) 사용자 취향 (세트 편집 탭)
    if user_sets:
        user_weight = np.zeros(45, dtype=float)
        for s in user_sets:
            for v in s:
                if 1 <= v <= 45:
                    user_weight[v - 1] += 1.0
        if user_weight.sum() > 0:
            user_weight = user_weight / user_weight.sum()
        else:
            user_weight = np.ones(45, dtype=float) / 45.0
    else:
        user_weight = np.ones(45, dtype=float) / 45.0

    # 5) 믹스
    #   HM 40% + 생일 35% + 형상 15% + 사용자취향 10%
    w = (
        0.40 * base +
        0.35 * birthday +
        0.15 * shape +
        0.10 * user_weight
    )

    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    return w


