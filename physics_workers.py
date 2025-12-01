#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
물리 시뮬레이션 멀티프로세스 워커 함수들
Windows에서 안전하게 동작하도록 별도 모듈로 분리
"""

from __future__ import annotations


def physics_basic_worker(seed: int) -> list[int] | None:
    """기본 물리시뮬 워커"""
    try:
        from lotto_physics import LottoChamber
        chamber = LottoChamber(seed=seed)
        numbers = chamber.run_until_complete(target_count=6, max_time=60.0)
        if len(numbers) >= 6:
            return sorted(numbers[:6])
        return None
    except Exception as e:
        print(f"[physics_basic_worker] Error: {e}")
        return None


def physics_cfd_worker(args: tuple) -> list[int] | None:
    """CFD 물리시뮬 워커"""
    try:
        seed, grid_size = args
        from lotto_physics import LottoChamberCFD
        chamber = LottoChamberCFD(seed=seed, grid_size=grid_size)
        numbers = chamber.run_until_complete(target_count=6, max_time=60.0)
        if len(numbers) >= 6:
            return sorted(numbers[:6])
        return None
    except Exception as e:
        print(f"[physics_cfd_worker] Error: {e}")
        return None


def physics_ultimate_worker(args: tuple) -> list[int] | None:
    """MQLE 끝판왕 물리시뮬 워커"""
    try:
        seed, grid_size, history_weights, mqle_threshold = args
        from lotto_physics import LottoChamberUltimate, _qh_score_simple

        chamber = LottoChamberUltimate(
            seed=seed,
            grid_size=grid_size,
            history_weights=history_weights,
            mqle_threshold=mqle_threshold,
        )
        numbers = chamber.run_until_complete(target_count=6, max_time=60.0)

        if len(numbers) < 6:
            return None

        numbers = sorted(numbers[:6])
        qh_score = _qh_score_simple(numbers)

        # 조건 완화: 0.3 이상이면 통과
        if qh_score >= 0.3:
            return numbers
        return None
    except Exception as e:
        print(f"[physics_ultimate_worker] Error: {e}")
        return None
