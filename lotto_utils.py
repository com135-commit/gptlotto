#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로또 유틸리티 함수들
"""

from __future__ import annotations
import numpy as np

# 전역 랜덤 생성기
_rng = np.random.default_rng()


def parse_sets_from_text(text: str) -> list[list[int]]:
    """텍스트에서 로또 번호 세트 파싱 (ML 점수 포함 가능)"""
    sets: list[list[int]] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue

        # ML 점수 부분 제거 ("[ML: XX.XX%]" 형식)
        clean_line = line.split("[ML:")[0].strip() if "[ML:" in line else line

        # 숫자만 추출 (쉼표를 공백으로 치환)
        nums = [int(x) for x in clean_line.replace(",", " ").split()]

        if len(nums) != 6:
            raise ValueError(f"각 줄은 정확히 6개 숫자여야 합니다: '{line}'")
        if any(n < 1 or n > 45 for n in nums):
            raise ValueError(f"숫자는 1~45 범위여야 합니다: '{line}'")
        if len(set(nums)) != 6:
            raise ValueError(f"중복 없는 6개여야 합니다: '{line}'")
        sets.append(sorted(nums))
    if not sets:
        raise ValueError("최소 1개 세트가 필요합니다.")
    return sets


def sets_to_text(sets: list[list[int]]) -> str:
    """로또 번호 세트를 텍스트로 변환"""
    return "\n".join(" ".join(map(str, s)) for s in sets)


def sets_to_text_with_scores(
    sets: list[list[int]],
    scores: list[float]
) -> str:
    """
    로또 번호 세트를 ML 점수와 함께 텍스트로 변환

    예: "1 2 3 4 5 6  [ML: 87.5%]"

    Parameters:
        sets: 번호 세트 리스트 (ML 점수 내림차순으로 정렬된 상태)
        scores: 각 세트의 ML 점수 리스트 (0.0 ~ 1.0)

    Returns:
        ML 점수가 포함된 텍스트 (각 세트마다 한 줄)
    """
    if len(sets) != len(scores):
        # 길이가 다르면 점수 없이 출력
        return sets_to_text(sets)

    lines = []
    for s, score in zip(sets, scores):
        nums_str = " ".join(f"{n:2d}" for n in s)  # 2자리로 정렬
        score_pct = score * 100.0
        lines.append(f"{nums_str}  [ML: {score_pct:5.2f}%]")

    return "\n".join(lines)


def default_sets() -> list[list[int]]:
    """기본 로또 번호 세트"""
    return [
        [7, 14, 21, 27, 33, 41],
        [4, 9, 18, 26, 32, 40],
        [3, 10, 15, 25, 30, 43],
        [6, 13, 22, 29, 35, 44],
        [8, 17, 23, 31, 37, 42],
        [2, 5, 11, 19, 28, 39],
    ]


def get_rng():
    """전역 랜덤 생성기 반환"""
    return _rng
