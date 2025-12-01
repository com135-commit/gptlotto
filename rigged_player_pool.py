def _build_synthetic_player_pool_chunk(n_players: int, weights) -> dict[tuple[int, ...], int]:
    """
    멀티프로세싱용: 일부 플레이어 수 만큼 티켓을 생성해서 부분 딕셔너리 반환.

    반환 형식:
      { (n1, n2, n3, n4, n5, n6): 그 조합을 산 가상 플레이어 수 }
    """

    rng = np.random.default_rng()

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

    # 디케이드(10단위) 구간
    decade_ranges = [
        np.arange(1, 11, dtype=int),     # 1~10
        np.arange(11, 21, dtype=int),    # 11~20
        np.arange(21, 31, dtype=int),    # 21~30
        np.arange(31, 41, dtype=int),    # 31~40
        np.arange(41, 46, dtype=int),    # 41~45
    ]

    # 소수, 배수, 피보나치 같은 수학 패턴용 집합
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43], dtype=int)
    multiples_of_3 = np.array([n for n in range(3, 46) if n % 3 == 0], dtype=int)
    multiples_of_7 = np.array([n for n in range(7, 46) if n % 7 == 0], dtype=int)
    fibo_base = [1, 2, 3, 5, 8, 13, 21, 34]
    fibo_nums = np.array([n for n in fibo_base if 1 <= n <= 45], dtype=int)

    # --- 유틸 함수들 ---

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
        """
        s = {int(v) for v in base if 1 <= int(v) <= 45}
        while len(s) < 6:
            v = int(rng.choice(xs, p=p_all))
            s.add(v)
        return tuple(sorted(s))

    def sum_constrained_ticket(sum_min: int, sum_max: int, max_try: int = 40) -> tuple[int, ...]:
        """
        합계가 [sum_min, sum_max] 범위에 들어오는 티켓을 만들려고 시도.
        실패하면 그냥 HM 랜덤 티켓으로 폴백.
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
          - 인접 또는 비슷한 번호 2개를 먼저 선택
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
        return sum_constrained_ticket(sum_min, max_try=max_try, sum_max=sum_max)

    # --- 패턴 함수들 정의 ---

    def pat_hm_random():
        # 1) 순수 HM 가중 랜덤
        return sample_from_candidates(xs, 6)

    def pat_birthday():
        # 2) 생일형 (1~31) + 가끔 월(1~12)을 섞는 변형
        if rng.random() < 0.7:
            birthday_range = np.arange(1, 32, dtype=int)
        else:
            birthday_range = np.arange(1, 32, dtype=int).tolist() + np.arange(1, 13, dtype=int).tolist()
        return sample_from_candidates(birthday_range, 6)

    def pat_low4_high2():
        # 3) 저고 패턴(저4 고2)
        base_low = sample_from_candidates(low_nums, 4)
        base_high = sample_from_candidates(high_nums, 2)
        return base_low + base_high

    def pat_low2_high4():
        # 4) 저고 패턴(저2 고4)
        base_low = sample_from_candidates(low_nums, 2)
        base_high = sample_from_candidates(high_nums, 4)
        return base_low + base_high

    def pat_odd4_even2():
        # 5) 홀짝 패턴 (홀4 짝2)
        # 일부는 3의 배수나 7의 배수를 섞어주는 변형
        if rng.random() < 0.3 and multiples_of_3.size >= 4:
            base = sample_from_candidates(odds, 2) + sample_from_candidates(multiples_of_3, 2)
            rest = sample_from_candidates(evens, 2)
            return base + rest
        ticket = sample_from_candidates(odds, 4) + sample_from_candidates(evens, 2)
        return ticket

    def pat_even4_odd2():
        # 6) 홀짝 패턴 (짝4 홀2)
        if rng.random() < 0.3 and multiples_of_3.size >= 2:
            base = sample_from_candidates(evens, 3) + sample_from_candidates(multiples_of_3, 1)
            rest = sample_from_candidates(odds, 2)
            return base + rest
        ticket = sample_from_candidates(evens, 4) + sample_from_candidates(odds, 2)
        return ticket

    def pat_full_consecutive():
        # 7) 6연속 패턴
        start = int(rng.integers(1, 40))  # 40까지면 start+5 = 45
        ticket = list(range(start, start + 6))
        return ticket

    def pat_partial_consecutive():
        # 8) 부분 연속 패턴 (3~4개 연속 + 나머지 랜덤)
        seg_len = int(rng.integers(3, 5))  # 3 또는 4개 연속
        max_start = 46 - seg_len
        start = int(rng.integers(1, max_start))
        seq = list(range(start, start + seg_len))
        rest_cnt = 6 - len(set(seq))
        rest = sample_from_candidates(xs, rest_cnt)
        ticket = seq + rest
        return ticket

    def pat_same_last_digit():
        # 9) 끝수(세로줄) 패턴
        d = int(rng.integers(0, 10))  # 0~9
        cand = [n for n in range(1, 46) if n % 10 == d]
        base = sample_from_candidates(cand, min(4, len(cand)))
        return base

    def pat_decade_cluster():
        # 10) 같은 10단위(디케이드) 클러스터 (3~4개)
        dec = decade_ranges[int(rng.integers(0, len(decade_ranges)))]
        main_cnt = int(rng.integers(3, 5))  # 3 또는 4개
        main = sample_from_candidates(dec, main_cnt)
        rest = sample_from_candidates(xs, 6 - len(set(main)))
        ticket = main + rest
        return ticket

    def pat_hot_focus():
        # 11) Hot 번호 위주
        base_hot = sample_from_candidates(hot_nums, 4)
        others = [n for n in xs if n not in base_hot]
        base_rest = sample_from_candidates(others, 2)
        ticket = base_hot + base_rest
        return ticket

    def pat_cold_focus():
        # 12) Cold 번호 위주
        cold_cnt = int(rng.integers(3, 5))   # 3 또는 4개
        base_cold = sample_from_candidates(cold_nums, cold_cnt)
        rest = sample_from_candidates(xs, 6 - len(set(base_cold)))
        ticket = base_cold + rest
        return ticket

    def pat_sum_range_A():
        # 13) 합계 범위 패턴 A (합계 100~160)
        return sum_constrained_ticket(100, 160)

    def pat_pair_sum_range():
        # 14) 합계 + '쌍' 패턴 (120~150)
        return pair_style_ticket(120, 150)

    def pat_hot_avoid():
        # 15) Hot 회피 패턴
        base = sample_from_candidates(non_hot_nums, 6)
        return base

    def pat_super_hot():
        # 16) 초핫(스트릭 느낌) 패턴
        base_hot2 = sample_from_candidates(super_hot_nums, 2)
        rest = sample_from_candidates(xs, 6 - len(set(base_hot2)))
        ticket = base_hot2 + rest
        return ticket

    def pat_low15_high30_mix():
        # 17) 초저번호/폼 편향
        low15 = np.arange(1, 16, dtype=int)   # 1~15
        main_cnt = int(rng.integers(3, 5))   # 3 또는 4개
        base_low15 = sample_from_candidates(low15, main_cnt)
        high30 = np.arange(30, 46, dtype=int)
        base_high30 = sample_from_candidates(high30, 1)
        rest = sample_from_candidates(xs, 6 - len(set(base_low15 + base_high30)))
        ticket = base_low15 + base_high30 + rest
        return ticket

    def pat_extreme_decade():
        # 18) 극단 디케이드 몰빵
        dec = decade_ranges[int(rng.integers(0, len(decade_ranges)))]
        main = sample_from_candidates(dec, min(5, len(dec)))
        rest = sample_from_candidates(xs, 6 - len(set(main)))
        ticket = main + rest
        return ticket

    def pat_extreme_allodd_or_alllow():
        # 19) 극단 선택: 올홀 or 올저, 혹은 소수/피보나치 몰빵
        r2 = rng.random()
        if r2 < 0.33:
            base = sample_from_candidates(odds, 6)
        elif r2 < 0.66:
            base = sample_from_candidates(low_nums, 6)
        else:
            # 소수/피보나치 섞기
            cand = np.unique(np.concatenate([primes, fibo_nums]))
            base = sample_from_candidates(cand, 6)
        return base

    # 패턴과 누적 확률(원래 if-elif 경계)을 그대로 매핑
    patterns = [
        (0.22,  pat_hm_random),
        (0.37,  pat_birthday),
        (0.47,  pat_low4_high2),
        (0.57,  pat_low2_high4),
        (0.65,  pat_odd4_even2),
        (0.73,  pat_even4_odd2),
        (0.76,  pat_full_consecutive),
        (0.80,  pat_partial_consecutive),
        (0.84,  pat_same_last_digit),
        (0.88,  pat_decade_cluster),
        (0.905, pat_hot_focus),
        (0.93,  pat_cold_focus),
        (0.95,  pat_sum_range_A),
        (0.965, pat_pair_sum_range),
        (0.98,  pat_hot_avoid),
        (0.987, pat_super_hot),
        (0.993, pat_low15_high30_mix),
        (0.997, pat_extreme_decade),
        (1.0,   pat_extreme_allodd_or_alllow),
    ]

    pool: dict[tuple[int, ...], int] = {}

    for _ in range(n_players):
        r = rng.random()
        ticket_raw = None

        for thresh, func in patterns:
            if r < thresh:
                ticket_raw = func()
                break

        # ticket_raw 가 튜플(이미 finalize된 티켓)이면 그대로,
        # 리스트/배열이면 finalize_ticket로 마무리
        if isinstance(ticket_raw, tuple):
            key = ticket_raw
        else:
            key = finalize_ticket(ticket_raw)

        pool[key] = pool.get(key, 0) + 1

    return pool
