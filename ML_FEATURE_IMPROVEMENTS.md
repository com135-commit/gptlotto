# ML 특징(Feature) 개선 방안

## 현재 특징 (10개)
1. f_mean: 평균값 / 45.0
2. f_std: 표준편차 / 20.0
3. evens: 짝수 개수 / 6.0
4. low: 1-20 번호 개수 / 6.0
5. mid: 21-35 번호 개수 / 6.0
6. high: 36-45 번호 개수 / 6.0
7. f_gmean: (간격 평균 - 8.0) / 8.0
8. f_gstd: 간격 표준편차 / 10.0
9. f_hmean: 히스토리 가중치 평균
10. f_hmax: 히스토리 가중치 최대값

---

## 추가 가능한 특징 (20개 이상)

### 📈 통계적 특징 (5개)
11. **최소값**: min(nums) / 45.0
12. **최대값**: max(nums) / 45.0
13. **중앙값**: median(nums) / 45.0
14. **범위**: (max - min) / 45.0
15. **사분위 범위(IQR)**: (Q3 - Q1) / 45.0

### 🔢 번호 패턴 특징 (8개)
16. **연속 번호 개수**: [1,2,3] → 3개 연속
17. **최대 연속 길이**: 가장 긴 연속 번호 체인
18. **끝자리 분포 엔트로피**: 0~9 끝자리의 다양성
19. **끝자리 중복**: 같은 끝자리가 몇 개?
20. **배수 개수**: 3의 배수, 5의 배수, 7의 배수
21. **소수 개수**: 소수가 몇 개?
22. **대칭성**: 번호들이 1~45 중앙(23)을 중심으로 대칭인지
23. **AC값** (평균 조합 복잡도): 조합론적 다양성 측정

### 📊 간격 패턴 특징 (5개)
24. **최소 간격**: min(gaps)
25. **최대 간격**: max(gaps)
26. **간격 중앙값**: median(gaps)
27. **간격 균일도**: 간격의 변동계수 (CV)
28. **간격 패턴**: 증가/감소 패턴 횟수

### 🎲 확률적 특징 (4개)
29. **과거 출현 빈도**: 각 번호가 과거에 나온 횟수 평균
30. **최근 출현도**: 최근 10회 내 출현 여부
31. **조합 희귀도**: 이 조합과 유사한 패턴의 과거 빈도
32. **번호 간 상관관계**: 함께 나온 적이 있는 번호 쌍 개수

### 🔄 고차원 특징 (4개)
33. **합의 끝자리**: sum(nums) % 10
34. **곱의 끝자리**: product(nums) % 10 (오버플로 주의)
35. **비트 패턴**: 각 번호를 비트로 표현했을 때 패턴
36. **해시 특징**: 조합의 해시값 기반 특징

---

## 구현 예시

```python
def _set_features_enhanced(
    nums: list[int],
    weights=None,
    history_df: pd.DataFrame | None = None,
) -> np.ndarray:
    nums = sorted(nums)
    arr = np.array(nums, dtype=float)

    # ===== 기존 특징 (10개) =====
    f_mean = arr.mean() / 45.0
    f_std = arr.std() / 20.0
    evens = sum(1 for v in nums if v % 2 == 0) / 6.0
    low = sum(1 for v in nums if 1 <= v <= 20) / 6.0
    mid = sum(1 for v in nums if 21 <= v <= 35) / 6.0
    high = sum(1 for v in nums if 36 <= v <= 45) / 6.0

    gaps = np.diff(arr)
    f_gmean = (gaps.mean() - 8.0) / 8.0 if len(gaps) > 0 else 0.0
    f_gstd = gaps.std() / 10.0 if len(gaps) > 0 else 0.0

    # 히스토리 가중치
    if weights is not None:
        w_arr = np.array(weights, dtype=float)
        ww = np.array([w_arr[int(v) - 1] for v in nums])
        f_hmean = float(ww.mean()) * len(w_arr)
        f_hmax = float(ww.max()) * len(w_arr)
    else:
        f_hmean = 0.0
        f_hmax = 0.0

    # ===== 새로운 특징 (20개+) =====

    # 통계적 특징
    f_min = arr.min() / 45.0
    f_max = arr.max() / 45.0
    f_median = float(np.median(arr)) / 45.0
    f_range = (arr.max() - arr.min()) / 45.0
    q1, q3 = np.percentile(arr, [25, 75])
    f_iqr = (q3 - q1) / 45.0

    # 번호 패턴
    consecutive_count = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    f_consecutive = consecutive_count / 5.0  # 최대 5쌍

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

    # 끝자리 분포
    last_digits = [n % 10 for n in nums]
    unique_last_digits = len(set(last_digits))
    f_last_digit_diversity = unique_last_digits / 6.0

    # 끝자리 중복
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

    # 간격 패턴
    if len(gaps) > 0:
        f_gap_min = gaps.min() / 10.0
        f_gap_max = gaps.max() / 10.0
        f_gap_median = float(np.median(gaps)) / 10.0
        f_gap_cv = (gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    else:
        f_gap_min = f_gap_max = f_gap_median = f_gap_cv = 0.0

    # 합과 곱의 특징
    f_sum_last_digit = (sum(nums) % 10) / 10.0

    # 과거 출현 빈도 (history_df 활용)
    f_freq_avg = 0.0
    f_recent = 0.0
    if history_df is not None and not history_df.empty:
        # 각 번호의 출현 빈도
        all_nums = []
        for _, row in history_df.iterrows():
            for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                if col in row:
                    all_nums.append(int(row[col]))

        from collections import Counter
        freq_counter = Counter(all_nums)
        avg_freq = np.mean([freq_counter.get(n, 0) for n in nums])
        max_freq = max(freq_counter.values()) if freq_counter else 1
        f_freq_avg = avg_freq / max_freq if max_freq > 0 else 0.0

        # 최근 10회 출현도
        recent_nums = set()
        for _, row in history_df.head(10).iterrows():
            for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
                if col in row:
                    recent_nums.add(int(row[col]))
        f_recent = sum(1 for n in nums if n in recent_nums) / 6.0

    # 대칭성 (중앙값 23 기준)
    center = 23.0
    symmetry_score = 1.0 - (sum(abs((n - center) - (center - n)) for n in nums) / (6 * 45))
    f_symmetry = max(0.0, symmetry_score)

    # ===== 특징 벡터 구성 =====
    feats = np.array([
        # 기존 10개
        f_mean, f_std, evens, low, mid, high,
        f_gmean, f_gstd, f_hmean, f_hmax,

        # 통계적 5개
        f_min, f_max, f_median, f_range, f_iqr,

        # 번호 패턴 8개
        f_consecutive, f_max_consecutive,
        f_last_digit_diversity, f_last_digit_dup,
        f_mult3, f_mult5, f_primes, f_symmetry,

        # 간격 패턴 4개
        f_gap_min, f_gap_max, f_gap_median, f_gap_cv,

        # 확률적 2개
        f_freq_avg, f_recent,

        # 고차원 1개
        f_sum_last_digit,
    ], dtype=float)

    return feats  # 총 30개 특징
```

---

## 추가 개선 사항

### 1. **더 많은 학습 데이터**
```python
# 현재: 최근 200회
max_rounds = 200

# 개선: 전체 데이터 (1000회+) 사용
max_rounds = None  # 전체 사용
```

### 2. **음성 샘플 개선**
```python
# 현재: 완전 랜덤
neg_sets = generate_random_sets(1000)

# 개선: "거의 비슷하지만 약간 다른" 샘플 추가
# → 모델이 더 세밀한 차이를 학습
for pos_set in pos_sets:
    # 1-2개 번호만 바꾼 변형 생성
    mutated = pos_set.copy()
    mutated[0] = random.choice([n for n in range(1, 46) if n not in mutated])
    neg_sets.append(mutated)
```

### 3. **더 강력한 모델**
```python
# 현재: 로지스틱 회귀
# 개선 옵션:

# A) 다층 신경망 (딥러닝)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layers=(50, 30, 10), max_iter=500)

# B) Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)

# C) Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200)
```

### 4. **학습 회차 늘리기**
```python
# 현재: 60 epochs
epochs = 60

# 개선: 조기 종료(Early Stopping) 적용
# 과적합 방지하면서 충분히 학습
epochs = 200
```

### 5. **교차 검증**
```python
from sklearn.model_selection import cross_val_score

# 모델 성능 정확히 측정
scores = cross_val_score(model, X, y, cv=5)
print(f"정확도: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 우선순위

1. ⭐⭐⭐ **특징 추가** (10개 → 30개) - 가장 큰 효과
2. ⭐⭐ **전체 데이터 사용** (200회 → 전체)
3. ⭐ **음성 샘플 개선** (하드 네거티브 추가)
4. ⭐ **더 강력한 모델** (선택사항, 과적합 주의)

---

## 예상 효과

**현재 모델**:
- 10개 특징
- 1,200개 샘플
- 단순 로지스틱 회귀
- → 정확도 약 70~75%

**개선 후 모델**:
- 30개 특징 ✨
- 6,000개+ 샘플 ✨
- 개선된 음성 샘플 ✨
- → 정확도 약 80~85% (예상)

**주의**: 로또는 독립 시행이므로 정확도가 높다고 당첨 확률이 올라가는 것은 아님!
하지만 "과거 패턴과 유사한" 번호 생성에는 도움이 됨.
