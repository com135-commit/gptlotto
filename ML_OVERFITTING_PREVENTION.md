# ML 과적합 방지 완벽 가이드

## 현재 상황
- **문제**: 그래디언트 부스팅이 로또 번호 예측에서 과적합
- **본질적 이슈**: 로또는 완전 무작위인데 ML이 패턴을 찾으려고 함
- **결과**: 45번 91.5% 출현, 고수 62.9% 편중

---

## 1. 근본적 접근: ML 사용하지 않기 (권장)

### ✅ 최선의 방법
```python
ml_weight = 0.0  # ML 완전 비활성화
```

**이유:**
- 로또는 본질적으로 무작위 (예측 불가능)
- ML은 노이즈를 패턴으로 착각
- Physics + QH + Pattern만으로도 충분히 다양한 번호 생성 가능

**장점:**
- 과적합 원천 차단
- 계산 속도 향상
- 진정한 무작위성 보장

---

## 2. ML을 꼭 사용해야 한다면

### A. 특징(Feature) 설계 개선 ✅ (이미 적용)

#### 제거한 편향 특징 (10개)
```python
# 큰 번호 편향
- f_mean      # 평균이 클수록 높은 점수 → 큰 번호 선호
- f_max       # 최대값 → 45번에 유리
- f_min       # 최소값 → 1번에 유리
- f_median    # 중앙값 → 큰 번호 선호
- f_hmax      # 가중치 최대값
- f_hmean     # 가중치 평균
- f_gmean     # 간격 평균

# 히스토리 편향
- f_freq_avg  # 과거 자주 나온 번호 선호
- f_recent    # 최근 10회 출현 번호 선호
```

#### 유지한 중립 특징 (20개)
```python
# 분산/분포 특징 (중립적)
- f_std       # 표준편차
- f_range     # 범위
- f_iqr       # IQR (사분위수 범위)
- evens       # 짝수 비율
- low/mid/high # 번호 범위 분포 (균등 분할)

# 패턴 특징 (중립적)
- f_consecutive       # 연속 번호
- f_last_digit_*      # 끝자리 다양성
- f_mult3/mult5       # 배수 개수
- f_primes            # 소수 개수
- f_gap_*             # 간격 패턴
```

---

### B. Regularization 강화

#### 현재 설정
```python
# lotto_generators.py:787-794
GradientBoostingClassifier(
    n_estimators=100,      # 트리 개수
    learning_rate=0.1,     # 학습률
    max_depth=5,           # 트리 깊이
    min_samples_split=5,   # 분할 최소 샘플
    min_samples_leaf=2,    # 리프 최소 샘플
    random_state=42,
)
```

#### 권장 강화 설정
```python
GradientBoostingClassifier(
    n_estimators=50,          # 100 → 50 (트리 수 감소)
    learning_rate=0.05,       # 0.1 → 0.05 (학습률 감소)
    max_depth=3,              # 5 → 3 (깊이 제한 강화)
    min_samples_split=10,     # 5 → 10 (분할 어렵게)
    min_samples_leaf=5,       # 2 → 5 (리프 크기 증가)
    subsample=0.8,            # 새로 추가 (배깅)
    max_features='sqrt',      # 새로 추가 (특징 샘플링)
    random_state=42,
)
```

**효과:**
- 모델 복잡도 감소 → 일반화 성능 향상
- 과적합 억제

---

### C. Cross-Validation 개선

#### 현재 코드
```python
# lotto_generators.py:811
cv_scores = cross_val_score(sklearn_model, Xn, y, cv=5)
```

#### 개선 방안
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified K-Fold (클래스 비율 유지)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(sklearn_model, Xn, y, cv=skf)

# 과적합 판정
train_acc = sklearn_model.score(Xn, y)
cv_mean = cv_scores.mean()

if train_acc - cv_mean > 0.10:  # 10% 이상 차이
    print("[경고] 과적합 의심!")
    print(f"  훈련 정확도: {train_acc:.2%}")
    print(f"  검증 정확도: {cv_mean:.2%}")
```

---

### D. Early Stopping (신경망/부스팅)

```python
GradientBoostingClassifier(
    n_estimators=200,
    validation_fraction=0.2,   # 검증 세트 20%
    n_iter_no_change=10,       # 10회 개선 없으면 중단
    tol=0.0001,                # 개선 임계값
)
```

---

### E. 음성 샘플링 개선

#### 현재 문제
```python
# 양성 샘플: 과거 당첨 번호 (45번 14% 포함)
# 음성 샘플: 완전 랜덤 (45번 2.2% 포함)
# → ML이 "45번 포함 = 양성"으로 학습
```

#### 개선 방안
```python
def generate_negative_samples_balanced(pos_sets, n_neg):
    """
    양성 샘플과 유사한 분포로 음성 샘플 생성
    """
    # 양성 샘플의 번호 분포 분석
    all_nums = []
    for s in pos_sets:
        all_nums.extend(s)

    from collections import Counter
    freq = Counter(all_nums)

    # 빈도 기반 확률 분포
    probs = np.array([freq.get(i, 0) for i in range(1, 46)])
    probs = probs / probs.sum()

    # 음성 샘플 생성 (양성과 유사한 분포)
    neg_sets = []
    for _ in range(n_neg):
        nums = np.random.choice(range(1, 46), size=6, replace=False, p=probs)
        neg_sets.append(sorted(nums.tolist()))

    return neg_sets
```

---

### F. 앙상블 다양성 확보

```python
from sklearn.ensemble import VotingClassifier

# 여러 모델 조합
rf = RandomForestClassifier(n_estimators=100, max_depth=3)
gb = GradientBoostingClassifier(n_estimators=50, max_depth=3)
lr = LogisticRegression(C=0.1)  # L2 regularization

# 소프트 보팅
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft',
    weights=[1, 1, 2]  # 로지스틱에 더 높은 가중치
)
```

---

### G. ml_weight 대폭 감소

#### 현재 설정
```python
ml_weight = 0.3  # MQLE에서 ML 30%
```

#### 권장 설정
```python
ml_weight = 0.05  # 5%로 감소

# MQLE 점수 계산
total_score = (
    0.95 * (0.5*qh + 0.2*weight + 0.1*pattern + 0.2*diversity) +
    0.05 * ml_score
)
```

**이유:**
- ML 영향력 최소화
- Physics/QH/Pattern 위주로 선택
- 다양성 확보

---

## 3. 구체적 코드 수정 방안

### A. Regularization 강화

```python
# lotto_generators.py:785-794 수정
elif model_type == "gradient_boosting":
    print("[ML 학습] 모델: 그래디언트 부스팅 (과적합 방지 강화)")
    sklearn_model = GradientBoostingClassifier(
        n_estimators=50,          # 100 → 50
        learning_rate=0.05,       # 0.1 → 0.05
        max_depth=3,              # 5 → 3
        min_samples_split=10,     # 5 → 10
        min_samples_leaf=5,       # 2 → 5
        subsample=0.8,            # 새로 추가
        max_features='sqrt',      # 새로 추가
        random_state=42,
    )
```

### B. 과적합 탐지 추가

```python
# lotto_generators.py:810-815 수정
# 교차 검증
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(sklearn_model, Xn, y, cv=skf)
train_acc = sklearn_model.score(Xn, y)

print(f"[ML 학습 완료] 훈련 정확도: {train_acc:.2%}")
print(f"[교차 검증] 평균: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# 과적합 경고
overfitting_gap = train_acc - cv_scores.mean()
if overfitting_gap > 0.10:
    print(f"[⚠️  경고] 과적합 의심! (Gap: {overfitting_gap:.2%})")
    print(f"         ml_weight를 낮추거나 모델을 단순화하세요")
```

### C. ml_weight 기본값 변경

```python
# lotto_main.py에서 GUI 기본값 수정
# 현재
self.rig_ml_weight = tk.DoubleVar(value=0.3)

# 수정
self.rig_ml_weight = tk.DoubleVar(value=0.05)
```

### D. 다양성 페널티 강화

```python
# lotto_physics.py:2050-2070 수정
total_score = (
    remaining_weight * 0.4 * qh_score +        # 0.5 → 0.4
    remaining_weight * 0.2 * weight_score +
    remaining_weight * 0.1 * pattern_score +   # 0.2 → 0.1
    remaining_weight * 0.3 * (-diversity_penalty / 3.0) +  # 0.1 → 0.3, 페널티 강화
    ml_w * ml_score
)
```

---

## 4. 과적합 체크리스트

### ✅ 훈련 후 확인사항

```python
# 1. 훈련 정확도 vs 검증 정확도
train_acc = 96%
cv_acc = 81%
gap = 15%  # ← 과적합!

# 2. 특징 중요도
feature_importances = [0.5, 0.3, 0.1, ...]  # 한 특징이 50%면 과적합

# 3. 실제 생성 결과
45번 출현: 91.5%  # ← 과적합!
고수 비율: 62.9%  # ← 과적합!

# 4. ML 점수 분산
ML_scores std = 0.0023  # ← 너무 작으면 과적합
```

### ✅ 정상 기준

```
✓ train_acc - cv_acc < 10%
✓ 특징 중요도가 골고루 분산 (max < 20%)
✓ 45번 출현 10-20% (정상 13.3%)
✓ 저/중/고수 각 30-37% (정상 33.3%)
✓ ML 점수 std > 0.05
```

---

## 5. 최종 권장 사항

### 🥇 1순위: ML 비활성화
```python
ml_weight = 0.0
```
- 가장 확실한 방법
- Physics + QH + Pattern만으로 충분

### 🥈 2순위: ML 최소화
```python
ml_weight = 0.05
+ Regularization 강화
+ 특징 편향 제거 (✅ 완료)
```

### 🥉 3순위: 모델 단순화
```python
model_type = "logistic"  # gradient_boosting 대신
```
- 로지스틱 회귀는 과적합 위험 낮음
- 선형 모델이라 편향 덜함

---

## 6. 테스트 스크립트

```bash
# 편향 제거 효과 테스트
python test_unbiased_features.py

# 가상조작 시뮬 테스트 (50개 생성)
python test_unbiased_rigged.py

# 결과 확인
# - 45번 출현: 10-20% (정상)
# - 고수 비율: 30-37% (정상)
# - ML 점수 std > 0.05 (다양성)
```

---

## 요약

| 방법 | 효과 | 구현 난이도 | 추천도 |
|------|------|------------|--------|
| ML 비활성화 (ml_weight=0) | ⭐⭐⭐⭐⭐ | ⭐ | 🥇 최고 |
| 편향 특징 제거 | ⭐⭐⭐⭐ | ⭐⭐ | ✅ 완료 |
| Regularization 강화 | ⭐⭐⭐⭐ | ⭐⭐ | 🥈 권장 |
| ml_weight 감소 (0.3→0.05) | ⭐⭐⭐ | ⭐ | 🥈 권장 |
| 모델 단순화 (Logistic) | ⭐⭐⭐ | ⭐ | 🥉 대안 |
| 음성 샘플링 개선 | ⭐⭐ | ⭐⭐⭐ | - |
| Cross-validation 강화 | ⭐⭐ | ⭐⭐ | - |

**결론**: ML을 완전히 끄거나(ml_weight=0), 최소화(0.05)하는 것이 가장 효과적!
