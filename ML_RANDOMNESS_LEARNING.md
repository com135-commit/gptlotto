# ML로 로또 번호의 랜덤성 학습하기

## 핵심 아이디어

**현재 문제점:**
- ML이 "당첨 번호 vs 비당첨 번호" 구분을 학습 → 불가능 (로또는 무작위)
- 양성 샘플(당첨 번호)과 음성 샘플(랜덤)이 본질적으로 동일
- ML이 노이즈를 패턴으로 착각

**새로운 접근:**
- ML이 "진짜 로또 번호의 랜덤 분포"를 학습
- 생성된 번호가 "얼마나 로또스러운가" 평가
- 과거 당첨 번호의 통계적 특성만 모방

---

## 1. 랜덤성 학습 방법

### A. One-Class Classification (이상 탐지)

#### 개념
```python
# 양성/음성 구분 대신, "정상" 분포만 학습
양성 샘플: 과거 당첨 번호 (200개)
음성 샘플: 없음!

# 목표: 과거 당첨 번호와 유사한 통계적 특성을 가진 번호 생성
```

#### 구현
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Isolation Forest: 이상치 탐지
model = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # 10%를 이상치로 간주
    random_state=42
)

# 과거 당첨 번호만 학습
X_positive = [_set_features(s) for s in past_winning_numbers]
model.fit(X_positive)

# 예측: 1 = 정상 (로또스러움), -1 = 이상 (비정상)
score = model.decision_function([candidate_features])
# score가 높을수록 과거 당첨 번호와 유사
```

#### 장점
- 양성/음성 구분 불필요
- "로또스러움" 정도만 평가
- 과적합 위험 낮음

---

### B. Distribution Matching (분포 매칭)

#### 개념
```python
# 과거 당첨 번호의 통계적 분포 학습
분포 특성:
- 번호 범위 분포 (저/중/고)
- 짝수/홀수 비율
- 연속 번호 빈도
- 간격 분포
- 끝자리 다양성

# 생성된 번호가 이 분포를 얼마나 잘 따르는지 평가
```

#### 구현
```python
def learn_lottery_distribution(past_numbers):
    """과거 당첨 번호의 분포 특성 학습"""
    stats = {
        'low_ratio': [],      # 저수 비율
        'mid_ratio': [],      # 중수 비율
        'high_ratio': [],     # 고수 비율
        'even_ratio': [],     # 짝수 비율
        'consecutive': [],    # 연속 번호 개수
        'gap_mean': [],       # 간격 평균
        'gap_std': [],        # 간격 표준편차
    }

    for nums in past_numbers:
        stats['low_ratio'].append(sum(1 for n in nums if 1<=n<=15) / 6)
        stats['mid_ratio'].append(sum(1 for n in nums if 16<=n<=30) / 6)
        stats['high_ratio'].append(sum(1 for n in nums if 31<=n<=45) / 6)
        stats['even_ratio'].append(sum(1 for n in nums if n%2==0) / 6)
        # ... 기타 통계

    # 각 특성의 평균과 표준편차 계산
    distribution = {
        key: {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        for key, values in stats.items()
    }

    return distribution

def score_by_distribution(candidate, distribution):
    """생성된 번호가 분포를 얼마나 잘 따르는지 평가"""
    # 각 특성 계산
    low_ratio = sum(1 for n in candidate if 1<=n<=15) / 6
    mid_ratio = sum(1 for n in candidate if 16<=n<=30) / 6
    high_ratio = sum(1 for n in candidate if 31<=n<=45) / 6
    even_ratio = sum(1 for n in candidate if n%2==0) / 6

    # 분포와의 거리 계산 (Gaussian likelihood)
    score = 0
    for ratio, dist_key in [
        (low_ratio, 'low_ratio'),
        (mid_ratio, 'mid_ratio'),
        (high_ratio, 'high_ratio'),
        (even_ratio, 'even_ratio')
    ]:
        mean = distribution[dist_key]['mean']
        std = distribution[dist_key]['std']

        # Gaussian probability
        likelihood = np.exp(-0.5 * ((ratio - mean) / std) ** 2)
        score += likelihood

    return score / 4  # 평균
```

#### 장점
- 해석 가능 (어떤 특성이 비정상인지 알 수 있음)
- 편향 없음 (분포만 따름)
- 과적합 없음 (통계량만 사용)

---

### C. Generative Adversarial Network (GAN)

#### 개념
```python
# Generator: 로또 번호 생성
# Discriminator: 진짜 로또 번호 vs 가짜 번호 구분

# 학습 과정:
1. Generator가 랜덤 번호 생성
2. Discriminator가 진짜/가짜 판별
3. Generator는 Discriminator를 속이려고 학습
4. 결과: 진짜 로또 번호와 구분 불가능한 번호 생성
```

#### 구현 (간단 버전)
```python
# Generator: 노이즈 → 로또 번호
generator = Sequential([
    Dense(128, activation='relu', input_dim=100),
    Dense(64, activation='relu'),
    Dense(45, activation='sigmoid')  # 45개 번호 확률
])

# Discriminator: 로또 번호 → 진짜/가짜
discriminator = Sequential([
    Dense(64, activation='relu', input_dim=20),  # 특징 벡터
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # 진짜=1, 가짜=0
])

# 학습
for epoch in range(1000):
    # Real samples
    real_numbers = sample_past_winning_numbers()
    real_features = [_set_features(n) for n in real_numbers]

    # Fake samples
    noise = np.random.randn(batch_size, 100)
    fake_numbers = generator.predict(noise)
    fake_features = [_set_features(n) for n in fake_numbers]

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_features, np.ones(...))
    d_loss_fake = discriminator.train_on_batch(fake_features, np.zeros(...))

    # Train generator (fool discriminator)
    g_loss = combined_model.train_on_batch(noise, np.ones(...))
```

#### 장점
- 강력한 생성 능력
- 복잡한 분포 학습 가능

#### 단점
- 학습 불안정
- 모드 붕괴 (같은 번호만 생성)
- 오버킬 (로또는 단순함)

---

## 2. 추천 방법: Hybrid Distribution Scoring

### 핵심 아이디어
```
ML 모델 = 분포 매칭 + 경량 Anomaly Detection
```

### 구현

```python
class LotteryRandomnessScorer:
    """로또 번호의 랜덤성 평가 (편향 없음)"""

    def __init__(self, past_numbers):
        self.past_numbers = past_numbers

        # 1. 분포 특성 학습
        self.distribution = self._learn_distribution()

        # 2. Isolation Forest (가벼운 이상 탐지)
        X = np.array([self._extract_features(nums) for nums in past_numbers])
        self.anomaly_detector = IsolationForest(
            n_estimators=50,
            contamination=0.05,
            random_state=42
        )
        self.anomaly_detector.fit(X)

    def _learn_distribution(self):
        """과거 당첨 번호의 통계적 분포"""
        stats = {
            'low': [], 'mid': [], 'high': [],
            'even': [], 'odd': [],
            'consecutive': [],
            'gap_mean': [], 'gap_std': [],
            'range': [], 'std': []
        }

        for nums in self.past_numbers:
            sorted_nums = sorted(nums)

            stats['low'].append(sum(1 for n in nums if 1<=n<=15))
            stats['mid'].append(sum(1 for n in nums if 16<=n<=30))
            stats['high'].append(sum(1 for n in nums if 31<=n<=45))
            stats['even'].append(sum(1 for n in nums if n%2==0))
            stats['odd'].append(sum(1 for n in nums if n%2==1))

            # 연속 번호
            consecutive = sum(1 for i in range(5) if sorted_nums[i+1]-sorted_nums[i]==1)
            stats['consecutive'].append(consecutive)

            # 간격
            gaps = np.diff(sorted_nums)
            stats['gap_mean'].append(gaps.mean())
            stats['gap_std'].append(gaps.std())

            # 범위와 분산
            stats['range'].append(sorted_nums[-1] - sorted_nums[0])
            stats['std'].append(np.std(sorted_nums))

        # 각 특성의 평균/표준편차
        return {
            key: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for key, values in stats.items()
        }

    def _extract_features(self, nums):
        """특징 벡터 추출 (편향 없는 중립적 특징)"""
        sorted_nums = sorted(nums)

        # 분포 특징
        low = sum(1 for n in nums if 1<=n<=15)
        mid = sum(1 for n in nums if 16<=n<=30)
        high = sum(1 for n in nums if 31<=n<=45)
        even = sum(1 for n in nums if n%2==0)

        # 통계 특징
        gaps = np.diff(sorted_nums)
        consecutive = sum(1 for i in range(5) if sorted_nums[i+1]-sorted_nums[i]==1)

        return np.array([
            low/6, mid/6, high/6,
            even/6,
            consecutive/5,
            gaps.mean()/10,
            gaps.std()/5,
            (sorted_nums[-1] - sorted_nums[0])/45,
            np.std(sorted_nums)/15
        ])

    def score(self, candidate):
        """
        후보 번호의 "로또스러움" 평가

        Returns:
            float: 0~1 사이 점수 (1 = 매우 로또스러움)
        """
        # 1. 분포 점수 (50%)
        dist_score = self._distribution_score(candidate)

        # 2. 이상 탐지 점수 (50%)
        features = self._extract_features(candidate)
        anomaly_score = self.anomaly_detector.score_samples([features])[0]
        # -1 ~ 0 사이 값 → 0 ~ 1로 정규화
        anomaly_score = (anomaly_score + 1) / 2

        # 최종 점수
        return 0.5 * dist_score + 0.5 * anomaly_score

    def _distribution_score(self, candidate):
        """분포 매칭 점수"""
        sorted_nums = sorted(candidate)

        # 각 특성 계산
        features = {
            'low': sum(1 for n in candidate if 1<=n<=15),
            'mid': sum(1 for n in candidate if 16<=n<=30),
            'high': sum(1 for n in candidate if 31<=n<=45),
            'even': sum(1 for n in candidate if n%2==0),
            'consecutive': sum(1 for i in range(5) if sorted_nums[i+1]-sorted_nums[i]==1),
            'gap_mean': np.diff(sorted_nums).mean(),
            'gap_std': np.diff(sorted_nums).std(),
            'range': sorted_nums[-1] - sorted_nums[0],
            'std': np.std(sorted_nums)
        }

        # Gaussian likelihood
        scores = []
        for key, value in features.items():
            if key in self.distribution:
                mean = self.distribution[key]['mean']
                std = self.distribution[key]['std']

                if std > 0:
                    # Gaussian probability
                    likelihood = np.exp(-0.5 * ((value - mean) / std) ** 2)
                    scores.append(likelihood)

        return np.mean(scores) if scores else 0.5
```

### 사용 예시

```python
# 1. 학습
past_numbers = [[1,2,3,4,5,6], [7,8,9,10,11,12], ...]  # 과거 200회
scorer = LotteryRandomnessScorer(past_numbers)

# 2. 평가
candidate1 = [1, 2, 3, 4, 5, 6]     # 연속 번호 - 비정상
candidate2 = [3, 14, 21, 28, 35, 42] # 균형 잡힌 번호 - 정상
candidate3 = [41, 42, 43, 44, 45, 1] # 고수 편중 - 비정상

score1 = scorer.score(candidate1)  # 낮은 점수
score2 = scorer.score(candidate2)  # 높은 점수
score3 = scorer.score(candidate3)  # 낮은 점수
```

---

## 3. 현재 시스템에 통합

### 수정 방안

```python
# lotto_generators.py

def train_randomness_scorer(history_df, **kwargs):
    """
    기존 train_ml_scorer 대체
    로또의 랜덤성을 학습하는 모델
    """
    # 과거 당첨 번호 추출
    past_numbers = []
    for row in history_df.itertuples(index=False):
        nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
        if len(nums) == 6:
            past_numbers.append(nums)

    # LotteryRandomnessScorer 학습
    scorer = LotteryRandomnessScorer(past_numbers)

    return {
        "type": "randomness_scorer",
        "scorer": scorer,
        "n_samples": len(past_numbers)
    }

def ml_score_set(nums, model, **kwargs):
    """
    기존 ml_score_set 수정
    """
    if model.get("type") == "randomness_scorer":
        scorer = model["scorer"]
        return scorer.score(nums)

    # 기존 로직 (sklearn 모델)
    else:
        # ... 기존 코드
```

### GUI 수정

```python
# lotto_main.py

def on_train_ml(self):
    """ML 학습 버튼 클릭"""
    try:
        # 기존: train_ml_scorer(..., model_type='gradient_boosting')
        # 신규: train_randomness_scorer(...)

        self.ml_model = train_randomness_scorer(
            history_df=self.history_df,
            max_rounds=200
        )

        print("[ML 학습 완료] 랜덤성 스코어러")
        print(f"  학습 샘플: {self.ml_model['n_samples']}개")

    except Exception as e:
        print(f"[오류] {e}")
```

---

## 4. 장단점 비교

### 현재 방식 (Classification)
```
목표: 당첨 번호 vs 비당첨 번호 구분
문제: 로또는 무작위라 구분 불가능
결과: 과적합, 편향 발생
```

### 새로운 방식 (Randomness Learning)
```
목표: 로또 번호의 통계적 특성 학습
방법: 분포 매칭 + 이상 탐지
결과: 편향 없음, 과적합 없음
```

### 비교표

| 항목 | Classification | Randomness Learning |
|------|----------------|---------------------|
| 학습 데이터 | 양성 + 음성 | 양성만 |
| 목표 | 구분 | 유사성 |
| 과적합 위험 | 높음 | 낮음 |
| 편향 | 발생 | 없음 |
| 해석성 | 낮음 | 높음 |
| 다양성 | 낮음 | 높음 |

---

## 5. 추천 방안

### 🥇 최선: Hybrid Distribution Scorer
```python
model_type = "randomness_scorer"
```
- 분포 매칭 + Isolation Forest
- 편향 없음, 과적합 없음
- 해석 가능

### 🥈 차선: ML 완전 비활성화
```python
ml_weight = 0.0
```
- Physics + QH + Pattern만 사용
- 가장 확실한 방법

### 🥉 현재: Classification + 강력한 Regularization
```python
model_type = "gradient_boosting"
ml_weight = 0.05
```
- 이미 적용한 방법
- 과적합 최소화했지만 여전히 위험 존재

---

## 6. 구현 우선순위

1. **즉시 적용**: ml_weight = 0 (ML 비활성화)
2. **단기**: LotteryRandomnessScorer 구현 및 테스트
3. **중기**: GUI에서 모델 타입 선택 옵션 추가
   - Classification (기존)
   - Randomness Scorer (신규)
4. **장기**: GAN 기반 생성 모델 실험

---

## 결론

**ML이 로또의 랜덤성을 학습하도록 하려면:**
1. ✅ **분류 문제가 아닌 분포 학습 문제로 접근**
2. ✅ **One-Class Learning 사용 (양성만 학습)**
3. ✅ **통계적 특성만 모방 (편향 제거)**
4. ✅ **이상 탐지로 비정상 번호 제외**

이 방식이 현재의 Classification 방식보다 훨씬 적합합니다!
