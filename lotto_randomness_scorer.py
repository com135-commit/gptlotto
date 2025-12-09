"""
로또 번호 랜덤성 평가 모델
- One-Class Learning: 과거 당첨 번호의 통계적 특성만 학습
- 분류 대신 분포 매칭: "얼마나 로또스러운가" 평가
- 편향 없음: 특정 번호를 선호하지 않음
"""
import numpy as np
from sklearn.ensemble import IsolationForest


class LotteryRandomnessScorer:
    """
    로또 번호의 랜덤성 평가 (One-Class Learning)

    기존 Classification 방식:
    - 당첨 번호 vs 비당첨 번호 구분 → 불가능 (로또는 무작위)
    - 과적합 발생, 편향 발생

    새로운 Randomness Learning 방식:
    - 과거 당첨 번호의 통계적 분포만 학습
    - 생성된 번호가 얼마나 로또스러운지 평가
    - 편향 없음, 과적합 없음
    """

    def __init__(self, past_numbers, contamination=0.05):
        """
        Args:
            past_numbers: 과거 당첨 번호 리스트 [[1,2,3,4,5,6], [7,8,9,...], ...]
            contamination: Isolation Forest의 이상치 비율 (기본 5%)
        """
        self.past_numbers = past_numbers

        # 1. 분포 특성 학습
        self.distribution = self._learn_distribution()

        # 2. Isolation Forest (가벼운 이상 탐지)
        X = np.array([self._extract_features(nums) for nums in past_numbers])
        self.anomaly_detector = IsolationForest(
            n_estimators=50,
            contamination=contamination,
            random_state=42
        )
        self.anomaly_detector.fit(X)

        print(f"[랜덤성 스코어러] 학습 완료")
        print(f"  샘플 수: {len(past_numbers)}개")
        print(f"  특징 수: {X.shape[1]}개")

    def _learn_distribution(self):
        """과거 당첨 번호의 통계적 분포 학습"""
        stats = {
            'low': [],       # 저수(1-15) 개수
            'mid': [],       # 중수(16-30) 개수
            'high': [],      # 고수(31-45) 개수
            'even': [],      # 짝수 개수
            'odd': [],       # 홀수 개수
            'consecutive': [],  # 연속 번호 개수
            'gap_mean': [],     # 간격 평균
            'gap_std': [],      # 간격 표준편차
            'range': [],        # 범위 (max - min)
            'std': []           # 표준편차
        }

        for nums in self.past_numbers:
            sorted_nums = sorted(nums)

            # 번호 범위 분포
            stats['low'].append(sum(1 for n in nums if 1 <= n <= 15))
            stats['mid'].append(sum(1 for n in nums if 16 <= n <= 30))
            stats['high'].append(sum(1 for n in nums if 31 <= n <= 45))

            # 짝수/홀수
            stats['even'].append(sum(1 for n in nums if n % 2 == 0))
            stats['odd'].append(sum(1 for n in nums if n % 2 == 1))

            # 연속 번호
            consecutive = sum(1 for i in range(5) if sorted_nums[i + 1] - sorted_nums[i] == 1)
            stats['consecutive'].append(consecutive)

            # 간격
            gaps = np.diff(sorted_nums)
            stats['gap_mean'].append(gaps.mean())
            stats['gap_std'].append(gaps.std())

            # 범위와 분산
            stats['range'].append(sorted_nums[-1] - sorted_nums[0])
            stats['std'].append(np.std(sorted_nums))

        # 각 특성의 평균과 표준편차 계산
        distribution = {}
        for key, values in stats.items():
            distribution[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return distribution

    def _extract_features(self, nums):
        """
        특징 벡터 추출 (편향 없는 중립적 특징만)

        주의: 평균, 최대값, 최소값 등 특정 번호를 선호하는 특징 제외
        """
        sorted_nums = sorted(nums)

        # 분포 특징 (비율로 정규화)
        low = sum(1 for n in nums if 1 <= n <= 15) / 6.0
        mid = sum(1 for n in nums if 16 <= n <= 30) / 6.0
        high = sum(1 for n in nums if 31 <= n <= 45) / 6.0
        even = sum(1 for n in nums if n % 2 == 0) / 6.0

        # 패턴 특징
        consecutive = sum(1 for i in range(5) if sorted_nums[i + 1] - sorted_nums[i] == 1) / 5.0

        # 간격 특징 (정규화)
        gaps = np.diff(sorted_nums)
        gap_mean = gaps.mean() / 10.0
        gap_std = gaps.std() / 5.0

        # 통계 특징 (정규화)
        range_val = (sorted_nums[-1] - sorted_nums[0]) / 45.0
        std_val = np.std(sorted_nums) / 15.0

        return np.array([
            low, mid, high,
            even,
            consecutive,
            gap_mean,
            gap_std,
            range_val,
            std_val
        ])

    def score(self, candidate):
        """
        후보 번호의 "로또스러움" 평가

        Args:
            candidate: 평가할 번호 [1, 14, 27, 34, 38, 45]

        Returns:
            float: 0~1 사이 점수 (1 = 매우 로또스러움, 0 = 비정상)
        """
        # 1. 분포 점수 (50%)
        dist_score = self._distribution_score(candidate)

        # 2. 이상 탐지 점수 (50%)
        features = self._extract_features(candidate)
        anomaly_score = self.anomaly_detector.score_samples([features])[0]

        # Isolation Forest 점수: -1 ~ 0 사이 (0에 가까울수록 정상)
        # 0 ~ 1로 정규화 (1에 가까울수록 정상)
        anomaly_score_normalized = (anomaly_score + 1.0) / 2.0

        # 최종 점수 (가중 평균)
        final_score = 0.5 * dist_score + 0.5 * anomaly_score_normalized

        return float(final_score)

    def _distribution_score(self, candidate):
        """
        분포 매칭 점수
        과거 당첨 번호의 통계적 분포와 얼마나 유사한가?
        """
        sorted_nums = sorted(candidate)

        # 각 특성 계산
        features = {
            'low': sum(1 for n in candidate if 1 <= n <= 15),
            'mid': sum(1 for n in candidate if 16 <= n <= 30),
            'high': sum(1 for n in candidate if 31 <= n <= 45),
            'even': sum(1 for n in candidate if n % 2 == 0),
            'consecutive': sum(1 for i in range(5) if sorted_nums[i + 1] - sorted_nums[i] == 1),
            'gap_mean': np.diff(sorted_nums).mean(),
            'gap_std': np.diff(sorted_nums).std(),
            'range': sorted_nums[-1] - sorted_nums[0],
            'std': np.std(sorted_nums)
        }

        # Gaussian likelihood 계산
        scores = []
        for key, value in features.items():
            if key in self.distribution:
                mean = self.distribution[key]['mean']
                std = self.distribution[key]['std']

                if std > 0:
                    # Gaussian probability density
                    likelihood = np.exp(-0.5 * ((value - mean) / std) ** 2)
                    scores.append(likelihood)

        return np.mean(scores) if scores else 0.5

    def get_distribution_summary(self):
        """학습한 분포 특성 요약"""
        print("\n학습한 로또 번호 분포 특성:")
        print("=" * 60)

        for key, stats in self.distribution.items():
            print(f"{key:12s}: 평균 {stats['mean']:5.2f}, "
                  f"표준편차 {stats['std']:5.2f}, "
                  f"범위 [{stats['min']:5.2f}, {stats['max']:5.2f}]")

        print("=" * 60)


def train_randomness_scorer(history_df, max_rounds=None, **kwargs):
    """
    로또 랜덤성 스코어러 학습 (train_ml_scorer 대체)

    Args:
        history_df: 과거 당첨 번호 데이터프레임
        max_rounds: 사용할 최근 회차 수 (None=전체)

    Returns:
        dict: 모델 딕셔너리
    """
    # 과거 당첨 번호 추출
    if max_rounds is None or max_rounds <= 0:
        df = history_df
    else:
        df = history_df.tail(max_rounds)

    past_numbers = []
    for row in df.itertuples(index=False):
        nums = sorted({int(v) for v in row if 1 <= int(v) <= 45})
        if len(nums) == 6:
            past_numbers.append(nums)

    if not past_numbers:
        raise ValueError("유효한 당첨 번호가 없습니다.")

    # LotteryRandomnessScorer 학습
    scorer = LotteryRandomnessScorer(past_numbers)

    # 분포 요약 출력
    scorer.get_distribution_summary()

    return {
        "type": "randomness_scorer",
        "scorer": scorer,
        "n_samples": len(past_numbers),
        "n_features": 9,
    }


def score_randomness(nums, model, **kwargs):
    """
    랜덤성 점수 계산 (ml_score_set 대체)

    Args:
        nums: 평가할 번호 [1, 14, 27, 34, 38, 45]
        model: train_randomness_scorer로 학습한 모델

    Returns:
        float: 0~1 사이 점수
    """
    if model.get("type") != "randomness_scorer":
        raise ValueError("randomness_scorer 모델이 아닙니다.")

    scorer = model["scorer"]
    return scorer.score(nums)


if __name__ == "__main__":
    # 테스트
    print("랜덤성 스코어러 테스트")
    print("=" * 60)

    # 더미 데이터 (실제로는 lotto.csv 사용)
    past_numbers = [
        [3, 12, 23, 28, 35, 42],
        [5, 14, 21, 29, 36, 41],
        [1, 15, 22, 30, 37, 43],
        [7, 16, 24, 31, 38, 44],
        [2, 13, 20, 27, 34, 40],
    ] * 40  # 200개

    # 학습
    scorer = LotteryRandomnessScorer(past_numbers)

    # 테스트
    test_cases = [
        ([1, 2, 3, 4, 5, 6], "연속 번호 - 비정상"),
        ([3, 14, 23, 28, 35, 42], "균형 잡힌 번호 - 정상"),
        ([41, 42, 43, 44, 45, 1], "고수 편중 - 비정상"),
        ([1, 3, 5, 7, 9, 11], "홀수만 - 비정상"),
        ([2, 4, 6, 8, 10, 12], "짝수만 - 비정상"),
        ([5, 14, 21, 28, 35, 42], "균형 2 - 정상"),
    ]

    print("\n평가 결과:")
    print("-" * 60)
    for nums, desc in test_cases:
        score = scorer.score(nums)
        print(f"{str(nums):30s} | {score:.4f} | {desc}")

    print("=" * 60)
