"""
다음 회차 정보 계산 유틸리티
"""
from datetime import datetime, timedelta
import pandas as pd


def get_next_round_info(history_df: pd.DataFrame | None) -> tuple[int | None, str | None]:
    """
    히스토리 DataFrame에서 다음 회차 번호와 예상 날짜를 계산합니다.

    Args:
        history_df: 로또 히스토리 DataFrame (round, date 컬럼 필요)

    Returns:
        (next_round, next_date_str) 튜플
        - next_round: 다음 회차 번호 (예: 1203)
        - next_date_str: 다음 추첨 예상 날짜 (예: "2025.12.20")
        - 정보가 없으면 (None, None)
    """
    if history_df is None or history_df.empty:
        return None, None

    try:
        # 가장 최근 회차 정보 추출
        latest_row = history_df.iloc[0]  # 첫 번째 행 (최신 데이터)

        # 회차 번호
        try:
            latest_round = int(latest_row.iloc[0])  # round 컬럼
        except (ValueError, IndexError):
            latest_round = None

        # 날짜
        try:
            latest_date_str = str(latest_row.iloc[1])  # date 컬럼
            latest_date = datetime.strptime(latest_date_str, "%Y.%m.%d")
        except (ValueError, IndexError):
            latest_date = None

        # 다음 회차 계산
        if latest_round is not None:
            next_round = latest_round + 1
        else:
            next_round = None

        # 다음 날짜 계산 (로또는 매주 토요일, 7일 후)
        if latest_date is not None:
            next_date = latest_date + timedelta(days=7)
            next_date_str = next_date.strftime("%Y.%m.%d")
        else:
            next_date_str = None

        return next_round, next_date_str

    except Exception as e:
        print(f"[WARN] 다음 회차 정보 계산 실패: {e}")
        return None, None


# 테스트
if __name__ == "__main__":
    # 테스트용 DataFrame 생성
    test_df = pd.DataFrame({
        'round': [1202, 1201, 1200],
        'date': ['2025.12.13', '2025.12.06', '2025.11.29'],
        'n1': [5, 7, 1],
        'n2': [12, 9, 2],
        'n3': [21, 24, 4],
        'n4': [33, 27, 16],
        'n5': [37, 35, 20],
        'n6': [40, 36, 32],
        'bonus': [7, 37, 45],
    })

    next_round, next_date = get_next_round_info(test_df)
    print(f"다음 회차: {next_round}")
    print(f"다음 날짜: {next_date}")
    # 예상 출력: 다음 회차: 1203, 다음 날짜: 2025.12.20
