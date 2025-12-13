"""
엑셀 HTML 파일 2개를 lotto.csv 형식으로 변환
- excel (21).xls: 회차 600~
- excel (22).xls: 회차 601~1202
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Excel HTML → lotto.csv 변환")
print("=" * 70)

# 1. 두 파일 읽기
print("\n[1] 파일 읽기")

tables1 = pd.read_html('e:/gptlotto/excel (21).xls', encoding='euc-kr')
df1 = tables1[1]  # 두 번째 테이블
print(f"   excel (21).xls: {df1.shape[0]-2}개 회차 (헤더 제외)")

tables2 = pd.read_html('e:/gptlotto/excel (22).xls', encoding='euc-kr')
df2 = tables2[1]  # 두 번째 테이블
print(f"   excel (22).xls: {df2.shape[0]-2}개 회차 (헤더 제외)")

# 2. 데이터 정리 함수
print("\n[2] 데이터 정리")

def process_dataframe(df):
    """
    헤더 제거 및 컬럼 추출

    컬럼 매핑:
    - 1: 회차
    - 2: 날짜
    - 13-18: 당첨번호 6개
    - 19: 보너스 번호
    """
    # 헤더 2줄 제거
    df_clean = df.iloc[2:].copy()
    df_clean.columns = range(len(df_clean.columns))

    # lotto.csv 형식으로 변환
    result = pd.DataFrame({
        'round': pd.to_numeric(df_clean[1], errors='coerce').astype('Int64'),
        'date': df_clean[2].astype(str),
        'n1': pd.to_numeric(df_clean[13], errors='coerce').astype('Int64'),
        'n2': pd.to_numeric(df_clean[14], errors='coerce').astype('Int64'),
        'n3': pd.to_numeric(df_clean[15], errors='coerce').astype('Int64'),
        'n4': pd.to_numeric(df_clean[16], errors='coerce').astype('Int64'),
        'n5': pd.to_numeric(df_clean[17], errors='coerce').astype('Int64'),
        'n6': pd.to_numeric(df_clean[18], errors='coerce').astype('Int64'),
        'bonus': pd.to_numeric(df_clean[19], errors='coerce').astype('Int64'),
    })

    # 결측치 제거
    result = result.dropna()

    return result

# 각 파일 처리
df1_clean = process_dataframe(df1)
df2_clean = process_dataframe(df2)

print(f"   excel (21).xls 처리 완료: {len(df1_clean)}개 회차")
print(f"   excel (22).xls 처리 완료: {len(df2_clean)}개 회차")

# 3. 두 파일 합치기
print("\n[3] 두 파일 합치기")

# 합치기
combined = pd.concat([df1_clean, df2_clean], ignore_index=True)
print(f"   합친 데이터: {len(combined)}개 회차")

# 중복 제거 (회차 기준)
combined = combined.drop_duplicates(subset=['round'], keep='last')
print(f"   중복 제거 후: {len(combined)}개 회차")

# 회차 내림차순 정렬 (최신이 위)
combined = combined.sort_values('round', ascending=False).reset_index(drop=True)

# 4. 샘플 확인
print("\n[4] 변환 결과 확인")
print(f"\n최신 5개 회차:")
print(combined.head(5).to_string(index=False))

print(f"\n가장 오래된 5개 회차:")
print(combined.tail(5).to_string(index=False))

# 5. 기존 lotto.csv와 비교
print("\n[5] 기존 lotto.csv와 비교")
lotto_old = pd.read_csv('e:/gptlotto/lotto.csv')
print(f"   기존 lotto.csv: {len(lotto_old)}개 회차 (최신: {lotto_old.iloc[0]['round']})")
print(f"   새 데이터: {len(combined)}개 회차 (최신: {combined.iloc[0]['round']})")

# 회차 1192 비교
print("\n   회차 1192 비교:")
row_old = lotto_old[lotto_old['round'] == 1192]
row_new = combined[combined['round'] == 1192]

if not row_old.empty:
    print(f"   기존: {row_old.iloc[0].to_dict()}")
if not row_new.empty:
    print(f"   새로: {row_new.iloc[0].to_dict()}")

# 6. 저장
print("\n[6] lotto.csv 저장")

# 기존 파일 백업
import shutil
import os
if os.path.exists('e:/gptlotto/lotto.csv'):
    shutil.copy('e:/gptlotto/lotto.csv', 'e:/gptlotto/lotto_backup.csv')
    print("   기존 파일 백업 완료: lotto_backup.csv")

# 새 파일 저장
combined.to_csv('e:/gptlotto/lotto.csv', index=False)
print("   새 파일 저장 완료: lotto.csv")

file_size = os.path.getsize('e:/gptlotto/lotto.csv') / 1024
print(f"   파일 크기: {file_size:.1f} KB")

print("\n" + "=" * 70)
print("변환 완료!")
print("=" * 70)
