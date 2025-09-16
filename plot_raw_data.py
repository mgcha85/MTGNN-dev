import pandas as pd
import matplotlib.pyplot as plt
import os
from config import NET_ROOT
import re

# 파일 경로
input_file = f"data/Norm_CyberTrend_Forecasting_All.csv"
output_dir = f"plots"
# input_file = f"{NET_ROOT}/data/sm_data.csv"
# output_dir = f"{NET_ROOT}/plots"

os.makedirs(output_dir, exist_ok=True)

def group_features_by_country(columns):
    """
    columns: list of strings, e.g. ["GDP-KR", "GDP-US", "Inflation-KR"]

    return: dict { feature: [countries...] }
    """
    feature_map = {}
    for col in columns:
        parts = col.split('-')
        if len(parts) < 2:
            continue  # 국가 정보가 없는 경우 스킵

        feature = "-".join(parts[:-1])   # 마지막 element 제외
        country = parts[-1]             # 마지막 element = 국가

        if feature not in feature_map:
            feature_map[feature] = []
        feature_map[feature].append(country)

    return feature_map



# CSV 불러오기 (row=시간, col=feature)
df = pd.read_csv(input_file,
                 parse_dates=[0],   # 첫 번째 컬럼을 날짜로 변환
                 date_format=lambda x: pd.to_datetime(x, format="%b-%y"))
df.set_index('Date', inplace=True)

feature_map = group_features_by_country(df.columns)
print(list(feature_map.keys()))
columns = [x for x in df.columns if '-ALL' in x]
df = df[columns]

# 시간축 (row index)
time = range(len(df))

# 각 feature별 시계열 그래프 저장
for col in df.columns:
    plt.figure(figsize=(14,6))
    plt.plot(time, df[col], label=f"Feature {col}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Time Series of Feature {col}")
    plt.legend()
    plt.grid(True)
    
    safe_col = re.sub(r'[^A-Za-z0-9_\-]', '_', str(col))
    save_path = os.path.join(output_dir, f"feature_{safe_col}.png")

    plt.savefig(save_path, dpi=200)
    plt.close()

print(f"✅ 저장 완료: {output_dir}")
