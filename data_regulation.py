import pandas as pd
from utils import open_yaml

config = open_yaml('config.yaml')

fname = 'Smoothed_CyberTrend_Forecasting_All'
df = pd.read_csv(f'data/{fname}.csv', index_col=0)

columns = config['attack_columns'] + config['solution_columns']
df = df[columns]
df.to_csv(f'data/{fname}.txt', sep="\t", index=False, header=None)
