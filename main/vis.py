import sys

from qsbacktester import *
import pandas as pd

"""
Note: please run optimization.py first to get the performance_full.csv, performance_train.csv and performance_test.csv
it will plot the heapmap of the backtesting result
"""
##############################################
#Const
START_DATE = '2020-06-01'
END_DATE = '2024-03-31'

exchange = 'binance'
instrument = 'perp'
price_folder_path = '../Data/Price'
metric_folder_path = f'../Data/OnChain'
##############################################
#Params
# metrics_name = 'balance_exchanges_relative'
metrics_name = 'BTC'
coin = 'BTC'
tf='8h'
# isMomentum=False
isMomentum=True
denoise_logic='vol+z'
# denoise_logic='ma_diff'
# denoise_logic='ln+z'
# denoise_logic='z'
# denoise_logic='pct+z'
# denoise_logic='ln+pct+z'
# denoise_logic='vol+z'
# denoise_logic='smac'

result_folder_path = f'result/{metrics_name}'
train_test_full = 'full'
kpi = 'strategy_sharpe'

##############################################

create_heatmap(
               metrics_name=metrics_name,
               train_test_full=train_test_full,
               coin=coin,tf=tf,
               isMomentum=isMomentum,
               denoise_logic=denoise_logic,
               kpi=kpi)
sys.exit()

heatmap_configs = [
    # {
    #     'metrics_name': 'balance_exchanges_relative',
    #     'train_test_full': 'train',
    #     'coin': 'BTC',
    #     'tf': '8h',
    #     'isMomentum': False,
    #     'denoise_logic': 'z',
    #     'kpi': 'strategy_sharpe'
    # },
    # {
    #     'metrics_name': 'balance_exchanges_relative',
    #     'train_test_full': 'test',
    #     'coin': 'BTC',
    #     'tf': '8h',
    #     'isMomentum': False,
    #     'denoise_logic': 'z',
    #     'kpi': 'strategy_sharpe'
    # },
    {
        'metrics_name': 'BTC',
        'train_test_full': 'full',
        'coin': 'BTC',
        'tf': '8h',
        'isMomentum': True,
        'denoise_logic': 'smac',
        # 'kpi': 'strategy_mdd'
        # 'kpi': 'strategy_calmar'
        'kpi': 'strategy_sharpe'
    },
    {
        'metrics_name': 'BTC',
        'train_test_full': 'full',
        'coin': 'BTC',
        'tf': '8h',
        'isMomentum': True,
        'denoise_logic': 'ma_diff',
        # 'kpi': 'strategy_mdd'
        # 'kpi': 'strategy_calmar'
        'kpi': 'strategy_sharpe'
    },
]

create_multiple_heatmaps(heatmap_configs)