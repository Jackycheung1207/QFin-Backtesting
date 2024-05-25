from qsbacktester import *
import pandas as pd
import time


START_DATE = '2020-06-01'
END_DATE = '2024-03-31'

exchange = 'binance'
instrument = 'perp'
price_folder_path = '../Data/Price'
metric_folder_path = f'../Data/OnChain'


metric_type = 'price'

asset_list = ['BTC']
metrics_name = 'BTC'
result_folder_path = f'result/{metrics_name}'
tf_list = ['8h']
lookback_period_list = np.arange(10, 300, 10)
isMomentum_list = [ True]
denoise_logic_list = ['vol+z']

train_kpi_df, test_kpi_df, full_kpi_df = optimization(asset_list, metrics_name,
                                                      metric_type=metric_type,
                                                      tf_list=tf_list,
                                                      lookback_period_list=lookback_period_list,
                                                      isMomentum_list=isMomentum_list,
                                                      denoise_logic_list=denoise_logic_list,
                                                      is_save_ts=True,
                                                      denoise=denoise,shift=0,unit_tc=0)
time.sleep(1)
print(full_kpi_df.head(50))
print(train_kpi_df.head(50))
print(test_kpi_df.head(50))

# full_kpi_df.to_csv(f'{result_folder_path}/performance_{denoise_logic_list}_full.csv')
# train_kpi_df.to_csv(f'{result_folder_path}/performance_{denoise_logic_list}_train.csv')
# test_kpi_df.to_csv(f'{result_folder_path}/performance_{denoise_logic_list}_test.csv')

