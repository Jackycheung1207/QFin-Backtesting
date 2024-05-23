from qsbacktester import *
import pandas as pd
import time
################################################################
#Const
START_DATE = '2020-06-01'
END_DATE = '2024-03-31'

exchange = 'binance'
instrument = 'perp'
price_folder_path = '../Data/Price'
metric_folder_path = f'../Data/OnChain'
################################################################
#Params
metric_type = 'price'
asset_list = ['BTC', 'ETH']
metrics_name = 'BTC'
result_folder_path = f'result/{metrics_name}'
tf_list = ['1h', '4h', '8h'][::-1]
lookback_period_list = np.arange(5, 505, 5)
isMomentum_list = [True, False]


# denoise_logic_list = ['vol+z']
# denoise_logic_list = ['z']
# denoise_logic_list = ['smac']
denoise_logic_list = ['ma_diff']
denoise_logic=denoise_logic_list[0] #TODO:

if __name__ == '__main__':
    train_kpi_df, test_kpi_df, full_kpi_df = optimization(asset_list, metrics_name,
                                                          metric_type=metric_type,
                                                          tf_list=tf_list,
                                                          lookback_period_list=lookback_period_list,
                                                          isMomentum_list=isMomentum_list,
                                                          denoise_logic_list=denoise_logic_list,
                                                          denoise=denoise,
                                                          shift=0,unit_tc=0 #ASSUMPTION
                                                          )
    time.sleep(1)
    print(full_kpi_df.head(50))
    print(train_kpi_df.head(50))
    print(test_kpi_df.head(50))

    full_kpi_df.to_csv(f'{result_folder_path}/{denoise_logic}/performance_{denoise_logic}_full.csv')
    train_kpi_df.to_csv(f'{result_folder_path}/{denoise_logic}/performance_{denoise_logic}_train.csv')
    test_kpi_df.to_csv(f'{result_folder_path}/{denoise_logic}/performance_{denoise_logic}_test.csv')


