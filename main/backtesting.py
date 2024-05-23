from qsbacktester import *
import pandas as pd

"""
Note: only run the following set of parameters
if isPlot = True,
it will plot the backtesting result, including the time series of return, cum return and drawdown
"""
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
metrics_name = 'BTC'
metric_type = 'price'
coin = 'BTC'
tf = '8h'
isMomentum = True
threshold = 0.2
lookback_period = 270
denoise_logic = 'vol+z'
train_size = 0.7
#
if __name__ == '__main__':
    train_kpi_dict, test_kpi_dict, full_set_dict = backtesting(coin, tf, metrics_name,
                                                               metric_type=metric_type,
                                                               isMomentum=isMomentum,
                                                               lookback_period=lookback_period,
                                                               threshold=threshold,
                                                               denoise_logic=denoise_logic,
                                                               denoise=denoise, isPlot=True,
                                                               train_size=train_size,
                                                               shift=0, unit_tc=0 #ASSUMPTION
                                                               )

    print(pd.DataFrame(
        {
            'train': train_kpi_dict,
            'test': test_kpi_dict,
            'full_set': full_set_dict
        }
    ).T)