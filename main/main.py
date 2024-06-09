from facade.qsbacktester import Backtester
import pandas as pd
import numpy as np

# To instantiate the backtester class, you need to provide the following parameters:
backtester = Backtester(
    metric_type='price',
    metric_name='BTC',
    coin='BTC',
    tf='8h',
    is_momentum=True,
    threshold=1.3,
    lookback_period=425,
    denoise_logic='z',
    shift=0,
    unit_tc=0.0,
    style='long&short',
    training_size=0.7
)

# To perform backtesting, you can use the following method:
backtester.backtesting(isPlot=True, train_size=0.7, is_save_ts=False)

# To optimize the hyperparamters, you can use the following method:
backtester.optimization(
    asset_list=['BTC'],
    tf_list=['8h'],
    lookback_period_list=np.arange(5,505,5),
    isMomentum_list=[True,False],
    denoise_logic_list=['z'],
    threshold_list=None,
    is_save_ts=False,
    is_plot=False,
    train_size=0.7
)

# Create an interactive heatmap, you can easily visualize is there any parameter plateau
backtester.create_heatmap_interactive(
    metrics_name='BTC',
    train_test_full='train',
    coin='BTC',
    tf='8h',
    isMomentum=True,
    denoise_logic='z',
    title=None,
    kpi='strategy_sharpe'
)

# To visualize the time series of return, cum return and drawdown, you can use the following method:
backtester.backtesting(isPlot=True, train_size=0.7, is_save_ts=False)




