import sys
from typing import Optional,List
from math import sqrt
import pandas as pd
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import statsmodels.api as sm

def compute_sharpe_ratio(return_series: pd.Series, annualized_factor: int=365) -> Optional[float]:
    mean_return: float = return_series.mean()
    return_std: float = return_series.std()
    if return_std == 0:
        logger.trace('Return std is 0!')
        return
    sharpe: float = mean_return / return_std * sqrt(annualized_factor)
    return sharpe

def compute_maximum_drawdown(return_series: pd.Series) -> float:
    cumul_return_series: pd.Series = return_series.cumsum()
    mdd: float = -(cumul_return_series-cumul_return_series.expanding().max()).min()
    return mdd

def compute_annualized_return(return_series: pd.Series, annualized_factor: int=365) -> float:
    num_day: int = return_series.size
    num_year: float = num_day / annualized_factor
    cumul_return_series: pd.Series = return_series.cumsum()
    cumul_return: float = cumul_return_series.iloc[-1]
    annualized_return: float = cumul_return / num_year
    return annualized_return

def compute_calmar_ratio(return_series: pd.Series, annualized_factor: int=365) -> Optional[float]:
    annualized_return: float = compute_annualized_return(return_series, annualized_factor)
    mdd: float = compute_maximum_drawdown(return_series)
    if mdd == 0:
        logger.trace('MDD is 0!')
        return
    calmar_ratio: float = annualized_return / mdd
    return calmar_ratio

def compute_max_recovery_time(return_series: pd.Series) -> int:
    cumul_return_series: pd.Series = return_series.cumsum()
    drawdown_series: pd.Series = cumul_return_series - cumul_return_series.expanding().max()
    drawdown_signal_series: pd.Series = np.where(drawdown_series < 0, 1, 0)

    duration_time_list: List[int] = []
    duration_time: int = 0
    for signal in drawdown_signal_series:
        if signal == 0:
            duration_time = 0
            duration_time_list.append(duration_time)
        if signal == 1:
            duration_time = duration_time + 1
            duration_time_list.append(duration_time)
    max_recovery_time: int = max(duration_time_list) + 1
    return max_recovery_time

def compute_expected_return(df:pd.DataFrame,style:str='long&short'):
    if style == 'long&short':
        suffix = ''
    elif style == 'long_only':
        suffix = '_long_only'
    elif style == 'short_only':
        suffix = '_short_only'

    df['trades']=df['position'+suffix] != df['position'+suffix].shift(1)

    df=df[df.trades]
    trades=len(df)
    df['p']=df['cumulative_pnl'+suffix] - df['cumulative_pnl'+suffix].shift(1)
    df['wl']=df.p >0 #bool

    win_rate=(df.wl.value_counts().tolist()[0]/trades)*100
    win_rate=round(win_rate,2)
    lose_rate=1-win_rate
    lose_rate=round(lose_rate,2)

    win_pnl=df[df.wl].p.mean()*100
    lose_pnl=df[df.wl < 1].p.mean()*100
    win_pnl=round(win_pnl,2)
    lose_pnl=round(lose_pnl,2)

    expected_return=round((win_rate*win_pnl + lose_rate * lose_pnl)/100,2)

    # print(f"""
    # Expected return={expected_return}%
    # win_rate={(win_rate)}
    # lose_rate={lose_rate}
    # win_pnl={win_pnl}
    # lose_pnl={lose_pnl}
    # """)
    return expected_return,win_rate,win_pnl,lose_pnl

def compute_long_dominance_by_duration(df:pd.DataFrame,style:str='long&short'):
    if style == 'long&short':
        suffix = ''
    elif style == 'long_only':
        suffix = '_long_only'
    elif style == 'short_only':
        suffix = '_short_only'

    list=df['position'+suffix].value_counts().tolist()
    length=len(list)
    if length==2:
        [l,s]=list
    elif length==3:
        [l,z ,s] = list
    else:
        l=s=np.nan


    # print(l,s)
    # print(round((l/(l+s)),4))
    long_dom=round((l/(l+s)),4)
    return long_dom

def compute_long_dominance_by_signal(df, style:str='long&short'):
    if style == 'long&short':
        suffix = ''
    elif style == 'long_only':
        suffix = '_long_only'
    elif style == 'short_only':
        suffix = '_short_only'

    list=df[df['position'+suffix] != df['position'+suffix].shift(1)]['position'+suffix].value_counts().tolist()
    length=len(list)
    if length==2:
        [l, s] = list
    elif length==3:
        [l, z,s] = list
    else:
        l=s=np.nan

    # print(l,s)
    # print(round((l/(l+s)),4))
    long_dom=round((l/(l+s)),4)
    return long_dom

def compute_holding_period(df:pd.DataFrame , style:str='long&short'):
    if style == 'long&short':
        suffix = ''
    elif style == 'long_only':
        suffix = '_long_only'
    elif style == 'short_only':
        suffix = '_short_only'

    empty_period = len(df[df['position'+suffix]==0])
    total_duration = len(df)
    # print(1-(empty_period/total_duration))
    hp=1 - (empty_period / total_duration)
    return round(hp,4)


def perform_t_test(strategy_returns, benchmark_returns, alpha_value=0.01):
    # Remove missing values from strategy_returns and benchmark_returns
    valid_mask = ~(np.isnan(strategy_returns) | np.isnan(benchmark_returns))
    strategy_returns = strategy_returns[valid_mask]
    benchmark_returns = benchmark_returns[valid_mask]

    # Check if there are any remaining missing or infinite values
    if np.isnan(strategy_returns).any() or np.isnan(benchmark_returns).any() or \
            np.isinf(strategy_returns).any() or np.isinf(benchmark_returns).any():
        return np.nan, np.nan

    # Calculate the differences between strategy_returns and benchmark_returns
    return_differences = strategy_returns - benchmark_returns

    # Perform one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(return_differences, 0)

    # Check if the p-value is less than the significance level
    if p_value < alpha_value:
        # If significant, return the t-statistic and p-value
        return t_statistic
    else:
        # If not significant, return np.nan for both t-statistic and p-value
        return np.nan

def calculate_alpha_beta(strategy_returns, benchmark_returns, alpha_value=0.05,annualized_factor:int=365):
    # Remove missing values from strategy_returns and benchmark_returns
    valid_mask = ~(np.isnan(strategy_returns) | np.isnan(benchmark_returns))
    strategy_returns = strategy_returns[valid_mask]
    benchmark_returns = benchmark_returns[valid_mask]

    # Check if there are any remaining missing or infinite values
    if np.isnan(strategy_returns).any() or np.isnan(benchmark_returns).any() or \
       np.isinf(strategy_returns).any() or np.isinf(benchmark_returns).any():
        return np.nan, np.nan

    # Add a constant column to the independent variable (benchmark returns)
    X = sm.add_constant(benchmark_returns)

    # Fit the linear regression model
    model = sm.OLS(strategy_returns, X).fit()

    # Extract the intercept (alpha) and coefficient (beta)
    alpha, beta = model.params.iloc[0], model.params.iloc[1]

    # Calculate the standard errors of alpha and beta
    alpha_se, beta_se = model.bse.iloc[0], model.bse.iloc[1]

    try:
        # Calculate the t-values of alpha and beta
        alpha_tval, beta_tval = alpha / alpha_se, beta / beta_se
    except Exception as E:
        print(E)
        logger.info(f"alpha={alpha}, beta={beta}")
        logger.info(f'alpha_se={alpha_se}, beta_se={beta_se}')

    # Check if alpha and beta are statistically significant
    alpha_sig = abs(alpha_tval) > stats.t.ppf(1 - alpha_value / 2, len(strategy_returns) - 2)
    beta_sig = abs(beta_tval) > stats.t.ppf(1 - alpha_value / 2, len(strategy_returns) - 2)

    # Return alpha and beta only if they are statistically significant, otherwise return np.nan
    if alpha_sig and beta_sig:
        annulized_alpha = (1+alpha) ** annualized_factor - 1
        annulized_beta = (1+beta) ** annualized_factor - 1
        return alpha, beta
    else:
        return np.nan, np.nan
######################Get info##########################

def get_var(upper,lower,unit_tc,tf,momentum,backtester):
    opt_result_dict = backtester.backtest_breakout_signal(
        upper, lower, unit_tc, is_long_exceeding_upper=momentum, cache_df=True, timeframe=tf)
    print(upper, lower, momentum, tf)
    _df_ = pd.DataFrame(opt_result_dict.items())
    _df_.index = opt_result_dict.keys()
    _df_ = _df_.drop([0], axis=1)
    _df_.columns = ['values']
    _df_ = _df_.T
    print(_df_)
    var_stra = backtester.compute_strat_var(0.95)
    benchmark_stra = backtester.compute_benchmark_var(0.95)
    print(f'stra var:{round(var_stra, 4) * 100}%')
    print(f'benchmark var:{round(benchmark_stra, 4) * 100}%')

def get_return_ts(upper,lower,unit_tc,momentum,backtester,file_name,return_ts_folder):
    opt_result_dict = backtester.backtest_breakout_signal(
        upper, lower, unit_tc, is_long_exceeding_upper=momentum, cache_df=True)
    return_ts = backtester._return_ts()
    # file_name = file_name.replace('.csv', f',{upper},{lower}.csv')
    file_name = file_name.replace('.csv', f',{upper},{lower}.csv')
    print(file_name)
    return_ts_path = os.path.join(return_ts_folder, file_name)
    return_ts.to_csv(return_ts_path)
    # print(return_ts.tail(50))
    # print(return_ts.position.value_counts())


########################Plot graph########################
def plot_yoy_return(return_ts_folder,file_name,upper,lower):
    # Extract year and month from date
    file_name=file_name.replace('.csv',f',{upper},{lower}.csv')
    path=os.path.join(return_ts_folder,file_name)

    returns=pd.read_csv(path)
    returns['timestamp'] = pd.to_datetime(returns['timestamp'])
    returns['year'] = returns['timestamp'].dt.year
    returns['month'] = returns['timestamp'].dt.month
    returns['pnl']=returns['pnl']*100
    # Pivot data for year-on-year heatmap
    yearly_returns = returns.pivot_table(values='pnl', index='year', columns='month', aggfunc='sum')

    # Calculate year-on-year return
    yearly_returns['YOY_return'] = yearly_returns.sum(axis=1)
    # print((yearly_returns['YOY_return']).cumsum())
    cmap = sns.diverging_palette(10, 150, s=80, l=50, n=256)

    print(yearly_returns['YOY_return'] )

    yearly_returns = yearly_returns.drop(columns='YOY_return')

    # Plotting year-on-year heatmap with year-on-year return
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(yearly_returns, cmap=cmap, annot=True, fmt='.2f', linewidths=0.5,center=0)
    plt.title('Year-on-Year Returns Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Year')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.show()

def plot_sr_hm(good_csv_path:str):
    df_init = pd.read_csv(good_csv_path)
    # df = df_init.iloc[:, 1:]
    df = df_init.drop('Unnamed: 0', axis=1)
    print(df_init.drop('Unnamed: 0', axis=1).head(10))
    df = df[['upper_param', 'lower_param', 'sharpe_ratio']]
    df_X = df.pivot(index='upper_param', columns='lower_param', values='sharpe_ratio')
    f, ax = plt.subplots(figsize=(10, 10))
    plt.title(good_csv_path.split('/')[-1])
    sns.heatmap(df_X, annot=True, linewidths=.5, ax=ax, cmap='Spectral')
    plt.show()
    return

######################Plot graph after backtest##########################

def plot_eq_curve(upper,lower,unit_tc,momentum,backtester):
    # plot equity curve
    opt_result_dict = backtester.backtest_breakout_signal(
        upper, lower, unit_tc, is_long_exceeding_upper=momentum, cache_df=True)
    backtester.plot_equity_curve()
    plt.show()

def plot_reg(backtester,tf):
    backtester.plot_regression_result(tf=tf)




