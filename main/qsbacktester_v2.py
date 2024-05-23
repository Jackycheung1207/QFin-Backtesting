from pprint import pprint
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import plotly.figure_factory as ff
from datetime import datetime
import plotly.graph_objects as go
from loguru import logger
from decimal import Decimal, ROUND_HALF_UP
import itertools
from util.util_performance import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class Backtester:
    def __init__(self,
                 metric_type: str = 'price',metric_name: str = 'BTC' , coin: str ='BTC', tf: str = '1d',
                 is_momentum: bool = True, threshold: float = None, lookback_period: int= None,
                 denoise_logic: str = 'z', shift: int = 0, style:str = 'long&short' ,training_size: float=0.7,
                 exchange:str = 'binance', instrument: str = 'perp', unit_tc: float = 0.0006):
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.coin = coin
        self.tf = tf
        self.is_momentum = is_momentum
        self.threshold = threshold
        self.lookback_period = lookback_period
        self.denoise_logic = denoise_logic
        self.shift = shift
        self.style = style
        self.training_size = training_size
        self.exchange = exchange
        self.instrument = instrument
        self.unit_tc = unit_tc

        # Const
        START_DATE = '2020-06-01'
        END_DATE = '2024-03-31'

        price_folder_path = '../Data/Price'
        metric_folder_path = f'../Data/OnChain'
        result_folder_path = f'result/{self.metrics_name}'



    def assign_position(self,df: pd.DataFrame, threshold: float):
        """
        Assign positions to a dataframe based on the signal column.

        Args:
            df: The dataframe to assign positions to.
            threshold: The threshold value.

        Returns:
            The dataframe with the position column added.
        """
        if threshold is not None:
            if not self.isMomentum:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > threshold, -1,
                                          np.where(df['signal'] < -threshold, 1, 0))
            else:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > threshold, 1,
                                          np.where(df['signal'] < -threshold, -1, 0))
        else:
            if not self.isMomentum:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > 0, -1,
                                          np.where(df['signal'] < 0, 1, 0))
            else:
                # Assign positions based on the signal column
                df['position'] = np.where(df['signal'] > 0, 1,
                                          np.where(df['signal'] < 0, -1, 0))

        return df

    def denoise(self, df: pd.DataFrame, lookback_period: int):
        if self.denoise_logic == 'z':
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['sd'] = df['metric'].rolling(lookback_period).std()
            df['signal'] = (df['metric'] - df['ma']) / df['sd']

        elif self.denoise_logic == 'pct':
            df['signal'] = df['metric'].pct_change()

        elif self.denoise_logic == 'ma_diff':
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['signal'] = (df['metric'] / df['ma']) - 1

        elif self.denoise_logic == 'smac':
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            conditions = [
                (df['metric'] > df['ma']),
                (df['metric'] < df['ma']),
            ]

            choices = [1, -1]
            df['signal'] = np.select(conditions, choices, default=0)

        elif self.denoise_logic == 'ln+z':
            df['metric'] = np.log(df['metric'])
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['sd'] = df['metric'].rolling(lookback_period).std()
            df['signal'] = (df['metric'] - df['ma']) / df['sd']

        elif self.denoise_logic == 'pct+z':
            df['metric'] = df['metric'].pct_change()
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['sd'] = df['metric'].rolling(lookback_period).std()
            df['signal'] = (df['metric'] - df['ma']) / df['sd']

        elif self.denoise_logic == 'ln+pct+z':
            df['metric'] = np.log(df['metric'])
            df['metric'] = df['metric'].pct_change()
            df['ma'] = df['metric'].rolling(lookback_period).mean()
            df['sd'] = df['metric'].rolling(lookback_period).std()
            df['signal'] = (df['metric'] - df['ma']) / df['sd']

        elif self.denoise_logic == 'vol+z':
            df['returns'] = df['metric'].pct_change()
            df['volatility'] = df['returns'].rolling(lookback_period).std()
            df['mean_volatility'] = df['volatility'].rolling(lookback_period).mean()
            df['sd_volatility'] = df['volatility'].rolling(lookback_period).std()
            df['signal'] = (df['volatility'] - df['mean_volatility']) / df['sd_volatility']

        return df

    def compute_expected_return_(self,df: pd.DataFrame):
        if self.style == 'long&short':
            suffix = ''
        elif self.style == 'long_only':
            suffix = '_long_only'
        elif self.style == 'short_only':
            suffix = '_short_only'

        df['trades'] = df['position' + suffix] != df['position' + suffix].shift(1)

        df = df[df.trades]
        trades = len(df)
        df['p'] = df['cum_strategy_ret' + suffix] - df['cum_strategy_ret' + suffix].shift(1)
        df['wl'] = df.p > 0  # bool

        win_rate = (df.wl.value_counts().tolist()[0] / trades) * 100
        win_rate = round(win_rate, 2)
        lose_rate = 1 - win_rate
        lose_rate = round(lose_rate, 2)

        win_pnl = df[df.wl].p.mean() * 100
        lose_pnl = df[df.wl < 1].p.mean() * 100
        win_pnl = round(win_pnl, 2)
        lose_pnl = round(lose_pnl, 2)

        expected_return = round((win_rate * win_pnl + lose_rate * lose_pnl) / 100, 2)

        # print(f"""
        # Expected return={expected_return}%
        # win_rate={(win_rate)}
        # lose_rate={lose_rate}
        # win_pnl={win_pnl}
        # lose_pnl={lose_pnl}
        # """)
        return expected_return, win_rate, win_pnl, lose_pnl

    def get_price_df(self,price_folder_path: str):
        price_file_name = f'{self.coin}_{self.tf}_{self.exchange}_{self.instrument}.csv'
        logger.info(f"READ: {price_file_name}")
        price_df = pd.read_csv(
            os.path.join(price_folder_path, price_file_name)
        )[['timestamp', 'close']]
        # price_df = price_df.rename(columns={'close': 'price'})
        return price_df

    def get_price_metric_df(self,price_folder_path: str):
        price_file_name = f'{self.coin}_{self.tf}_{self.exchange}_{self.instrument}.csv'
        logger.info(f"READ: {price_file_name}")
        metric_df = pd.read_csv(
            os.path.join(price_folder_path, price_file_name)
        )[['timestamp', 'close']]
        metric_df = metric_df.rename(columns={'close': 'metric'})
        return metric_df

    def get_metric_df(self,metric_folder_path: str):
        if self.tf != '1d': tf = '1h'
        metric_file_name = f'{self.metrics_name}_{self.coin}_{self.tf}.csv'
        logger.info(f"READ: {metric_file_name}")
        metric_df = pd.read_csv(
            os.path.join(metric_folder_path, metric_file_name), index_col=0
        ).reset_index()
        metric_df = metric_df[['timestamp', self.metrics_name]]
        metric_df = metric_df.rename(columns={self.metrics_name: 'metric'})
        return metric_df

    def data_preprocessing(self,price_df: pd.DataFrame, metric_df: pd.DataFrame,
                           metrics_name: str, shift: int = 2,
                           START_DATE: str = '2020-06-01',
                           END_DATE: str = '2024-03-31'):

        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
        price_df = price_df[['timestamp', 'close']]
        price_df = price_df[(price_df['timestamp'] >= START_DATE) & (price_df['timestamp'] <= END_DATE)]
        # price_df.index = price_df.timestamp

        metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], utc=True)
        # metric_df = metric_df.rename(columns={metrics_name:'metric'})
        logger.debug(f'shift: {shift} bar')
        metric_df['metric'] = metric_df['metric'].shift(self.shift)  # TODO: shift 2 bar / hours
        metric_df = metric_df[(metric_df['timestamp'] > START_DATE) & (metric_df['timestamp'] <= END_DATE)]
        # metric_df.index = metric_df.timestamp
        df = pd.DataFrame()
        df = pd.merge(price_df, metric_df, left_on='timestamp', right_on='timestamp', how='left')
        # Find the index of the first non-NaN value in the 'metric' column
        first_valid_idx = df['metric'].first_valid_index()
        # Find the index of the last non-NaN value in the 'metric' column
        last_valid_idx = df['metric'].last_valid_index()
        # Drop leading and trailing NaN values from the DataFrame
        df = df.loc[first_valid_idx:last_valid_idx]
        df = df[['timestamp', 'close', 'metric']]
        df = df.set_index('timestamp')
        if df['close'].isna().sum() > 0:
            logger.warning(f"df close na count: {df['close'].isna().sum()}")
        if df['metric'].isna().sum() > 0:
            logger.warning(f"df metric na count: {df['metric'].isna().sum()}")

        return df

    def calculations(self,df: pd.DataFrame, threshold: float, isMomentum: bool,
                     unit_tc: float = 0.0006):
        df['position'] = 0
        df['ret'] = 0
        df['strategy_ret'] = 0
        df['cum_ret'] = 0
        df['transaction_cost'] = 0
        df['strategy_drawdown'] = 0
        df['benchmark_drawdown'] = 0

        df = self.assign_position(df, threshold, isMomentum)
        df['ret'] = df['close'].pct_change()
        df['benchmark_ret'] = df['close'].pct_change()
        df['cum_benchmark_ret'] = df['benchmark_ret'].cumsum()
        logger.debug(f"unit_tc: {unit_tc}")
        df['transaction_cost'] = abs(df['position'] - df['position'].shift(1)) * unit_tc
        df['strategy_ret'] = df['position'].shift(1) * df['ret'] - df['transaction_cost']
        df['cum_strategy_ret'] = df['strategy_ret'].cumsum()
        df['cum_strategy_ret'] = df['cum_strategy_ret'].fillna(0)
        #
        # Calculate the strategy drawdown series
        df['strategy_cumulative_max'] = df['cum_strategy_ret'].cummax()
        df['strategy_drawdown'] = df['cum_strategy_ret'] - df['strategy_cumulative_max']

        #
        # Calculate the benchmark drawdown series
        df['benchmark_cumulative_max'] = df['cum_benchmark_ret'].cummax()
        df['benchmark_drawdown'] = df['cum_benchmark_ret'] - df['benchmark_cumulative_max']

        return df

    def compute_kpi(self,df: pd.DataFrame, tf_int: int):
        kpi_dict = {}
        annualized_factor = 365 * 24 / tf_int
        logger.info(f"annualized factor: {annualized_factor}")

        strategy_sharpe = compute_sharpe_ratio(df['strategy_ret'], annualized_factor=annualized_factor)
        benchmark_sharpe = compute_sharpe_ratio(df['benchmark_ret'], annualized_factor=annualized_factor)

        strategy_mdd = compute_maximum_drawdown(df['strategy_ret'])
        benchmark_mdd = compute_maximum_drawdown(df['benchmark_ret'])

        strategy_ar = compute_annualized_return(df['strategy_ret'], annualized_factor=annualized_factor)
        benchamrk_ar = compute_annualized_return(df['benchmark_ret'], annualized_factor=annualized_factor)

        strategy_calmar = compute_calmar_ratio(df['strategy_ret'], annualized_factor=annualized_factor)
        benchamrk_calmar = compute_calmar_ratio(df['benchmark_ret'], annualized_factor=annualized_factor)

        strategy_recovery = compute_max_recovery_time(df['strategy_ret'])
        benchmark_recovery = compute_max_recovery_time(df['benchmark_ret'])

        strategy_t = perform_t_test(df['strategy_ret'], df['benchmark_ret'])

        alpha, beta = calculate_alpha_beta(df['strategy_ret'], df['benchmark_ret'])

        ############################################
        if (strategy_sharpe is not None):
            if (strategy_sharpe > benchmark_sharpe):
                logger.success(f"Sharpe: strategy={strategy_sharpe} || benchmark={benchmark_sharpe}")
                logger.success(f"MDD: strategy={strategy_mdd} || benchmark={benchmark_mdd}")

        ############################################
        expected_return, win_rate, win_pnl, lose_pnl = self.compute_expected_return_(df, style='long&short')
        long_dom = compute_long_dominance_by_duration(df, style='long&short')
        trading_number: float = abs(df['position']).sum()
        trading_freq: float = trading_number / len(df)

        position_counts = df['position'].value_counts().to_dict()
        # Create a mapping for the key replacements
        key_mapping = {
            0.0: 'no position',
            1.0: 'long',
            -1.0: 'short'
        }
        # Replace the keys in the position_counts dictionary
        position_counts = {key_mapping.get(key, key): value for key, value in position_counts.items()}
        # Calculate the sum of the values
        total_count = sum(position_counts.values())
        # Normalize the values by dividing by the sum and round to 4 decimal places
        position_counts_normalized = {
            key: Decimal(str(value / total_count)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            for key, value in position_counts.items()
        }

        kpi_dict = {
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,

            'strategy_mdd': strategy_mdd,
            'benchmark_mdd': benchmark_mdd,

            'strategy_ar': strategy_ar,
            'benchamrk_ar': benchamrk_ar,

            'strategy_calmar': strategy_calmar,
            'benchamrk_calmar': benchamrk_calmar,

            'strategy_recovery': strategy_recovery,
            'benchmark_recovery': benchmark_recovery,

            'strategy_t': strategy_t,
            'alpha': alpha,
            'beta': beta,
            ############################################
            'EV': expected_return,
            'win_rate': win_rate,
            'win_pnl': win_pnl,
            'lose_pnl': lose_pnl,

            'time_in_market': 1 - (position_counts_normalized.get('no position') or 0),
            'long_dominance_by_duration': long_dom,
        }
        if position_counts_normalized.get('no position') != 1:
            try:
                kpi_dict['long_dominance_count'] = round(position_counts_normalized.get('long') / (
                            position_counts_normalized.get('long') + position_counts_normalized.get('short')), 4)
            except:
                kpi_dict['long_dominance_count'] = 0
        else:
            kpi_dict['long_dominance_count'] = 0
        kpi_dict['trading_number'] = trading_number
        kpi_dict['trading_freq'] = trading_freq
        kpi_dict['duration'] = len(df)

        return kpi_dict

    def backtesting(self,coin: str, tf: str, metrics_name: str,
                    lookback_period: int, threshold: float, isMomentum: bool, denoise_logic: str, denoise=denoise,
                    metric_type: str = 'onchain', risk_free: float = 0.0,
                    isPlot: bool = False, train_size: float = 0.7, shift=None, unit_tc: float = 0.0006,
                    is_save_ts: bool = False):

        # unit_tc=0.0006
        # default_shift = 2
        tf_int = int(tf.replace('h', ''))
        tf_resample = tf.upper()
        exchange = 'binance'
        instrument = 'perp'
        price_folder_path = '../Data/Price'
        metric_folder_path = f'../Data/OnChain'

        if shift is None:
            if metric_type == 'price':
                shift = 1
            elif metric_type == 'onchain':
                shift = 2
            else:
                raise ValueError(f"metric_type: {metric_type} is not supported")

        price_df = self.get_price_df(price_folder_path, coin, tf, exchange, instrument)

        if metric_type == 'price':
            metric_df = self.get_price_metric_df(price_folder_path, coin, tf, exchange, instrument)
        elif metric_type == 'onchain':
            metric_df = self.get_metric_df(metric_folder_path, metrics_name, coin, tf)
        else:
            raise ValueError(f"metric_type: {metric_type} is not supported")

        df_clean = self.data_preprocessing(price_df, metric_df, metrics_name, shift=shift)

        split_timestamp = df_clean.index[int(len(df_clean) * train_size)]
        train_df = df_clean[:split_timestamp]
        test_df = df_clean[split_timestamp:]

        df_denoise = denoise(df=df_clean, denoise_logic=denoise_logic, lookback_period=lookback_period).dropna()
        df_denoise_train = denoise(df=train_df, denoise_logic=denoise_logic, lookback_period=lookback_period).dropna()
        df_denoise_test = denoise(df=test_df, denoise_logic=denoise_logic, lookback_period=lookback_period).dropna()

        df_denoise = self.calculations(df_denoise, threshold, isMomentum, unit_tc)
        df_denoise_train = self.calculations(df_denoise_train, threshold, isMomentum, unit_tc)
        df_denoise_test = self.calculations(df_denoise_test, threshold, isMomentum, unit_tc)

        if is_save_ts:
            file_name = f'{metrics_name}|{coin}|{tf}|{lookback_period}|{threshold}|{isMomentum}|{denoise_logic}.csv'
            time_series_folder_path = os.path.join('result', metrics_name, denoise_logic, 'time_series')

            if not os.path.exists(time_series_folder_path):
                os.makedirs(time_series_folder_path)
            df_denoise.to_csv(os.path.join(time_series_folder_path, file_name.replace('.csv', '_full.csv')))
            df_denoise_train.to_csv(os.path.join(time_series_folder_path, file_name.replace('.csv', '_train.csv')))
            df_denoise_test.to_csv(os.path.join(time_series_folder_path, file_name.replace('.csv', '_test.csv')))

        train_kpi_dict = self.compute_kpi(df_denoise_train, tf_int)
        test_kpi_dict = self.compute_kpi(df_denoise_test, tf_int)
        full_set_dict = self.compute_kpi(df_denoise, tf_int)

        train_kpi_dict['set'] = 'train'
        test_kpi_dict['set'] = 'test'
        full_set_dict['set'] = 'full'

        for i in [train_kpi_dict, test_kpi_dict, full_set_dict]:
            i['shift'] = shift
            i['unit_tc'] = unit_tc
            i['annulaized_factor'] = 365 * 24 / tf_int

        if isPlot:
            # # Plot for df_denoise_train
            # plt.figure(figsize=(10, 6))
            # df_denoise_train[['cum_strategy_ret', 'cum_benchmark_ret']].plot()
            # plt.title('Cumulative Returns - Train Set')
            # plt.xlabel('Date')
            # plt.ylabel('Cumulative Returns')
            # plt.legend(['Strategy Returns', 'Benchmark Returns'])
            # plt.tight_layout()
            # # Plot for df_denoise_test
            # plt.figure(figsize=(10, 6))
            # df_denoise_test[['cum_strategy_ret', 'cum_benchmark_ret']].plot()
            # plt.title('Cumulative Returns - Test Set')
            # plt.xlabel('Date')
            # plt.ylabel('Cumulative Returns')
            # plt.legend(['Strategy Returns', 'Benchmark Returns'])
            # plt.tight_layout()
            # # Plot for df_denoise
            # plt.figure(figsize=(10, 6))
            # df_denoise[['cum_strategy_ret', 'cum_benchmark_ret']].plot()
            # plt.title('Cumulative Returns - Full Dataset')
            # plt.xlabel('Date')
            # plt.ylabel('Cumulative Returns')
            # plt.legend(['Strategy Returns', 'Benchmark Returns'])
            # plt.tight_layout()

            plt.show()

            fig = make_subplots(rows=3, cols=3, subplot_titles=(
                'Return Time Series - Train', 'Return Time Series - Test', 'Return Time Series - Full',
                'Cumulative Returns - Train', 'Cumulative Returns - Test', 'Cumulative Returns - Full',
                'Drawdown - Train', 'Drawdown - Test', 'Drawdown - Full'))

            # Return Time Series - Train
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=1, col=1)

            # Return Time Series - Test
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=1, col=2)
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=1, col=2)

            # Return Time Series - Full
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['strategy_ret'], name='Strategy Returns',
                                     line=dict(color='blue')),
                          row=1, col=3)
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['benchmark_ret'], name='Benchmark Returns',
                                     line=dict(color='orange')),
                          row=1, col=3)

            # Cumulative Returns - Train
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['cum_strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=2, col=1)
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['cum_benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=2, col=1)

            # Cumulative Returns - Test
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['cum_strategy_ret'], name='Strategy Returns',
                           line=dict(color='blue')),
                row=2, col=2)
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['cum_benchmark_ret'], name='Benchmark Returns',
                           line=dict(color='orange')),
                row=2, col=2)

            # Cumulative Returns - Full
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['cum_strategy_ret'], name='Strategy Returns',
                                     line=dict(color='blue')),
                          row=2, col=3)
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['cum_benchmark_ret'], name='Benchmark Returns',
                                     line=dict(color='orange')),
                          row=2, col=3)

            # Drawdown - Train
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['strategy_drawdown'], name='Strategy Drawdown',
                           line=dict(color='blue')),
                row=3, col=1)
            fig.add_trace(
                go.Scatter(x=df_denoise_train.index, y=df_denoise_train['benchmark_drawdown'],
                           name='Benchmark Drawdown',
                           line=dict(color='orange')),
                row=3, col=1)

            # Drawdown - Test
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['strategy_drawdown'], name='Strategy Drawdown',
                           line=dict(color='blue')),
                row=3, col=2)
            fig.add_trace(
                go.Scatter(x=df_denoise_test.index, y=df_denoise_test['benchmark_drawdown'], name='Benchmark Drawdown',
                           line=dict(color='orange')),
                row=3, col=2)

            # Drawdown - Full
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['strategy_drawdown'], name='Strategy Drawdown',
                                     line=dict(color='blue')),
                          row=3, col=3)
            fig.add_trace(go.Scatter(x=df_denoise.index, y=df_denoise['benchmark_drawdown'], name='Benchmark Drawdown',
                                     line=dict(color='orange')),
                          row=3, col=3)

            # Update layout
            fig.update_layout(height=1080, width=1920,
                              title_text=f"Strategy vs Benchmark Performance (Training size: {train_size}) || params: {coin} | {tf} | {metrics_name} | {lookback_period} | {threshold} | {denoise_logic} | {isMomentum}",
                              legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1))
            fig.show()
        kpi_dict = {
            'train': train_kpi_dict,
            'test': test_kpi_dict,
            'full_set': full_set_dict
        }

        return train_kpi_dict, test_kpi_dict, full_set_dict

    def optimization(self,asset_list: list, metrics_name: str,
                     tf_list: list, lookback_period_list: list,
                     isMomentum_list: list, denoise_logic_list: list, threshold_list: Optional[List] = None,
                     denoise=denoise, metric_type: str = 'onchain', shift=None, unit_tc: float = 0.0006,
                     is_save_ts: bool = False):

        if threshold_list is None:
            if denoise_logic_list == ['z'] or denoise_logic_list == ['pct+z'] or denoise_logic_list == [
                'ln+z'] or denoise_logic_list == ['ln+pct+z'] or denoise_logic_list == [
                'vol+z'] or denoise_logic_list == ['power+z']:
                threshold_list = np.arange(0.1, 3.0, 0.1)
            elif denoise_logic_list == ['pct'] or denoise_logic_list == ['ma_diff']:
                threshold_list = np.arange(0.00, 0.2, 0.0025)
            elif denoise_logic_list == ['ma_diff_adj']:
                threshold_list = np.arange(0.05, 1.05, 0.05)
            elif denoise_logic_list == ['z_adj']:
                threshold_list = np.arange(0.1, 3.0, 0.1)
            elif denoise_logic_list == ['smac']:
                threshold_list = [0]

            logger.debug('Optimization with default threshold_list')

        # Create a list of all hyperparameter combinations
        hyperparameter_combinations = list(
            itertools.product(asset_list, tf_list, isMomentum_list, lookback_period_list, threshold_list,
                              denoise_logic_list))
        # print(pd.DataFrame(hyperparameter_combinations))
        logger.info(f"Total number of hyperparameter combinations: {len(hyperparameter_combinations)}")

        exchange = 'binance'
        instrument = 'perp'
        price_folder_path = '../Data/Price'
        metric_folder_path = f'../Data/OnChain'

        kpi_list = []
        train_kpi_dict_list = []
        test_kpi_dict_list = []
        full_set_dict_list = []

        # Iterate over each hyperparameter combination and perform backtesting
        for coin, tf, isMomentum, lookback_period, threshold, denoise_logic in hyperparameter_combinations:
            print('=' * 200)
            logger.info(
                f'coin:{coin}| tf:{tf}| isMomentum:{isMomentum}| lookback_period:{lookback_period}| threshold:{threshold}| denoise_logic:{denoise_logic}')

            if os.path.join(metrics_name, denoise_logic) not in os.listdir('../main/result'):
                result_folder_path = os.path.join('result', metrics_name, denoise_logic)
                if not os.path.exists(result_folder_path):
                    os.makedirs(result_folder_path)
                    logger.debug(f"Create folder: {result_folder_path}")
                else:
                    pass
                    # logger.debug(f"Folder already exists: {result_folder_path}")

            kpi = {}
            train_kpi_dict = {}
            test_kpi_dict = {}
            full_set_dict = {}

            train_kpi_dict, test_kpi_dict, full_set_dict = self.backtesting(coin, tf, metrics_name,
                                                                       metric_type=metric_type,
                                                                       isMomentum=isMomentum,
                                                                       lookback_period=lookback_period,
                                                                       threshold=threshold,
                                                                       denoise_logic=denoise_logic,
                                                                       denoise=denoise, isPlot=False, shift=shift,
                                                                       unit_tc=unit_tc, is_save_ts=False)

            train_kpi_dict['coin'] = coin
            train_kpi_dict['tf'] = tf
            train_kpi_dict['isMomentum'] = isMomentum
            train_kpi_dict['lookback_period'] = lookback_period
            train_kpi_dict['threshold'] = threshold
            train_kpi_dict['denoise_logic'] = denoise_logic
            train_kpi_dict_list.append(train_kpi_dict)

            test_kpi_dict['coin'] = coin
            test_kpi_dict['tf'] = tf
            test_kpi_dict['isMomentum'] = isMomentum
            test_kpi_dict['lookback_period'] = lookback_period
            test_kpi_dict['threshold'] = threshold
            test_kpi_dict['denoise_logic'] = denoise_logic
            test_kpi_dict_list.append(test_kpi_dict)

            full_set_dict['coin'] = coin
            full_set_dict['tf'] = tf
            full_set_dict['isMomentum'] = isMomentum
            full_set_dict['lookback_period'] = lookback_period
            full_set_dict['threshold'] = threshold
            full_set_dict['denoise_logic'] = denoise_logic
            full_set_dict_list.append(full_set_dict)

        print('+++' * 50)

        train_kpi_df = pd.DataFrame(train_kpi_dict_list).sort_values(by='strategy_sharpe', ascending=False)
        test_kpi_df = pd.DataFrame(test_kpi_dict_list).sort_values(by='strategy_sharpe', ascending=False)
        full_kpi_df = pd.DataFrame(full_set_dict_list).sort_values(by='strategy_sharpe', ascending=False)

        # return pd.DataFrame(kpi_list).sort_values(by='strategy_sharpe', ascending=False)
        return train_kpi_df, test_kpi_df, full_kpi_df

    def create_heatmap(self,
            metrics_name: str = 'balance_exchanges_relative',
            train_test_full: str = 'full', coin: str = 'BTC',
            tf: str = '8h', isMomentum: bool = False, denoise_logic: str = 'z',
            title=None, kpi: str = 'strategy_sharpe', ):

        result_folder_path = f'result/{metrics_name}'
        opt_df = pd.read_csv(f'{result_folder_path}/{denoise_logic}/performance_{denoise_logic}_{train_test_full}.csv',
                             index_col=0)

        df = opt_df[
            (opt_df.coin == coin) &
            (opt_df.tf == tf) &
            (opt_df.isMomentum == isMomentum) &
            (opt_df.denoise_logic == denoise_logic)
            ]
        heatmap_data = df.pivot(index='lookback_period', columns='threshold', values=kpi)
        annotations = heatmap_data.applymap(lambda x: f'{x:.2f}' if not pd.isnull(x) else '')
        custom_colorscale = [[0, 'red'],
                             [0.5, 'white'],
                             [0.8, 'orange'],
                             [1, 'green']]
        fig = ff.create_annotated_heatmap(
            z=heatmap_data.to_numpy(),
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            annotation_text=annotations.to_numpy(),
            colorscale=custom_colorscale,
            zmid=0.0,
            hoverinfo='z',
            showscale=True
        )
        if title:
            header = title
        else:
            header = f"{metrics_name}_{coin}_{tf}_{isMomentum}_{denoise_logic}_{train_test_full}"

        # Update layout
        fig.update_layout(
            title={
                # 'text': metrics_name+f'_{coin}_{tf}'+'_heatmap',
                'text': header,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='threshold',
                tickmode='array',
                tickvals=heatmap_data.columns.tolist(),
                ticktext=[str(val) for val in heatmap_data.columns.tolist()],
                side='bottom'
            ),
            yaxis=dict(
                title='lookback_period',
                tickmode='array',
                tickvals=heatmap_data.index.tolist(),
                ticktext=[str(val) for val in heatmap_data.index.tolist()]
            ),
            # width=1920,
            # height=1080,
            autosize=True
        )

        fig.update_xaxes(side='bottom')
        fig.show()



if __name__ == '__main__':
    backtester = Backtester(
        metric_type='price',
        metric_name='BTC',
        coin='BTC',
        tf='8h',
        is_momentum=True,
        threshold=None,
        lookback_period=None,
        denoise_logic='ma_diff',
        shift=0,
        scale='long&short',
        training_size=0.7
    )